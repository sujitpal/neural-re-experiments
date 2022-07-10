import json
import logging
import numpy as np
import operator
import os
import re
import shutil

from transformers import AutoTokenizer
from datasets import ClassLabel, DatasetDict, load_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def compute_entity_token_positions(text: str, 
                                   head_ent_type: str,
                                   tail_ent_type: str) -> list:
    tokens = text.split()
    head_start = [i for i, token in enumerate(tokens)
                  if token.startswith("<{:s}>".format(head_ent_type))][0]
    head_end = [i for i, token in enumerate(tokens)
                if token.endswith("</{:s}>".format(head_ent_type))][0]
    tail_start = [i for i, token in enumerate(tokens)
                  if token.startswith("<{:s}>".format(tail_ent_type))][0]
    tail_end = [i for i, token in enumerate(tokens)
                if token.endswith("</{:s}>".format(tail_ent_type))][0]
    return [head_start, head_end, tail_start, tail_end]


def reformat_prepared_jsonl(jsonl_str: str, 
                            tokenizer: AutoTokenizer,
                            tag_type: str = None) -> dict:
    rec = json.loads(jsonl_str, strict=False)
    # print(json.dumps(rec, indent=2))

    text = rec["text"]
    label = rec["r"]

    # insert the tags
    head_ent, tail_ent = rec["h"], rec["t"]
    ent_spans = [(head_ent["start"], head_ent["end"], head_ent["type"]),
                 (tail_ent["start"], tail_ent["end"], tail_ent["type"])]
    ent_spans = sorted(ent_spans, key=operator.itemgetter(0))
    # print("ent_spans:", ent_spans)
    
    span_pts = [0]
    for start, end, _ in ent_spans:
        span_pts.append(start)
        span_pts.append(end)
    span_pts.append(len(text))
    span_ranges = [(span_pts[i], span_pts[i+1]) for i in range(len(span_pts) - 1)]
    # print("span_ranges:", span_ranges)
    splits = [text[start:end] for start, end in span_ranges]
    # print("splits:", splits)

    head_ent_type, tail_ent_type = "E1", "E2"
    if tag_type == "entity_type":
        head_ent_type, tail_ent_type = [span[2] for span in ent_spans]
        head_ent_type = "E1:" + head_ent_type
        tail_ent_type = "E2:" + tail_ent_type

    splits[1] = " <{:s}>".format(head_ent_type) + splits[1] + "</{:s}> ".format(head_ent_type)
    splits[3] = " <{:s}>".format(tail_ent_type) + splits[3] + "</{:s}> ".format(tail_ent_type)
    text_em = "".join(splits)

    # add the [CLS] and [SEP] tags to text
    text_a = " ".join([tokenizer.cls_token, text_em, tokenizer.sep_token])
    text_a = preprocess_text(text_a)
    # print("text_a:", text_a)

    # compute entity token positions
    mention_token_ids = compute_entity_token_positions(text_a, head_ent_type, tail_ent_type)

    # if tag_type is None, remove entity tags from text
    if tag_type is None:
        text_a = re.sub("<.*?>", " ", text_a)
        text_a = re.sub("<\/.*?>", " ", text_a)
        text_a = re.sub("\s+", " ", text_a)

    output_rec = {
        "text": text_a,
        "rel_label": label,
        "mention_token_ids": mention_token_ids
    }
    return output_rec


def reformat_jsonl_file(jsonl_input_fp: str,
                        json_output_fp: str,
                        tokenizer: AutoTokenizer,
                        tag_type: str = None) -> int:
    fprep = open(jsonl_input_fp, "r")
    ftemp = open(json_output_fp, "w")
    num_reformatted = 0
    for line in fprep:
        jsonl_str = line.strip()
        output_rec = reformat_prepared_jsonl(jsonl_str, tokenizer, tag_type=tag_type)
        ftemp.write(json.dumps(output_rec) + "\n")
        num_reformatted += 1

    fprep.close()
    ftemp.close()
    return num_reformatted


def build_raw_dataset(prep_dir: str,
                      temp_dir: str,
                      tokenizer: AutoTokenizer,
                      tag_type: str = None) -> DatasetDict:
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir)
    splits = ["train", "val", "test"]
    for split in splits:
        prep_fp = os.path.join(prep_dir, "{:s}.jsonl".format(split))
        temp_fp = os.path.join(temp_dir, "{:s}.json".format(split))
        num_reformatted = reformat_jsonl_file(
            prep_fp, temp_fp, tokenizer, tag_type=tag_type)
        logger.info("reformatting {:s} -> {:s} ({:d} records)".format(
            prep_fp, temp_fp, num_reformatted))

    data_files = {split: os.path.join(temp_dir, "{:s}.json".format(split)) 
                  for split in splits}
    raw_dataset = load_dataset("json", data_files=data_files)
    return raw_dataset


def build_label_mappings(labels_fp: str) -> tuple:
    relations = []
    with open(labels_fp, "r", encoding="utf-8") as frel:
        for line in frel:
            relations.append(line.strip())
    rel_tags = ClassLabel(names=relations)
    label2id = {name: rel_tags.str2int(name) for name in relations}
    id2label = {id: rel_tags.int2str(id) for id in range(len(relations))}
    return label2id, id2label, relations


def encode_data(examples: list,
                label2id: dict,
                tokenizer: AutoTokenizer,
                max_length: int,
                align_mention_token_ids: bool = False,
                update_token_type_ids: bool = False) -> dict:
    tokenized_inputs = tokenizer(examples["text"], 
                                 padding=True,
                                 truncation=True,
                                 max_length=max_length)
    tokenized_inputs["label"] = [label2id[label] for label in examples["rel_label"]]
    if align_mention_token_ids:
        aligned_mention_token_ids, updated_token_type_ids = [], []
        mention_token_ids = examples["mention_token_ids"]
        token_type_ids = tokenized_inputs["token_type_ids"]
        for i, (e1_start, e1_end, e2_start, e2_end) in enumerate(mention_token_ids):
            word_ids = tokenized_inputs.word_ids(i)
            try:
                aligned_mention_token_id = [
                    min([ix for ix, wid in enumerate(word_ids) if wid == e1_start]),
                    max([ix+1 for ix, wid in enumerate(word_ids) if wid == e1_end]),
                    min([ix for ix, wid in enumerate(word_ids) if wid == e2_start]),
                    max([ix+1 for ix, wid in enumerate(word_ids) if wid == e2_end])
                ]
                aligned_mention_token_ids.append(aligned_mention_token_id)
            except ValueError:
                # can happen if spans occur outside max_length
                align_mention_token_ids.append([0, 0, 0, 0])
            if update_token_type_ids:
                token_type_id = np.array(token_type_ids[i])
                token_type_id[aligned_mention_token_id[0]:aligned_mention_token_id[1]] = 1
                token_type_id[aligned_mention_token_id[2]:aligned_mention_token_id[3]] = 1
                updated_token_type_ids.append(token_type_id)

        tokenized_inputs["mention_token_ids"] = align_mention_token_ids
        if update_token_type_ids:
            tokenized_inputs["token_type_ids"] = updated_token_type_ids

    return tokenized_inputs
