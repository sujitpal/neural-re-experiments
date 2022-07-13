import json
import logging
import numpy as np
import operator
import os
import re
import shutil

from datasets import ClassLabel, DatasetDict, load_dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer, DataCollatorWithPadding


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

    # - if tag_type == None -- no positional information needed
    # - if tag_type == positional -- positional information only, no tags
    # - if tag_type == entity -- entity markers E1 and E2
    # - if tag_type == entity_type -- entity type markers (relations)
    if tag_type is not None:
        if tag_type == "positional":
            # we calculate the entity positions and return this information
            splits[1] = " >>{:s}<< ".format(splits[1])
            splits[3] = " >>{:s}<< ".format(splits[3])
        elif tag_type == "entity":
            head_ent_type, tail_ent_type = "E1", "E2"
            splits[1] = " <{:s}>".format(head_ent_type) + splits[1] + "</{:s}> ".format(head_ent_type)
            splits[3] = " <{:s}>".format(tail_ent_type) + splits[3] + "</{:s}> ".format(tail_ent_type)
        elif tag_type == "entity_type":
            head_ent_type, tail_ent_type = [span[2] for span in ent_spans]
            head_ent_type = "E1:" + head_ent_type
            tail_ent_type = "E2:" + tail_ent_type
            splits[1] = " <{:s}>".format(head_ent_type) + splits[1] + "</{:s}> ".format(head_ent_type)
            splits[3] = " <{:s}>".format(tail_ent_type) + splits[3] + "</{:s}> ".format(tail_ent_type)
        else:
            pass

    text_em = "".join(splits)

    # add the [CLS] and [SEP] tags to text
    text_a = " ".join([tokenizer.cls_token, text_em, tokenizer.sep_token])
    text_a = preprocess_text(text_a)

    if tag_type == "positional":
        tokens = text_a.split()
        starts = [i for i, token in enumerate(tokens) if token.startswith(">>")]
        ends = [i for i, token in enumerate(tokens) if token.endswith("<<")]
        raw_mention_token_ids = [starts[0], ends[0], starts[1], ends[1]]
        text_a = text_a.replace(">>", " ").replace("<<", " ")
        text_a = re.sub(r"\s+", " ", text_a)
        output_rec = {
            "text": text_a,
            "rel_label": label,
            "raw_mention_token_ids": raw_mention_token_ids
        }
    else:
        output_rec = {
            "text": text_a,
            "rel_label": label,
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


def build_label_mappings(prep_dir: str) -> tuple:
    relations = []
    labels_fp = os.path.join(prep_dir, "relations.txt")
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
                tags_added_to_text: str = None,
                mention_token_ids_src: str = None,
                position_embedding: bool = False) -> dict:

    # - if tag_type is None, skip vocab add and post-compute positions
    # - if tag_type is positional, use raw_mention_token_ids to compute
    #   post-toknization positions and output into mention_token_ids
    # - if tag_type is entity or entity_type, add tags to tokenizer vocab
    #   and compute mention_token_ids using tokenizer
    # add mention tokens to tokenizer vocabulary
    if tags_added_to_text == "entity" or tags_added_to_text == "entity_type":
        tag_tokens = []
        if tags_added_to_text == "entity_type":
            for relation in label2id.keys():
                for prefix in ["E1", "E2"]:
                    tag_tokens.append("<{:s}:{:s}>".format(prefix, relation))
                    tag_tokens.append("</{:s}:{:s}>".format(prefix, relation))
        else:
            for relation in ["E1", "E2"]:
                tag_tokens.append("<{:s}>".format(relation))
                tag_tokens.append("</{:s}>".format(relation))
        tokenizer.add_tokens(tag_tokens)

    # tokenize input
    tokenized_inputs = tokenizer(examples["text"], 
                                 padding=True,
                                 truncation=True,
                                 max_length=max_length)
    tokenized_inputs["label"] = [label2id[label] for label in examples["rel_label"]]

    # compute mention token positions
    if mention_token_ids_src is not None:
        mention_token_ids = []
        if mention_token_ids_src == "tokenizer":
            # use tags to compute token positions
            tag_token_ids = set(tokenizer.convert_tokens_to_ids(tag_tokens))
            for i in range(len(tokenized_inputs.input_ids)):
                try:
                    mtis = [j for j, x in enumerate(tokenized_inputs.input_ids[i])
                            if x in tag_token_ids]
                    if len(mtis) != 4:
                        mtis = [-1, -1, -1, -1]
                except IndexError:
                    mtis = [-1, -1, -1]
                mention_token_ids.append(mtis)
        elif mention_token_ids_src == "raw":
            # use raw_mention_token_ids and tokenized_inputs.word_ids() to
            # compute token positions
            raw_mention_token_ids = examples["raw_mention_token_ids"]
            for i, (e1_start, e1_end, e2_start, e2_end) in enumerate(raw_mention_token_ids):
                word_ids = tokenized_inputs.word_ids(i)
                try:
                    mention_token_ids.append([
                        min([ix for ix, wid in enumerate(word_ids) if wid == e1_start]),
                        max([ix+1 for ix, wid in enumerate(word_ids) if wid == e1_end]),
                        min([ix for ix, wid in enumerate(word_ids) if wid == e2_start]),
                        max([ix+1 for ix, wid in enumerate(word_ids) if wid == e2_end])
                    ])
                except ValueError:
                    mention_token_ids.append([-1, -1, -1, -1])

        if len(mention_token_ids) > 0:
            tokenized_inputs["mention_token_ids"] = mention_token_ids

    if mention_token_ids_src is not None and position_embedding:
        token_type_ids_upd = []
        token_type_ids = tokenized_inputs.token_type_ids
        for i in range(len(tokenized_inputs.input_ids)):
            mtis = mention_token_ids[i]
            token_type_id = np.array(token_type_ids[i])
            token_type_id[mtis[0]:mtis[1]] = 1
            token_type_id[mtis[2]:mtis[3]] = 1
            token_type_ids_upd.append(token_type_id)
        tokenized_inputs["token_type_ids"] = token_type_ids_upd

    return tokenized_inputs


# DATAPREP_FACTORY = {
#     "standard": {
#         "tag_type": None,
#         "align_mention_token_ids": False,
#         "update_token_type_ids": False,
#         "remove_columns": ["text", "rel_label", "mention_token_ids"]
#     },
#     "standard_pos": {
#         "tag_type": None,
#         "align_mention_token_ids": False,
#         "update_token_type_ids": False,
#         "remove_columns": ["text", "rel_label", "mention_token_ids"]
#     },
#     "positional_embedding": {
#         "tag_type": None,
#         "align_mention_token_ids": False,
#         "update_token_type_ids": False,
#         "remove_columns": ["text", "rel_label", "mention_token_ids"]
#     },
#     "entity_marker": {
#         "tag_type": None,
#         "align_mention_token_ids": False,
#         "update_token_type_ids": False,
#         "remove_columns": ["text", "rel_label", "mention_token_ids"]
#     },
#     "entity_type_marker": {
#         "tag_type": None,
#         "align_mention_token_ids": False,
#         "update_token_type_ids": False,
#         "remove_columns": ["text", "rel_label", "mention_token_ids"]
#     }
# }


def build_dataloader_for_split(enc_dataset: DatasetDict, 
                      tokenizer: AutoTokenizer,
                      split: str,
                      batch_size: int,
                      test_mode: bool) -> DataLoader:
    collate_fn = DataCollatorWithPadding(tokenizer, padding="longest", 
                                         return_tensors="pt")
    if test_mode:
        sampler = SubsetRandomSampler(
            np.random.randint(0, enc_dataset[split].num_rows, 1000).tolist())
        return DataLoader(enc_dataset[split],
                          batch_size=batch_size, 
                          sampler=sampler, 
                          collate_fn=collate_fn)
    else:
        return DataLoader(enc_dataset[split],
                          batch_size=batch_size, 
                          shuffle=True, 
                          collate_fn=collate_fn)


def build_dataloaders(enc_dataset: DatasetDict, 
                      tokenizer: AutoTokenizer,
                      batch_size: int,
                      test_mode: str = False) -> tuple:
    dataloaders = []
    for split in ["train", "val", "test"]:
        dataloaders.append(
            build_dataloader_for_split(enc_dataset, tokenizer, split, 
                                       batch_size, test_mode=test_mode))
    return dataloaders
