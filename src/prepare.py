import json
import logging
import os
import typing

from sklearn.model_selection import train_test_split


def parse_text(file_path: str) -> str:
    """Reads text file into string.

    Parameters:
    file_path (str): path to file to read

    Returns:
    str: Text contents of file

    """
    with open(file_path, "r", encoding="utf-8") as ftext:
        text = ftext.read()
    return text


def parse_entity_and_relation(file_path: str, text: str) -> tuple:
    """Read BRAT style annotation file and create one or more SpaCy
       style relation JSON records.

       Parameters:
       file_path (str): path of annotation file to read
       text (str): text on which annotation is computed

       Returns:
       tuple:list of (SpaCy style) structured records, each representing
            a relation triple, and list of entity types

    """
    entity_dict, recs, entity_types = {}, [], []
    with open(file_path, "r") as fann:
        for line in fann:
            line = line.strip()
            if line.startswith("T"):
                # entities
                eid, etype_grp, espan = line.split("\t")
                etype, estart, eend = etype_grp.split()
                estart = int(estart)
                eend = int(eend)
                entity_dict[eid] = (etype, estart, eend, espan)
                entity_types.append(etype)
            elif line.startswith("R"):
                # relations
                rid, rgrp = line.split("\t")
                rtype, rsub_grp, robj_grp = rgrp.split()
                if rsub_grp.startswith("Arg1"):
                    rsub_type, rsub_start, rsub_end, _ = entity_dict[rsub_grp.split(":")[1]]
                    robj_type, robj_start, robj_end, _ = entity_dict[robj_grp.split(":")[1]]
                else:
                    rsub_type, rsub_start, rsub_end, _ = entity_dict[robj_grp.split(":")[1]]
                    robj_type, robj_start, robj_end, _ = entity_dict[rsub_grp.split(":")[1]]
                rec = {
                    "text": text,
                    "h": {
                        "type": rsub_type,
                        "start": rsub_start,
                        "end": rsub_end,
                        "span": text[rsub_start : rsub_end],
                    },
                    "t": {
                        "type": robj_type,
                        "start": robj_start,
                        "end": robj_end,
                        "span": text[robj_start : robj_end],
                    },
                    "r": rtype
                }
                recs.append(rec)
            else:
                pass

    return recs, entity_types


def write_data(data: list, file_path: str) -> int:
    """Write list of relation records into specified file_path

    Parameters:
    data (list): list of relation records to write.
    file_path (str): path of JSON-L file to write records to.
    
    Returns:
    int: number of records written

    """
    num_written = 0
    with open(file_path, "w", encoding="utf-8") as fout:
        for rec in data:
            fout.write(json.dumps(rec) + "\n")
            num_written += 1
    return num_written


def write_labels(labels: list, label_path: str) -> int:
    relations = sorted(list(set(labels)))
    with open(label_path, "w", encoding="utf-8") as flab:
        for relation in relations:
            flab.write(relation + "\n")
    return len(relations)


def write_entity_types(entity_types: list, entity_path: str) -> list:
    entity_types = sorted(list(set(entity_types)))
    with open(entity_path, "w", encoding="utf-8") as fent:
        for entity_type in entity_types:
            fent.write(entity_type + "\n")
    return len(entity_types)


if __name__ == "__main__":

    DATA_DIR = "../data"

    logging.basicConfig(level=logging.INFO)

    raw_data_dir = os.path.join(DATA_DIR, "raw")
    txt_files = [fn for fn in os.listdir(raw_data_dir) if fn.endswith(".txt")]
    ann_files = [fn.replace(".txt", ".ann") for fn in txt_files]

    data, labels, entity_types = [], [], []
    for txt_file, ann_file in zip(txt_files, ann_files):
        text = parse_text(os.path.join(raw_data_dir, txt_file))
        recs, etypes = parse_entity_and_relation(
            os.path.join(raw_data_dir, ann_file), text)
        data.extend(recs)
        labels.extend([rec["r"] for rec in recs])
        entity_types.extend(etypes)

    assert len(data) == len(labels)
    logging.info("read: #-records: {:d}".format(len(data)))

    data_tv, data_test, labels_tv, labels_test = train_test_split(
        data, labels, stratify=labels, train_size=0.8, random_state=0)
    data_train, data_val, labels_train, labels_val = train_test_split(
        data_tv, labels_tv, stratify=labels_tv, train_size=0.9, random_state=0)

    logging.info("split: train: {:d}, validation: {:d}, test: {:d}".format(
        len(data_train), len(data_val), len(data_test)))

    prepared_data_dir = os.path.join(DATA_DIR, "prepared")
    num_train = write_data(data_train, os.path.join(prepared_data_dir, "train.jsonl"))
    num_val = write_data(data_val, os.path.join(prepared_data_dir, "val.jsonl"))
    num_test = write_data(data_test, os.path.join(prepared_data_dir, "test.jsonl"))

    logging.info("written: train: {:d}, validation: {:d}, test: {:d}".format(
        num_train, num_val, num_test))

    num_labels = write_labels(labels, os.path.join(prepared_data_dir, "relations.txt"))
    logging.info("written: {:d} labels".format(num_labels))

    num_entities = write_entity_types(
        entity_types, os.path.join(prepared_data_dir, "entities.txt"))
    logging.info("written: {:d} entity types".format(num_entities))
