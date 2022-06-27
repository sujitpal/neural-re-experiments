import argparse
import json
from lib2to3.pgen2 import token
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import re
import shutil
import torch
import typing

from datasets import load_dataset, ClassLabel
from functools import partial
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import AdamW

from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, 
    classification_report, accuracy_score
)
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, get_scheduler
)
from transformers.modeling_outputs import SequenceClassifierOutput


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def read_configuration() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="path to configuration file",
                        required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as fconf:
        conf = json.loads(fconf.read())
    return conf


def preprocess_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def reformat_json(infile: str, outfile: str, tokenizer: object) -> None:
    num_recs = 0
    fout = open(outfile, "w")
    with open(infile, "r") as fin:
        for line in fin:
            rec = json.loads(line.strip())
            text = preprocess_text(rec["text"])
            text_a = " ".join([tokenizer.cls_token, text, tokenizer.sep_token])
            label = rec["r"]
            output_rec = { "text": text_a, "label_s": label }
            fout.write(json.dumps(output_rec) + "\n")
            num_recs += 1
    logging.info("preprocessing: {:s} -> {:s} ({:d} records)".format(infile, outfile, num_recs))


def build_raw_dataset(conf: dict, tokenizer: object) -> object:
    prepdir = conf["prep_data_dir"]
    tempdir = conf["temp_data_dir"]
    shutil.rmtree(tempdir, ignore_errors=True)
    os.makedirs(tempdir)
    splits = ["train", "val", "test"]
    for split in splits:
        reformat_json(os.path.join(prepdir, "{:s}.jsonl".format(split)),
                      os.path.join(tempdir, "{:s}.json".format(split)),
                      tokenizer)
    data_files = {split: os.path.join(tempdir, "{:s}.json".format(split)) 
                  for split in splits}
    raw_dataset = load_dataset("json", data_files=data_files)
    return raw_dataset


def build_label_mappings(conf: dict) -> tuple:
    relations = []
    label_file = os.path.join(conf["prep_data_dir"], "relations.txt")
    with open(label_file, "r", encoding="utf-8") as frel:
        for line in frel:
            relations.append(line.strip())
    rel_tags = ClassLabel(names=relations)
    label2id = {name: rel_tags.str2int(name) for name in relations}
    id2label = {id: rel_tags.int2str(id) for id in range(len(relations))}
    return label2id, id2label, relations


def encode_data(conf: dict, label2id: dict, examples: list) -> object:
    tokenized_inputs = tokenizer(examples["text"], 
                                padding=True, truncation=True,
                                max_length=conf["max_length"])
    tokenized_inputs["label"] = [label2id[label] for label in examples["label_s"]]
    return tokenized_inputs


def build_encoded_dataset(raw_dataset: object, data_encoder_fn: object) -> object:
    enc_dataset = raw_dataset.map(data_encoder_fn, 
        batched=True, remove_columns=["text", "label_s"])
    return enc_dataset


def build_dataloaders(conf: dict, encoded_dataset: object,
                      tokenizer: object) -> tuple:
    collate_fn = DataCollatorWithPadding(tokenizer, padding="longest", 
                                         return_tensors="pt")
    batch_size = conf["batch_size"]
    if conf["test_mode"] == "true":
        train_dl = DataLoader(encoded_dataset["train"],
                              sampler=SubsetRandomSampler(
                                  np.random.randint(
                                    0, encoded_dataset["train"].num_rows, 
                                    1000).tolist()),
                              batch_size=batch_size, 
                              collate_fn=collate_fn)
        val_dl = DataLoader(encoded_dataset["val"], 
                            sampler=SubsetRandomSampler(
                                np.random.randint(
                                    0, encoded_dataset["val"].num_rows,
                                    200).tolist()),
                            batch_size=batch_size, 
                            collate_fn=collate_fn)
        test_dl = DataLoader(encoded_dataset["test"], 
                             sampler=SubsetRandomSampler(
                                np.random.randint(
                                    0, encoded_dataset["test"].num_rows, 
                                    100).tolist()),
                             batch_size=batch_size, 
                             collate_fn=collate_fn)
    else:
        train_dl = DataLoader(encoded_dataset["train"],
                              shuffle=True,
                              batch_size=batch_size, 
                              collate_fn=collate_fn)
        val_dl = DataLoader(encoded_dataset["val"], 
                            shuffle=False,
                            batch_size=batch_size, 
                            collate_fn=collate_fn)
        test_dl = DataLoader(encoded_dataset["test"], 
                             shuffle=False,
                             batch_size=batch_size, 
                             collate_fn=collate_fn)

    return train_dl, val_dl, test_dl


def create_model_dir(conf: dict) -> str:
    model_dir = os.path.join(conf["model_data_dir"], conf["target_model_name"])
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        os.makedirs(model_dir)
    return model_dir


def compute_accuracy(labels: list, logits: list) -> float:
    preds_cpu = torch.argmax(logits, dim=-1).cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    return accuracy_score(labels_cpu, preds_cpu)


def do_train(model: object, train_dl: object,
             device: object, optimizer: object, 
             lr_scheduler: object) -> float:
    train_loss = 0
    model.train()
    for bid, batch in enumerate(train_dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        train_loss += loss.detach().cpu().numpy()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return train_loss


def do_eval(model: object, eval_dl: object, 
            device: object) -> tuple:
    model.eval()
    eval_loss, eval_score, num_batches = 0, 0, 0
    for bid, batch in enumerate(eval_dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss

        eval_loss += loss.detach().cpu().numpy()
        eval_score += compute_accuracy(batch["labels"], outputs.logits)
        num_batches += 1

    eval_score /= num_batches
    return eval_loss, eval_score


def save_checkpoint(model: object, model_dir: str, epoch: int) -> None:
    model.save_pretrained(os.path.join(model_dir, "ckpt-{:d}".format(epoch)))


def save_training_history(history: list, model_dir: object, epoch: int) -> None:
    fhist = open(os.path.join(model_dir, "history.tsv"), "w")
    for epoch, train_loss, eval_loss, eval_score in history:
        fhist.write("{:d}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(
            epoch, train_loss, eval_loss, eval_score))
    fhist.close()


def save_training_plots(history: list, model_dir: str) -> None:
    plt.subplot(2, 1, 1)
    plt.plot([train_loss for _, train_loss, _, _ in history], label="train")
    plt.plot([eval_loss for _, _, eval_loss, _ in history], label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc="best")

    plt.subplot(2, 1, 2)
    plt.plot([eval_score for _, _, _, eval_score in history], label="validation")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc="best")

    plt.tight_layout()

    plt.savefig(os.path.join(model_dir, "training_plots.png"))


def generate_labels_and_predictions(model: object, test_dl: object) -> tuple:
    ytrue, ypred = [], []
    for batch in test_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            ytrue.extend(labels)
            ypred.extend(predictions)
    return ytrue, ypred


def save_evaluation_artifacts(test_accuracy: float, clf_report: str, 
                              conf_matrix: object, relations: list,
                              model_dir: str) -> None:
    with open(os.path.join(model_dir, "eval-report.txt"), "w", encoding="utf-8") as fout:
        fout.write("** test accuracy: {:.3f}\n\n".format(test_accuracy))
        fout.write("** classification report\n")
        fout.write(clf_report + "\n")
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=relations)
    disp.plot(cmap="Blues", values_format="0.2f", ax=ax, colorbar=False)
    plt.title("Normalized Confusion Matrix")    
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))



logging.basicConfig(level=logging.INFO)

set_random_seed(42)
conf = read_configuration()

base_model_name = conf["base_model_name"]

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
raw_dataset = build_raw_dataset(conf, tokenizer)
label2id, id2label, relations = build_label_mappings(conf)

data_encoder_fn = partial(encode_data, conf, label2id)
encoded_dataset = build_encoded_dataset(raw_dataset, data_encoder_fn)

train_dl, val_dl, test_dl = build_dataloaders(conf, encoded_dataset, tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(base_model_name, num_labels=len(relations))
model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name, config=config)
model = model.to(device)

optimizer = AdamW(model.parameters(), 
                 lr=conf["learning_rate"],
                 weight_decay=conf["weight_decay"])

num_epochs = 1 if conf["test_mode"] == "true" else conf["num_epochs"]
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler("linear",
                             optimizer=optimizer,
                             num_warmup_steps=0,
                             num_training_steps=num_training_steps)

model_dir = create_model_dir(conf)
history = []
for epoch in range(num_epochs):
    train_loss = do_train(model, train_dl, device, optimizer, lr_scheduler)
    eval_loss, eval_score = do_eval(model, val_dl, device)
    history.append((epoch + 1, train_loss, eval_loss, eval_score))
    logging.info("EPOCH {:d}, train loss: {:.3f}, val loss: {:.3f}, val-acc: {:.5f}".format(
        epoch + 1, train_loss, eval_loss, eval_score))
    save_checkpoint(model, model_dir, epoch + 1)
    save_training_history(history, model_dir, epoch + 1)

save_training_plots(history, model_dir)

# evaluation
ytrue, ypred = generate_labels_and_predictions(model, test_dl)

test_accuracy = accuracy_score(ytrue, ypred)
print("test accuracy: {:.3f}".format(test_accuracy))

clf_report = classification_report(ytrue, ypred, target_names=relations)
print(clf_report)

conf_matrix = confusion_matrix(ytrue, ypred, normalize="true")
print(conf_matrix)

save_evaluation_artifacts(test_accuracy, clf_report, conf_matrix, 
                          relations, model_dir)