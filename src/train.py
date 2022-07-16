import argparse
from codecs import ignore_errors
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import torch

from functools import partial
from torch.optim import AdamW

from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, 
    classification_report, accuracy_score
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler

import dataprep
import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



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


def do_train(model: object,
             train_dl: DataLoader,
             device: torch.cuda.device,
             optimizer: torch.optim.Optimizer, 
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


def do_eval(model: object,
            eval_dl: DataLoader, 
            device: torch.cuda.device) -> tuple:
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


def save_checkpoint(model: object,
                    tokenizer: AutoTokenizer,
                    model_dir: str,
                    epoch: int) -> None:
    if isinstance(model, torch.nn.Module):
        ckpt_dir = os.path.join(model_dir, "ckpt-{:d}".format(epoch))
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        torch.save(model, os.path.join(ckpt_dir, "model.pt"))
        tokenizer_dir = os.path.join(model_dir, "tokenizer")
        if not os.path.exists(tokenizer_dir):
            tokenizer.save_pretrained(tokenizer_dir)
    else:
        model.save_pretrained(os.path.join(model_dir, "ckpt-{:d}".format(epoch)))
        tokenizer.save_pretrained(os.path.join(model_dir, "ckpt-{:d}".format(epoch)))


def save_training_history(history: list, model_dir: object) -> None:
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


def save_evaluation_artifacts(test_accuracy: float,
                              clf_report: str, 
                              conf_matrix: object,
                              relations: list,
                              model_dir: str) -> None:
    with open(os.path.join(model_dir, "eval-report.txt"), "w", encoding="utf-8") as fout:
        fout.write("** test accuracy: {:.3f}\n\n".format(test_accuracy))
        fout.write("** classification report\n")
        fout.write(clf_report + "\n")
    _, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=relations)
    disp.plot(cmap="Blues", values_format="0.2f", ax=ax, colorbar=False)
    plt.title("Normalized Confusion Matrix")    
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))


################################## main ##################################

if __name__ == "__main__":
    set_random_seed(42)

    conf = read_configuration()

    base_model_name = conf["base_model_name"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    prep_dir = conf["prep_data_dir"]
    temp_dir = conf["temp_data_dir"]

    raw_dataset = dataprep.build_raw_dataset(
        prep_dir, temp_dir, tokenizer,
        tag_type=conf["mention_tag_type"])

    label2id, id2label, relations = dataprep.build_label_mappings(
        conf["prep_data_dir"])

    data_encoder_fn = partial(dataprep.encode_data, 
        label2id=label2id, 
        tokenizer=tokenizer,
        max_length=conf["max_length"],
        tags_added_to_text=conf["mention_tag_added_to_text"],
        mention_token_ids_src=conf["mention_token_ids_src"],
        position_embedding=conf["mention_position_embedding"])
    if conf["model_pattern"] == "cls":
        enc_dataset = raw_dataset.map(
            data_encoder_fn, batched=True,
            remove_columns=conf["raw_columns_to_remove"])
    else:
        enc_dataset = (raw_dataset
            .map(
                data_encoder_fn, batched=True,
                remove_columns=conf["raw_columns_to_remove"])
            .filter(lambda x: x["mention_token_ids"] != [-1, -1, -1, -1])
        )

    train_dl, val_dl, test_dl = dataprep.build_dataloaders(
        enc_dataset, tokenizer, conf["batch_size"], conf["test_mode"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_fn = models.MODEL_FACTORY[conf["model_pattern"]]
    model = model_fn(
        base_model_name, num_labels=len(relations), 
        vocab_size=len(tokenizer.vocab))
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
        logger.info("EPOCH {:d}, train loss: {:.3f}, val loss: {:.3f}, val-acc: {:.5f}".format(
            epoch + 1, train_loss, eval_loss, eval_score))
        save_checkpoint(model, tokenizer, model_dir, epoch + 1)

    save_training_history(history, model_dir)
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
