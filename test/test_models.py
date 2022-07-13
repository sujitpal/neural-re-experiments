# call from parent directory: python -m unittest test/test_models.py 
import os
from tkinter import W
import unittest

from functools import partial

from src.dataprep import (
    build_raw_dataset, build_label_mappings,
    encode_data, build_dataloader_for_split
)
from src.models import *

class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.prep_dir = "data/prepared"
        cls.temp_dir = "data/temp"
        cls.base_model_name = "bert-base-cased"
        cls.max_length = 400
        cls.batch_size = 32
        cls.test_mode = True
        cls.num_classes = 8

    def _build_dataset_for_task(self, tokenizer, 
                               tag_type_raw, tag_type_enc,
                               mention_token_id_src, position_embedding,
                               remove_raw_columns):
        raw_dataset = build_raw_dataset(
            TestModels.prep_dir, TestModels.temp_dir, tokenizer,
            tag_type=tag_type_raw)
        label2id, _, _ = build_label_mappings(TestModels.prep_dir)

        data_encoder_fn = partial(encode_data, 
            label2id=label2id, 
            tokenizer=tokenizer,
            max_length=TestModels.max_length,
            tags_added_to_text=tag_type_enc,
            mention_token_ids_src=mention_token_id_src,
            position_embedding=position_embedding)
        enc_dataset = raw_dataset.map(
            data_encoder_fn, batched=True,
            remove_columns=remove_raw_columns)

        train_dl = build_dataloader_for_split(
            enc_dataset, tokenizer, "train", 
            batch_size=TestModels.batch_size, test_mode=TestModels.test_mode)
        return train_dl

    def _set_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device


    def test_get_and_run_standard_cls_model(self):
        tokenizer = AutoTokenizer.from_pretrained(TestModels.base_model_name)
        train_dl = self._build_dataset_for_task(
            tokenizer, None, None, None, False, ["text", "rel_label"])
        
        model = cls_model(
            TestModels.base_model_name, TestModels.num_classes, None)
        self.assertIsNotNone(model)
        device = self._set_device()
        model = model.to(device)
        
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            break
        self.assertIsNotNone(outputs)
        self.assertIsInstance(outputs, SequenceClassifierOutput)
        self.assertEqual(len(outputs.keys()), 2)
        self.assertTrue("loss" in outputs.keys())
        self.assertTrue("logits" in outputs.keys())
        self.assertIsInstance(outputs.loss.detach().cpu().numpy().item(), float)
        self.assertEqual(outputs.logits.detach().cpu().numpy().shape, 
            (TestModels.batch_size, TestModels.num_classes))

    def test_get_and_run_standard_mention_pooling_model(self):
        tokenizer = AutoTokenizer.from_pretrained(TestModels.base_model_name)
        train_dl = self._build_dataset_for_task(
            tokenizer, "positional", None, "raw", False, 
            ["text", "rel_label", "raw_mention_token_ids"])

        model = mention_pooling_model(
            TestModels.base_model_name, TestModels.num_classes, None)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, nn.Module)
        device = self._set_device()
        model = model.to(device)

        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            break
        self.assertIsNotNone(outputs)
        self.assertIsInstance(outputs, SequenceClassifierOutput)
        self.assertEqual(len(outputs.keys()), 2)
        self.assertTrue("loss" in outputs.keys())
        self.assertTrue("logits" in outputs.keys())
        self.assertIsInstance(outputs.loss.detach().cpu().numpy().item(), float)
        self.assertEqual(outputs.logits.detach().cpu().numpy().shape, 
            (TestModels.batch_size, TestModels.num_classes))

    def test_get_and_run_pos_emb_mention_pooling_model(self):
        tokenizer = AutoTokenizer.from_pretrained(TestModels.base_model_name)
        train_dl = self._build_dataset_for_task(
            tokenizer, "positional", None, "raw", True, 
            ["text", "rel_label", "raw_mention_token_ids"])

        model = mention_pooling_model(
            TestModels.base_model_name, TestModels.num_classes, None)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, nn.Module)
        device = self._set_device()
        model = model.to(device)

        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            break
        self.assertIsNotNone(outputs)
        self.assertIsInstance(outputs, SequenceClassifierOutput)
        self.assertEqual(len(outputs.keys()), 2)
        self.assertTrue("loss" in outputs.keys())
        self.assertTrue("logits" in outputs.keys())
        self.assertIsInstance(outputs.loss.detach().cpu().numpy().item(), float)
        self.assertEqual(outputs.logits.detach().cpu().numpy().shape, 
            (TestModels.batch_size, TestModels.num_classes))

    def test_get_and_run_entity_marker_cls_model(self):
        tokenizer = AutoTokenizer.from_pretrained(TestModels.base_model_name)
        train_dl = self._build_dataset_for_task(
            tokenizer, "entity", "entity", None, False, ["text", "rel_label"])
        new_vocab_size = len(tokenizer.vocab)

        model = cls_model(
            TestModels.base_model_name, TestModels.num_classes,
            vocab_size=new_vocab_size)
        self.assertIsNotNone(model)
        device = self._set_device()
        model = model.to(device)

        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            break
        self.assertIsNotNone(outputs)
        self.assertIsInstance(outputs, SequenceClassifierOutput)
        self.assertEqual(len(outputs.keys()), 2)
        self.assertTrue("loss" in outputs.keys())
        self.assertTrue("logits" in outputs.keys())
        self.assertIsInstance(outputs.loss.detach().cpu().numpy().item(), float)
        self.assertEqual(outputs.logits.detach().cpu().numpy().shape, 
            (TestModels.batch_size, TestModels.num_classes))

    def test_get_and_run_entity_marker_mention_pooling_model(self):
        tokenizer = AutoTokenizer.from_pretrained(TestModels.base_model_name)
        train_dl = self._build_dataset_for_task(
            tokenizer, "entity", "entity", "tokenizer", False, ["text", "rel_label"])
        new_vocab_size = len(tokenizer.vocab)

        model = mention_pooling_model(
            TestModels.base_model_name, TestModels.num_classes, new_vocab_size)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, nn.Module)
        device = self._set_device()
        model = model.to(device)

        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            break
        self.assertIsNotNone(outputs)
        self.assertIsInstance(outputs, SequenceClassifierOutput)
        self.assertEqual(len(outputs.keys()), 2)
        self.assertTrue("loss" in outputs.keys())
        self.assertTrue("logits" in outputs.keys())
        self.assertIsInstance(outputs.loss.detach().cpu().numpy().item(), float)
        self.assertEqual(outputs.logits.detach().cpu().numpy().shape, 
            (TestModels.batch_size, TestModels.num_classes))
