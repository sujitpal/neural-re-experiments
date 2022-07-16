# call from parent directory: python -m unittest test/test_dataprep.py
import os
import unittest

from functools import partial
from transformers import AutoTokenizer

from src.dataprep import *

INPUT_JSONL = """{"text": "\nThis paper proposes an approach to  full parsing  suitable for  Information Extraction  from  texts . Sequences of cascades of  rules  deterministically analyze the  text , building  unambiguous structures . Initially basic  chunks  are analyzed; then  argumental relations  are recognized; finally  modifier attachment  is performed and the  global parse tree  is built. The approach was proven to work for three  languages  and different  domains . It was implemented in the  IE module  of  FACILE, a EU project for multilingual text classification and IE .\n", "h": {"type": "Generic", "start": 377, "end": 385, "span": "approach"}, "t": {"type": "Generic", "start": 24, "end": 32, "span": "approach"}, "r": "COREF"}"""

class TestDataPrep(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        cls.prep_dir = "data/prepared"
        cls.temp_dir = "data/temp"
        cls.max_length = 400
        cls.batch_size = 32

    def test_reformat_prepared_jsonl_standard(self):
        output_rec = reformat_prepared_jsonl(
            INPUT_JSONL.strip(), TestDataPrep.tokenizer)
        self.assertIsNotNone(output_rec)
        self.assertEqual(len(output_rec.keys()), 2)
        self.assertEqual(output_rec["text"], """[CLS] This paper proposes an approach to full parsing suitable for Information Extraction from texts . Sequences of cascades of rules deterministically analyze the text , building unambiguous structures . Initially basic chunks are analyzed; then argumental relations are recognized; finally modifier attachment is performed and the global parse tree is built. The approach was proven to work for three languages and different domains . It was implemented in the IE module of FACILE, a EU project for multilingual text classification and IE . [SEP]""")
        self.assertEqual(output_rec["rel_label"], "COREF")

    def test_reformat_prepared_jsonl_positional(self):
        output_rec = reformat_prepared_jsonl(
            INPUT_JSONL.strip(), TestDataPrep.tokenizer, tag_type="positional")
        self.assertIsNotNone(output_rec)
        # print("output_rec:", output_rec)
        self.assertEqual(len(output_rec.keys()), 3)
        self.assertEqual(output_rec["text"], """[CLS] This paper proposes an approach to full parsing suitable for Information Extraction from texts . Sequences of cascades of rules deterministically analyze the text , building unambiguous structures . Initially basic chunks are analyzed; then argumental relations are recognized; finally modifier attachment is performed and the global parse tree is built. The approach was proven to work for three languages and different domains . It was implemented in the IE module of FACILE, a EU project for multilingual text classification and IE . [SEP]""")
        self.assertEqual(output_rec["rel_label"], "COREF")
        self.assertEqual(output_rec["raw_mention_token_ids"], [5, 5, 53, 53])

    def test_reformat_prepared_jsonl_ent_markers(self):
        output_rec = reformat_prepared_jsonl(
            INPUT_JSONL.strip(), TestDataPrep.tokenizer,
            tag_type="entity")
        self.assertIsNotNone(output_rec)
        self.assertEqual(len(output_rec.keys()), 2)
        self.assertEqual(output_rec["text"], """[CLS] This paper proposes an <E1>approach</E1> to full parsing suitable for Information Extraction from texts . Sequences of cascades of rules deterministically analyze the text , building unambiguous structures . Initially basic chunks are analyzed; then argumental relations are recognized; finally modifier attachment is performed and the global parse tree is built. The <E2>approach</E2> was proven to work for three languages and different domains . It was implemented in the IE module of FACILE, a EU project for multilingual text classification and IE . [SEP]""")
        self.assertEqual(output_rec["rel_label"], "COREF")

    def test_reformat_prepared_jsonl_ent_type_markers(self):
        output_rec = reformat_prepared_jsonl(
            INPUT_JSONL.strip(), TestDataPrep.tokenizer,
            tag_type="entity_type")
        self.assertIsNotNone(output_rec)
        self.assertEqual(len(output_rec.keys()), 2)
        self.assertEqual(output_rec["text"], """[CLS] This paper proposes an <E1:Generic>approach</E1:Generic> to full parsing suitable for Information Extraction from texts . Sequences of cascades of rules deterministically analyze the text , building unambiguous structures . Initially basic chunks are analyzed; then argumental relations are recognized; finally modifier attachment is performed and the global parse tree is built. The <E2:Generic>approach</E2:Generic> was proven to work for three languages and different domains . It was implemented in the IE module of FACILE, a EU project for multilingual text classification and IE . [SEP]""")
        self.assertEqual(output_rec["rel_label"], "COREF")

    def test_build_raw_dataset_default(self):
        raw_dataset = build_raw_dataset(
            TestDataPrep.prep_dir, TestDataPrep.temp_dir, TestDataPrep.tokenizer,
            tag_type=None)
        self.assertEqual(len(raw_dataset.keys()), 3)
        self.assertEqual(raw_dataset["train"].num_rows, 4603)
        self.assertEqual(len(raw_dataset["train"].features.keys()), 2)

    def test_build_raw_dataset_entity_marker_tags(self):
        raw_dataset = build_raw_dataset(
            TestDataPrep.prep_dir, TestDataPrep.temp_dir, TestDataPrep.tokenizer, 
            tag_type="entity")
        self.assertEqual(len(raw_dataset.keys()), 3)
        self.assertEqual(raw_dataset["train"].num_rows, 4603)
        self.assertEqual(len(raw_dataset["train"].features.keys()), 2)

    def test_build_raw_dataset_entity_type_tags(self):
        raw_dataset = build_raw_dataset(
            TestDataPrep.prep_dir, TestDataPrep.temp_dir, TestDataPrep.tokenizer,
            tag_type="entity_type")
        self.assertEqual(len(raw_dataset.keys()), 3)
        self.assertEqual(raw_dataset["train"].num_rows, 4603)
        self.assertEqual(len(raw_dataset["train"].features.keys()), 2)

    def test_build_label_mapping(self):
        label2id, id2label, relations = build_label_mappings(TestDataPrep.prep_dir)
        self.assertEqual(len(relations), 8)
        self.assertEqual(len(label2id), len(relations))
        self.assertEqual(len(id2label), len(relations))

    def test_encode_data_standard_cls(self):
        raw_dataset = build_raw_dataset(
            TestDataPrep.prep_dir, TestDataPrep.temp_dir, TestDataPrep.tokenizer,
            tag_type=None)
        examples = raw_dataset["train"][0:5]
        label2id, id2label, relations = build_label_mappings(TestDataPrep.prep_dir)
        entity_types = get_entity_type_list(TestDataPrep.prep_dir)
        tokenized_inputs = encode_data(
            examples, label2id, TestDataPrep.tokenizer, TestDataPrep.max_length,
            entity_types, tags_added_to_text=None,
            mention_token_ids_src=None, position_embedding=False)
        self.assertEqual(len(tokenized_inputs.keys()), 4)
        self.assertTrue("input_ids" in tokenized_inputs.keys())
        self.assertTrue("token_type_ids" in tokenized_inputs.keys())
        self.assertTrue("attention_mask" in tokenized_inputs.keys())
        self.assertTrue("label" in tokenized_inputs.keys())

    def test_encode_data_standard_mention_pooling(self):
        raw_dataset = build_raw_dataset(
            TestDataPrep.prep_dir, TestDataPrep.temp_dir, TestDataPrep.tokenizer,
            tag_type="positional")
        examples = raw_dataset["train"][0:5]
        label2id, id2label, relations = build_label_mappings(TestDataPrep.prep_dir)
        entity_types = get_entity_type_list(TestDataPrep.prep_dir)
        tokenized_inputs = encode_data(
            examples, label2id, TestDataPrep.tokenizer, TestDataPrep.max_length,
            entity_types, tags_added_to_text=None,
            mention_token_ids_src="raw", position_embedding=False)
        self.assertEqual(len(tokenized_inputs.keys()), 5)
        self.assertTrue("input_ids" in tokenized_inputs.keys())
        self.assertTrue("token_type_ids" in tokenized_inputs.keys())
        self.assertTrue("attention_mask" in tokenized_inputs.keys())
        self.assertTrue("label" in tokenized_inputs.keys())
        self.assertTrue("mention_token_ids" in tokenized_inputs.keys())

    def test_encode_data_positional_embedding_mention_pooling(self):
        raw_dataset = build_raw_dataset(
            TestDataPrep.prep_dir, TestDataPrep.temp_dir, TestDataPrep.tokenizer,
            tag_type="entity")
        examples = raw_dataset["train"][0:5]
        label2id, id2label, relations = build_label_mappings(TestDataPrep.prep_dir)
        entity_types = get_entity_type_list(TestDataPrep.prep_dir)
        tokenized_inputs = encode_data(
            examples, label2id, TestDataPrep.tokenizer, TestDataPrep.max_length,
            entity_types, tags_added_to_text="entity", 
            mention_token_ids_src="tokenizer", position_embedding=True)
        self.assertEqual(len(tokenized_inputs.keys()), 5)
        self.assertTrue("input_ids" in tokenized_inputs.keys())
        self.assertTrue("token_type_ids" in tokenized_inputs.keys())
        self.assertTrue("attention_mask" in tokenized_inputs.keys())
        self.assertTrue("label" in tokenized_inputs.keys())
        self.assertTrue("mention_token_ids" in tokenized_inputs.keys())

    def test_encode_data_entity_markers_cls(self):
        raw_dataset = build_raw_dataset(
            TestDataPrep.prep_dir, TestDataPrep.temp_dir, TestDataPrep.tokenizer,
            tag_type="entity")
        examples = raw_dataset["train"][0:5]
        label2id, id2label, relations = build_label_mappings(TestDataPrep.prep_dir)
        entity_types = get_entity_type_list(TestDataPrep.prep_dir)
        tokenized_inputs = encode_data(
            examples, label2id, TestDataPrep.tokenizer, TestDataPrep.max_length,
            entity_types, tags_added_to_text="entity",
            mention_token_ids_src="tokenizer", position_embedding=False)
        self.assertEqual(len(tokenized_inputs.keys()), 5)
        self.assertTrue("input_ids" in tokenized_inputs.keys())
        self.assertTrue("token_type_ids" in tokenized_inputs.keys())
        self.assertTrue("attention_mask" in tokenized_inputs.keys())
        self.assertTrue("label" in tokenized_inputs.keys())
        self.assertTrue("mention_token_ids" in tokenized_inputs.keys())

    def test_encode_data_entity_markers_mention_pooling(self):
        raw_dataset = build_raw_dataset(
            TestDataPrep.prep_dir, TestDataPrep.temp_dir, TestDataPrep.tokenizer,
            tag_type="entity")
        examples = raw_dataset["train"][0:5]
        label2id, id2label, relations = build_label_mappings(TestDataPrep.prep_dir)
        entity_types = get_entity_type_list(TestDataPrep.prep_dir)
        tokenized_inputs = encode_data(
            examples, label2id, TestDataPrep.tokenizer, TestDataPrep.max_length,
            entity_types, tags_added_to_text="entity",
            mention_token_ids_src="tokenizer", position_embedding=False)
        self.assertEqual(len(tokenized_inputs.keys()), 5)
        self.assertTrue("input_ids" in tokenized_inputs.keys())
        self.assertTrue("token_type_ids" in tokenized_inputs.keys())
        self.assertTrue("attention_mask" in tokenized_inputs.keys())
        self.assertTrue("label" in tokenized_inputs.keys())
        self.assertTrue("mention_token_ids" in tokenized_inputs.keys())

    def test_create_encoded_dataset(self):
        raw_dataset = build_raw_dataset(
            TestDataPrep.prep_dir, TestDataPrep.temp_dir, TestDataPrep.tokenizer,
            tag_type="entity")
        label2id, id2label, relations = build_label_mappings(TestDataPrep.prep_dir)
        entity_types = get_entity_type_list(TestDataPrep.prep_dir)
        data_encoder_fn = partial(encode_data, 
            label2id=label2id, 
            tokenizer=TestDataPrep.tokenizer,
            max_length=TestDataPrep.max_length,
            entity_types=entity_types,
            tags_added_to_text="entity",
            mention_token_ids_src="tokenizer",
            position_embedding=False)
        enc_dataset = raw_dataset.map(
            data_encoder_fn, batched=True,
            remove_columns=["text", "rel_label"])
        self.assertIsInstance(enc_dataset, DatasetDict)
        self.assertEqual(len(enc_dataset.keys()), 3)
        self.assertEquals(len(enc_dataset["train"].features.keys()), 5)
        self.assertTrue("attention_mask" in enc_dataset["train"].features.keys())
        self.assertTrue("input_ids" in enc_dataset["train"].features.keys())
        self.assertTrue("token_type_ids" in enc_dataset["train"].features.keys())
        self.assertTrue("label" in enc_dataset["train"].features.keys())
        self.assertTrue("mention_token_ids" in enc_dataset["train"].features.keys())

    def test_create_and_run_dataloader(self):
        raw_dataset = build_raw_dataset(
            TestDataPrep.prep_dir, TestDataPrep.temp_dir, TestDataPrep.tokenizer,
            tag_type="entity")
        label2id, id2label, relations = build_label_mappings(TestDataPrep.prep_dir)
        entity_types = get_entity_type_list(TestDataPrep.prep_dir)

        data_encoder_fn = partial(encode_data, 
            label2id=label2id, 
            tokenizer=TestDataPrep.tokenizer,
            max_length=TestDataPrep.max_length,
            entity_types=entity_types,
            tags_added_to_text="entity",
            mention_token_ids_src="tokenizer",
            position_embedding=False)
        enc_dataset = raw_dataset.map(
            data_encoder_fn, batched=True,
            remove_columns=["text", "rel_label"])

        train_dl = build_dataloader_for_split(
            enc_dataset, TestDataPrep.tokenizer,
            "train",
            batch_size=TestDataPrep.batch_size,
            test_mode=True)
        self.assertIsNotNone(train_dl)
        self.assertIsInstance(train_dl, DataLoader)

        for batch in train_dl:
            print("batch:", batch.keys())
            self.assertEqual(batch["attention_mask"].cpu().detach().numpy().shape[0],
                TestDataPrep.batch_size)
            self.assertEqual(batch["input_ids"].cpu().detach().numpy().shape[0],
                TestDataPrep.batch_size)
            self.assertEqual(batch["token_type_ids"].cpu().detach().numpy().shape[0],
                TestDataPrep.batch_size)
            self.assertEqual(batch["labels"].cpu().detach().numpy().shape[0],
                TestDataPrep.batch_size)
            self.assertEqual(batch["attention_mask"].cpu().detach().numpy().shape[1],
                batch["input_ids"].cpu().detach().numpy().shape[1])
            self.assertEqual(batch["attention_mask"].cpu().detach().numpy().shape[1],
                batch["token_type_ids"].cpu().detach().numpy().shape[1])
            break
