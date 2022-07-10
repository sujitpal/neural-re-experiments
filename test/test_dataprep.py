# call from parent directory: python -m unittest test/test_dataprep.py
import os
import unittest

from transformers import AutoTokenizer

from src.dataprep import *

INPUT_JSONL = """{"text": "\nThis paper proposes an approach to  full parsing  suitable for  Information Extraction  from  texts . Sequences of cascades of  rules  deterministically analyze the  text , building  unambiguous structures . Initially basic  chunks  are analyzed; then  argumental relations  are recognized; finally  modifier attachment  is performed and the  global parse tree  is built. The approach was proven to work for three  languages  and different  domains . It was implemented in the  IE module  of  FACILE, a EU project for multilingual text classification and IE .\n", "h": {"type": "Generic", "start": 377, "end": 385, "span": "approach"}, "t": {"type": "Generic", "start": 24, "end": 32, "span": "approach"}, "r": "COREF"}"""

class TestDataPrep(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        cls.label_fp = os.path.join("data/prepared/relations.txt")

    def test_reformat_prepared_jsonl_standard(self):
        output_rec = reformat_prepared_jsonl(
            INPUT_JSONL.strip(), TestDataPrep.tokenizer)
        self.assertIsNotNone(output_rec)
        # print("output_rec:", output_rec)
        self.assertEqual(len(output_rec.keys()), 3)
        self.assertEqual(output_rec["text"], """[CLS] This paper proposes an approach to full parsing suitable for Information Extraction from texts . Sequences of cascades of rules deterministically analyze the text , building unambiguous structures . Initially basic chunks are analyzed; then argumental relations are recognized; finally modifier attachment is performed and the global parse tree is built. The approach was proven to work for three languages and different domains . It was implemented in the IE module of FACILE, a EU project for multilingual text classification and IE . [SEP]""")
        self.assertEqual(output_rec["rel_label"], "COREF")
        self.assertEqual(output_rec["mention_token_ids"], [5, 5, 53, 53])

    def test_reformat_prepared_jsonl_ent_markers(self):
        output_rec = reformat_prepared_jsonl(
            INPUT_JSONL.strip(), TestDataPrep.tokenizer,
            tag_type="entity")
        self.assertIsNotNone(output_rec)
        # print("output_rec:", output_rec)
        self.assertEqual(len(output_rec.keys()), 3)
        self.assertEqual(output_rec["text"], """[CLS] This paper proposes an <E1>approach</E1> to full parsing suitable for Information Extraction from texts . Sequences of cascades of rules deterministically analyze the text , building unambiguous structures . Initially basic chunks are analyzed; then argumental relations are recognized; finally modifier attachment is performed and the global parse tree is built. The <E2>approach</E2> was proven to work for three languages and different domains . It was implemented in the IE module of FACILE, a EU project for multilingual text classification and IE . [SEP]""")
        self.assertEqual(output_rec["rel_label"], "COREF")
        self.assertEqual(output_rec["mention_token_ids"], [5, 5, 53, 53])

    def test_reformat_prepared_jsonl_ent_type_markers(self):
        output_rec = reformat_prepared_jsonl(
            INPUT_JSONL.strip(), TestDataPrep.tokenizer,
            tag_type="entity_type")
        self.assertIsNotNone(output_rec)
        # print("output_rec:", output_rec)
        self.assertEqual(len(output_rec.keys()), 3)
        self.assertEqual(output_rec["text"], """[CLS] This paper proposes an <E1:Generic>approach</E1:Generic> to full parsing suitable for Information Extraction from texts . Sequences of cascades of rules deterministically analyze the text , building unambiguous structures . Initially basic chunks are analyzed; then argumental relations are recognized; finally modifier attachment is performed and the global parse tree is built. The <E2:Generic>approach</E2:Generic> was proven to work for three languages and different domains . It was implemented in the IE module of FACILE, a EU project for multilingual text classification and IE . [SEP]""")
        self.assertEqual(output_rec["rel_label"], "COREF")
        self.assertEqual(output_rec["mention_token_ids"], [5, 5, 53, 53])

    def test_build_raw_dataset_default(self):
        prep_dir = "data/prepared"
        temp_dir = "data/temp"
        raw_dataset = build_raw_dataset(prep_dir, temp_dir, TestDataPrep.tokenizer)
        self.assertEqual(len(raw_dataset.keys()), 3)
        self.assertEqual(raw_dataset["train"].num_rows, 4603)
        self.assertEqual(len(raw_dataset["train"].features.keys()), 3)

    def test_build_raw_dataset_entity_marker_tags(self):
        prep_dir = "data/prepared"
        temp_dir = "data/temp"
        raw_dataset = build_raw_dataset(
            prep_dir, temp_dir, TestDataPrep.tokenizer, tag_type="entity")
        self.assertEqual(len(raw_dataset.keys()), 3)
        self.assertEqual(raw_dataset["train"].num_rows, 4603)
        self.assertEqual(len(raw_dataset["train"].features.keys()), 3)

    def test_build_raw_dataset_entity_type_tags(self):
        prep_dir = "data/prepared"
        temp_dir = "data/temp"
        raw_dataset = build_raw_dataset(
            prep_dir, temp_dir, TestDataPrep.tokenizer, tag_type="entity_type")
        self.assertEqual(len(raw_dataset.keys()), 3)
        self.assertEqual(raw_dataset["train"].num_rows, 4603)
        self.assertEqual(len(raw_dataset["train"].features.keys()), 3)

    def test_build_label_mapping(self):
        label2id, id2label, relations = build_label_mappings(TestDataPrep.label_fp)
        self.assertEqual(len(relations), 8)
        self.assertEqual(len(label2id), len(relations))
        self.assertEqual(len(id2label), len(relations))

    def test_encode_data_default(self):
        prep_dir = "data/prepared"
        temp_dir = "data/temp"
        raw_dataset = build_raw_dataset(prep_dir, temp_dir, TestDataPrep.tokenizer)
        examples = raw_dataset["train"][0:5]
        label2id, id2label, relations = build_label_mappings(TestDataPrep.label_fp)
        tokenized_inputs = encode_data(examples, label2id, TestDataPrep.tokenizer, 400)
        self.assertEqual(len(tokenized_inputs.keys()), 4)
        self.assertTrue("input_ids" in tokenized_inputs.keys())
        self.assertTrue("token_type_ids" in tokenized_inputs.keys())
        self.assertTrue("attention_mask" in tokenized_inputs.keys())
        self.assertTrue("label" in tokenized_inputs.keys())

    def test_encode_data_with_mention_token_ids(self):
        prep_dir = "data/prepared"
        temp_dir = "data/temp"
        raw_dataset = build_raw_dataset(
            prep_dir, temp_dir, TestDataPrep.tokenizer, tag_type="entity")
        examples = raw_dataset["train"][0:5]
        label2id, id2label, relations = build_label_mappings(TestDataPrep.label_fp)
        tokenized_inputs = encode_data(examples, label2id, TestDataPrep.tokenizer, 
            400, align_mention_token_ids=True)
        self.assertEqual(len(tokenized_inputs.keys()), 5)
        self.assertTrue("input_ids" in tokenized_inputs.keys())
        self.assertTrue("token_type_ids" in tokenized_inputs.keys())
        self.assertTrue("attention_mask" in tokenized_inputs.keys())
        self.assertTrue("label" in tokenized_inputs.keys())
        self.assertTrue("mention_token_ids" in tokenized_inputs.keys())

    def test_encode_data_with_mention_token_ids_and_token_type_ids(self):
        prep_dir = "data/prepared"
        temp_dir = "data/temp"
        raw_dataset = build_raw_dataset(
            prep_dir, temp_dir, TestDataPrep.tokenizer, tag_type="entity")
        examples = raw_dataset["train"][0:5]
        label2id, id2label, relations = build_label_mappings(TestDataPrep.label_fp)
        tokenized_inputs = encode_data(examples, label2id, TestDataPrep.tokenizer, 
            400, align_mention_token_ids=True, update_token_type_ids=True)
        self.assertEqual(len(tokenized_inputs.keys()), 5)
        self.assertTrue("input_ids" in tokenized_inputs.keys())
        self.assertTrue("token_type_ids" in tokenized_inputs.keys())
        self.assertTrue("attention_mask" in tokenized_inputs.keys())
        self.assertTrue("label" in tokenized_inputs.keys())
        self.assertTrue("mention_token_ids" in tokenized_inputs.keys())
