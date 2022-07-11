# call from parent directory: python -m unittest test/test_models.py 
import os
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
        cls.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        cls.label_fp = os.path.join("data/prepared/relations.txt")
        cls.max_length = 400
        cls.batch_size = 32
        cls.test_mode = True
        cls.num_classes = 8

    def _build_dataset_for_task(self, tag_type, align_mention_token_ids,
                               update_token_type_ids, remove_columns):
        prep_dir = "data/prepared"
        temp_dir = "data/temp"
        raw_dataset = build_raw_dataset(
            prep_dir, temp_dir, TestModels.tokenizer, tag_type=tag_type)
        label2id, _, _ = build_label_mappings(TestModels.label_fp)

        data_encoder_fn = partial(encode_data, 
            label2id=label2id, 
            tokenizer=TestModels.tokenizer,
            max_length=TestModels.max_length,
            align_mention_token_ids=align_mention_token_ids,
            update_token_type_ids=update_token_type_ids)
        enc_dataset = raw_dataset.map(
            data_encoder_fn, batched=True,
            remove_columns=remove_columns)

        train_dl = build_dataloader_for_split(
            enc_dataset, TestModels.tokenizer, "train", 
            batch_size=TestModels.batch_size, test_mode=TestModels.test_mode)
        return train_dl


    def test_get_and_run_cls_model(self):
        model = cls_model("bert-base-cased", 8, None)
        self.assertIsNotNone(model)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            # test with standard input
            train_dl = self._build_dataset_for_task(
                None, False, False, ["text", "rel_label", "mention_token_ids"])
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
