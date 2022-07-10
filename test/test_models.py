# call from parent directory: python -m unittest test/test_models.py 
import unittest

from src.models import *

class TestModels(unittest.TestCase):

    def test_get_and_run_cls_model(self):
        model = cls_model("bert-base-cased", 8, None)
        self.assertIsNotNone(model)
