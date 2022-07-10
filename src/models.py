from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, get_scheduler
)
from transformers.modeling_outputs import SequenceClassifierOutput


def cls_model(model_name: str, 
              num_labels: int, 
              vocab_size: int = None) -> AutoModelForSequenceClassification:
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config)
    if vocab_size is not None:
        model.resize_token_embeddings(vocab_size)
    return model

