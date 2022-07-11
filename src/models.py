import logging
import torch
import torch.nn as nn

from transformers import (
    AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    DataCollatorWithPadding, get_scheduler
)
from transformers.modeling_outputs import SequenceClassifierOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cls_model(base_model_name: str, 
              num_labels: int, 
              vocab_size: int = None) -> AutoModelForSequenceClassification:
    config = AutoConfig.from_pretrained(base_model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, config=config)
    if vocab_size is not None:
        model.resize_token_embeddings(vocab_size)
    return model


class MentionPoolingForRelationExtraction(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super(MentionPoolingForRelationExtraction, self).__init__()
        self.num_labels = num_labels
        # encoder (body)
        config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(
            base_model_name, config=config)
        # head
        try:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        except AttributeError:
            self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size * 2)
        self.linear = nn.Linear(config.hidden_size * 2, self.num_labels)

    def forward(self, input_ids, attention_mask, mention_token_ids,
                token_type_ids=None, labels=None):
        if token_type_ids is None:
            # distilbert does not provide token_type_ids
            outputs = self.encoder(input_ids,
                                   attention_mask=attention_mask,
                                   output_hidden_states=False).last_hidden_state
        else:
            outputs = self.encoder(input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask,
                                   output_hidden_states=False).last_hidden_state

        sub_maxpool, obj_maxpool = [], []
        for bid in range(outputs.size(0)):
            e1_start, e1_end, e2_start, e2_end = mention_token_ids[bid]
            sub_span = torch.max(outputs[bid, e1_start:e1_end, :],
                                 dim=0, keepdim=True).values
            obj_span = torch.max(outputs[bid, e2_start:e2_end, :],
                                 dim=0, keepdim=True).values
            sub_maxpool.append(sub_span)
            obj_maxpool.append(obj_span)
        
        sub_emb = torch.cat(sub_maxpool, dim=0)
        obj_emb = torch.cat(obj_maxpool, dim=0)
        rel_input = torch.cat([sub_emb, obj_emb], dim=-1)

        rel_input = self.layer_norm(rel_input)
        rel_input = self.dropout(rel_input)
        logits = self.linear(rel_input)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return SequenceClassifierOutput(loss, logits)
        else:
            return SequenceClassifierOutput(None, logits)


def mention_pooling_model(base_model_name: str,
                          num_labels: int,
                          vocab_size: int = None) -> nn.Module:
    model = MentionPoolingForRelationExtraction(
        base_model_name, num_labels)
    if vocab_size is not None:
        model.encoder.resize_token_embeddings(vocab_size)
    return model


class EntityStartForRelationExtraction(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super(EntityStartForRelationExtraction, self).__init__()
        self.num_labels = num_labels
        # encoder (body)
        config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=config)
        # head
        try:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        except AttributeError:
            self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size * 2)
        self.linear = nn.Linear(config.hidden_size * 2, self.num_labels)

    def forward(self, input_ids, attention_mask, mention_token_ids,
                token_type_ids=None, labels=None):
        if token_type_ids is None:
            # distilbert does not provide token_type_ids
            outputs = self.encoder(input_ids,
                                   attention_mask=attention_mask,
                                   output_hidden_states=False).last_hidden_state
        else:
            outputs = self.encoder(input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask,
                                   output_hidden_states=False).last_hidden_state

        sub_start, obj_start = [], []
        for bid in range(outputs.size(0)):
            e1_start, e1_end, e2_start, e2_end = mention_token_ids[bid]
            sub_span = outputs[bid, e1_start, :].unsqueeze(0)
            obj_span = outputs[bid, e2_start, :].unsqueeze(0)
            sub_start.append(sub_span)
            obj_start.append(obj_span)

        sub_emb = torch.cat(sub_start, dim=0)
        obj_emb = torch.cat(obj_start, dim=0)
        rel_input = torch.cat([sub_emb, obj_emb], dim=-1)

        rel_input = self.layer_norm(rel_input)
        rel_input = self.dropout(rel_input)
        logits = self.linear(rel_input)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return SequenceClassifierOutput(loss, logits)
        else:
            return SequenceClassifierOutput(None, logits)


def entity_start_model(base_model_name: str,
                       num_labels: int, 
                       vocab_size: int = None) -> nn.Module:
    model = EntityStartForRelationExtraction(
        base_model_name, num_labels)
    if vocab_size is not None:
        model.encoder.resize_token_embeddings(vocab_size)
    return model


MODEL_FACTORY = {
    "cls": cls_model,
    "mention_pooling": mention_pooling_model,
    "entity_start": entity_start_model
}

