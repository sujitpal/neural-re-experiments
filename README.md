## neural-re-experiments

## Introduction

A repository of experiments with various Transformer based Relation Extraction architectures.

The code here started off as an offshoot of my tutorial on [Transformer based Named Entity Recognition (NER) and Relation Extraction (RE)](https://github.com/sujitpal/ner-re-with-transformers-odsc2022) at ODSC East 2022. In this repository I explore all six architectures referenced in the figure (reproduced below) from the paper [Matching the Blanks: Distributional Similarity for Relation Learning ](https://arxiv.org/abs/1906.03158) by Soares et al (2019).

<img src="https://github.com/sujitpal/ner-re-with-transformers-odsc2022/blob/main/figures/re-transformer-archs.png"/>

All six models are implemented in Pytorch, and trained and evaluated using the [SciERC dataset](http://nlp.cs.washington.edu/sciIE/), a collection of 500 scientific abstracts annotated with 6 entity types and 8 relationship types. All Relation Extractions are created by fine-tuning a base pre-trained Transformer model such as BERT. Each architecture has been fine tuned with BERT, DistilBERT and XL-Net.

## Methods

* Each abstract in the input can contain multiple (subject, predicate, object) triples. The preprocessing step parses this out into a JSONL file, with each triple in its own record. The format of each record is similar to the SpaCy RE format. We also partition the input 80/10/20 into training, validation and test splits.
* Following models are trained and evaluated.
  * **Standard - CLS (model A)** -- each input text (multiple sentences in an abstract) is tokenized and padded with a `[CLS]` and `[SEP]` token, and the relation is predicted from the hidden state corresponding to the `[CLS]` token. Note that this provides almost no information, and can even confuse the training, since there can be multiple relation triples within a single input text. 
  * **Standard - Mention Pooling (model B)** -- input is similar to model A, but in addition, we capture the position of each of the entity token spans connecting the relation to be predicted. These token spans are used to maxpool over the tokens corresponding to the subject and object entity spans, and concatenated to produce input for the relation prediction.
  * **Positional Embedding - Mention Pooling (model C)** -- Here in addition to the text input, we also provide the positional embedding for the entity token spans by marking them as `token_type_id=1`. Note that our implementation differs slightly from that described in the paper, but we are limited to using the existing `token_type_ids` provided by the transformer model being fine-tuned.
  * **Entity Markers - CLS (model D)** -- Input is similar to model C above, but output is just the hidden state corresponding to the `[CLS]` token.
  * **Entity Markers - Mention Pooling (model E)** -- The text is pre-procesed to enclose the subject and object entity spans with entity markers `<E1>`, `</E1>`, `<E2>` and `</E2>`. On the output side, the spans (including the entity marker token positions) are maxpooled to create subject and object embeddings, which are then concatenated to produce an input representation for the relation.
  * **Entity Markers - Entity Start (model F)** -- This model is the same as the previous model on the input side, but on the output side, instead of maxpooling across the tokens in the subject and object spans, we concatenate only the states corresponding to the start tokens `<E1>` and `<E2>` to produce the input relation type representation.

## Results


## Future Work

* Would be interesting to test out variations of the six model architectures proposed above, to understand the effect of model performance in response to increasing the (NER) information used during training.
* Would be interesting to try these models out with other popular transformer models such as RoBERTa and ALBERT as well.
* Some other interesting Relation Extraction Datasets are listed below. Each set has a different mix of relation types vs number of examples, it will be interesting to see how the above models perform against them.
  * [SemEval 2010 Task B](http://www.kozareva.com/downloads.html) -- 10 relation types, 10,717 examples
  * [TACRED](https://nlp.stanford.edu/projects/tacred/) -- 41 relation types, 106,264 examples
  * [FewRel](http://www.zhuhao.me/fewrel/) -- 100 relation types, 70,000 examples

