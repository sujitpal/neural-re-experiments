## neural-re-experiments

## Introduction

A repository of experiments with various Transformer based Relation Extraction architectures.

The code here started off as an offshoot of my [tutorial on Transformer based Named Entity Recognition (NER) and Relation Extraction (RE) at ODSC East 2022](https://github.com/sujitpal/ner-re-with-transformers-odsc2022). In this repository I explore all six architectures referenced in the figure (reproduced below) from the paper [Matching the Blanks: Distributional Similarity for Relation Learning ](https://arxiv.org/abs/1906.03158) by Soares et al (2019).

<img src="https://github.com/sujitpal/ner-re-with-transformers-odsc2022/blob/main/figures/re-transformer-archs.png"/>

All six models are implemented in Pytorch, and trained and evaluated using the [SciERC dataset](http://nlp.cs.washington.edu/sciIE/), a collection of 500 scientific abstracts annotated with 6 entity types and 8 relationship types. All Relation Extractions are created by fine-tuning a base pre-trained Transformer model such as BERT. Each architecture has been fine tuned with BERT, DistilBERT and XL-Net.

## Methods

## Results


## Future Work

* Would be interesting to test out variations of the six model architectures proposed above, to understand the effect of model performance in response to increasing the (NER) information used during training.
* Would be interesting to try these models out with other popular transformer models such as RoBERTa and ALBERT as well.
* Some other interesting Relation Extraction Datasets are listed below. Each set has a different mix of relation types vs number of examples, it will be interesting to see how the above models perform against them.
  * [SemEval 2010 Task B](http://www.kozareva.com/downloads.html) -- 10 relation types, 10,717 examples
  * [TACRED](https://nlp.stanford.edu/projects/tacred/) -- 41 relation types, 106,264 examples
  * [FewRel](http://www.zhuhao.me/fewrel/) -- 100 relation types, 70,000 examples

