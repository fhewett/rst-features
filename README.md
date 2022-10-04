# rst-features
Code to extract sentence- or segment-level features from RST trees

## What is this repository for? 

This repository is the code used for the paper [Extractive summarisation for German-language data: a text-level approach with discourse features](https://aclanthology.org/venues/coling/). We extract features from .rs3 formatted files (RST trees) and use them as input for some basic extractive summarisation models (Logistic Regression, a feed-forward network and an LSTM network). The .rs3 files we work with are from the [Potsdam Commentary Corpus](http://angcl.ling.uni-potsdam.de/resources/pcc.html).

You can also use this code to extract features from your .rs3 files. The features are: the nuclearity status, the relation, the depth and the most-nuclearity-ness, either at segment- (EDU-) or sentence-level.

## Associated files

- Potsdam Commentary Corpus (PCC)
- Summaries of the PCC

## Further information

More information can be found in the paper.

## Usage

- To replicate the models in the paper just run ....sh
- To create the PCC features used in the paper or create your own features run .....sh

## Citation

If you use anything from this repository please cite the following paper:

Freya Hewett and Manfred Stede. Extractive summarisation for German-language data: a text-level approach with discourse features. In *Proceedings of the 29th International Conference on Computational Linguistics (COLING)*. 2022. To appear.

