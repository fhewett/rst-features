# rst-features
Code to extract sentence- or segment-level features from RST trees

## What is this repository for? 

This repository is the code used for the paper [Extractive summarisation for German-language data: a text-level approach with discourse features](https://aclanthology.org/venues/coling/). We extract features from .rs3 formatted files (RST trees) and use them as input for some basic extractive summarisation models (Logistic Regression, a feed-forward network and an LSTM network). The .rs3 files we work with are from the [Potsdam Commentary Corpus](http://angcl.ling.uni-potsdam.de/resources/pcc.html).

You can also use this code to extract features from your .rs3 files. The features are: the nuclearity status, the relation, the depth and the most-nuclearity-ness, either at segment- (EDU-) or sentence-level.

## Usage

- To replicate the models in the paper just run replicate_results.sh. This creates a virtual environment and runs the models using the features `edu_level_pcc.json` and/or `sent_level_pcc.json`.
- To create the PCC features used in the paper, first download the [PCC](http://angcl.ling.uni-potsdam.de/resources/pcc.html) and then download the [summaries](https://github.com/fhewett/pcc-summaries). Put the `pcc-summaries` folder in the `PotsdamCommentaryCorpus` folder.  
- To extract features from your own data run `rst-create-features.py`. It takes the arguments `corpus_path` (where is your corpus saved) and `level` (do you want `edu`, `sentence` or `both` level features). If you are using your own corpus you should have two folders within your `corpus_path`: one called `rst` with .rs3 formatted files and one called `syntax` with .txt files with one sentence per line. The file names should be identical (apart from the file ending).

## Associated files

- Potsdam Commentary Corpus [(PCC)](http://angcl.ling.uni-potsdam.de/resources/pcc.html)
- [Summaries of the PCC](https://github.com/fhewett/pcc-summaries)

## Citation & further information

More information can be found in the paper. If you use anything from this repository please cite the following paper:

Freya Hewett and Manfred Stede. Extractive summarisation for German-language data: a text-level approach with discourse features. In *Proceedings of the 29th International Conference on Computational Linguistics (COLING)*. 2022. To appear.

