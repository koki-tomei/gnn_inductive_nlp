<div id="top"></div>

## Overview

This is the source code used in the experiments of our paper (Combined GNN specialized in inductive completion and PLM for natural language induction).
It is a framework for solving inductive reasoning tasks in natural language form using GNNs.
The implementation is built upon [PyTorch](https://pytorch.org).

## Setup

### Dependencies

1. Download torch-related and other major required libraries

```shell
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install transformers==3.4.0 \
    && pip install nltk spacy==2.1.6
python3 -m spacy download en
pip install torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html \
    && pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html \
    && pip install torch-geometric==1.7.0 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
```

2. Download the remaining dependencies by `pip install -r requirements.txt`

### Obtaining Datasets

Download the CLUTRR dataset from [here](https://drive.google.com/file/d/1SEq_e1IVCDDzsBIBhoUQ5pOVH5kxRoZF/view).

Place the extracted data in `experiments/data/clutrr`.
Place a folder named with the id of the subset (e.g. data_089907f8).

## Preprocess

To perform data preprocessing, run:

```shell
cd experiments
./run_clutrr_preprocess.sh
```

## Experiments

To perform experiments, run:

```
./run_experiments.sh
```
