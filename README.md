# Graph Classification via Reference Distributions Learning: Theory and Practice

The repository is the official PyTorch implementation of the experiments in the following paper:

*Graph Classification via Reference Distributions Learning: Theory and Practice.* Zixiao Wang, Jicong Fan, NeurIPS 2024.

## Installation

Install Pytorch following the instructions on the [official website](https://pytorch.org/).

Then install the other dependencies with:

```shell
pip install -r requirements.txt
```

## Run

```shell
python exp.py
```

The default dataset is MUTAG. Hyper-parameters need to be specified through the commandline arguments. Please refer to our paper for the details of how we set the hyper-parameters.

To learn hyper-parameters to be specified, you directly look into the `exp.py` or type

```shell
python exp.py --help
```
