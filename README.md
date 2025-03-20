

# SIMBA


`Simba` is a transformer neural network able to predict structural similarity based on tandem mass spectrometry (MS/MS) from 2 spectra. The model predicts 2 metrics: (1) the Maximum Common Edge Substructure (MCES) distance, and  (2) the substructure edit distance.

This repository provides with the necessary code to preprocess the data, train the model as well as to compute inference based on  user spectra.


## Setup

### Requirements

Python 3.11.7.

### Installation
This installation is expected to take 10-20 minutes.

###  Environment

It is recommended you can create a conda environment with the corresponding dependencies:

```
conda env create -f environment.yml
conda activate transformers
```

## Getting started: How to prepare data, train a model, and compute similarities.
We recommend to run the complete tutorial in [notebooks/MS2DeepScore_tutorial.ipynb](https://github.com/matchms/ms2deepscore/blob/main/notebooks/MS2DeepScore_tutorial.ipynb) 
for a more extensive fully-working example on test data. The expected run time on a laptop is less than 5 minutes, including automatic model and dummy data download. 
Alternatively there are some example scripts below.
If you are not familiar with `matchms` yet, then we also recommand our [tutorial on how to get started using matchms](https://blog.esciencecenter.nl/build-your-own-mass-spectrometry-analysis-pipeline-in-python-using-matchms-part-i-d96c718c68ee).

## 1) Compute structural similarities

We provide a SIMBA model trained on around 300,000 spectra coming from NIST20 and MassSpecGym. The model can be found at [INSERT LOCATION]. The model works on positive mode for protonized adducts.

A example dataset can be found in [LOCATION IN THE REPOSITORY]. You can use your own data using the .mgf format.


## 2) Train a SIMBA model

