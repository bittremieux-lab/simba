
<img src="docs/simba_logo.png" width="150" style="display: block; margin: auto;"/>

# SIMBA


`Simba` is a transformer neural network able to predict structural similarity based on tandem mass spectrometry (MS/MS) data from 2 spectra. The model predicts 2 metrics: (1) the Maximum Common Edge Substructure (MCES) distance, and  (2) the substructure edit distance.

This repository provides with the necessary code to preprocess the data, train the model as well as to compute inference based on  user spectra.


## Setup

### Requirements

Python 3.11.7.

### Installation
This installation is expected to take 10-20 minutes.

###  Environment

You can create a conda environment with the corresponding dependencies:

```
conda env create -f environment.yml
conda activate transformers
```

## Getting started: How to prepare data, train a model, and compute similarities.


## 1) Compute structural similarities

We provide a SIMBA model trained on around 300,000 spectra coming from NIST20 and MassSpecGym. The model can be found at [INSERT LOCATION]. The model works on positive mode for protonized adducts.

We recommend to use the tutorial in [Run Inference](https://github.com/bittremieux-lab/simba/tree/main/notebooks/final_tutorials/run_inference.ipynb)  for an extensive example on test data. The expected run time on a laptop is less than 10 minutes, including automatic model and data download. 

A example dataset can be found in [LOCATION IN THE REPOSITORY]. You can use your own data using the .mgf format.


## 2) Train a SIMBA model

