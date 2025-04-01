
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
conda activate simba
```

After activating the conda environment, the module SIMBA is installed through:

```
pip install -e .
```

In case you are working on MAC, install the corresponding xz dependency through Homebrew:

```
brew install xz
```

## Getting started: How to prepare data, train a model, and compute similarities.


## 1) Compute structural similarities

We provide a SIMBA model trained on around 300,000 spectra coming from NIST20 and MassSpecGym. The model can be found at [INSERT LOCATION]. The model works on positive mode for protonized adducts.

We recommend to use the tutorial in [Run Inference](https://github.com/bittremieux-lab/simba/tree/main/notebooks/final_tutorials/run_inference.ipynb)  for an extensive example on test data. The expected run time on a laptop is less than 10 minutes, including automatic model and data download. 

A example dataset can be found in [LOCATION IN THE REPOSITORY]. You can use your own data using the .mgf format.


## 2) Train a SIMBA model



## 3) Inference time for CPU/GPU 

Using an Apple M3 Pro with 36GB of RAM, SIMBA obtains the embeddings of 100,000 embeddings in approximately 1 minute. These embeddings are the vectors used by the headers that compute the final outputs of the model.

<img src="docs/nn_architecture.png" width="300" style="display: block; margin: auto;"/>

SIMBA is able to cache embeddings of spectra previously computed to accelerate library search. In this way, it is not necessary to compute again the embeddings of the reference spectra.

To compute the susbtructure edit distance and MCES distance between 100,000 precomputed embeddings and 1 query spectra, it takes 10 seconds.



