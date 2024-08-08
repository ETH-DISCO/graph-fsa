# Paper Code

This repository contains the code for the paper "GraphFSA: A Finite State Automaton Framework for Algorithmic Learning on Graphs". 
## Prerequisites

- Anaconda or Miniconda installed. If not installed, you can download and install it from [here](https://www.anaconda.com/products/distribution).

## Setup and Installation

To run the code, it's recommended to use the provided conda environment. Here's how to set it up:

1. Clone the repository

2. Create a new conda environment from the provided environment.yml file ``` conda env create -f environment.yml```


3. Activate the environment
``` conda activate graphfsa```


## Reproducability
For reproducability please check the reproducability.txt in the folders of the specific experiment categrories. 

## Folder Structure

Our experiments can be divided into three experiment categories

1. Cellular Automata
2. Dataset Generator
3. Algorithms


The dataset generator uses a custom generator package located in [graph-generator/fsm_dataset](graph-generator/fsm_dataset). This graph generator allows us to create random datasets. It is important that we additionally install this dataset generator by calling

```bash 
cd graph-generator/fsm_dataset
pip install -e .
```

To train with the configurations used for the paper, we refer to the commands in `experiments.md`
