# MNIST Exploration
This repo contains an exploration of the MNIST data set using Tensorflow.<br>
Several factors are being explored:
* Accuracy impacts of various types of model architectures/layers including -
  * Number of layers
  * Convolutions vs dropout
  * Dropout
  * Batch normalization
  * Activation functions
  * Different optimization functions
* The effects of gaussian noise of classification performance
* The effects of mislabeled data on the classification performance

## Repo Structure
* Exploration - Contains markdown files with Q/A and model accuracy evaluations
  * Images - Contains saved images for analysis
* Models - The saved jsons/numpy arrays of run performances
* src - The scripts used in the analysis
  * pymodels - Folder containing python files, one for each model
  * datagen.py - Data generator/augmenter
  * eval.py - The core training/testing module
  * graph.py - Used to visualize the results from the training/testing
  * summarize.py - Small utility used to quickly see the hyperparameters used in each run
  * utils.py - Utilities for saving models
