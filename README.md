# TensorTango: A Neural Network Library from Scratch

TensorTango is a Python library for implementing neural networks from scratch, aimed at replicating some functionalities of TensorFlow. This library includes components for computing the loss function, implementing optimizers, initializing neuron layers and activation functions, performing forward and backward propagation, iterating through data in batches, and training neural networks.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Autoencoder Implementation](#autoencoder-implementation)
- [WineQuality Dataset](#winequality-dataset)
- [XGBoost Classifier](#xgboost-classifier)
- [Contributing](#contributing)

## Introduction

TensorTango is developed to provide a foundational understanding of neural networks by building them from scratch. It offers a learning opportunity for those interested in deep learning concepts and TensorFlow internals.

## Features

- **Loss Function**: Compute various loss functions such as mean squared error, categorical cross-entropy, etc.
- **Optimizer**: Implement optimizers like stochastic gradient descent (SGD), Adam, RMSprop, etc.
- **Neuron Layers**: Initialize layers such as dense layers, convolutional layers, etc.
- **Activation Functions**: Support for activation functions like ReLU, sigmoid, tanh, etc.
- **Forward and Backward Propagation**: Implement both forward and backward passes for gradient calculation.
- **Batch Processing**: Process data in batches for efficient training.
- **Training Loop**: Train neural networks with customizable options for epochs, batch size, learning rate, etc.


## Autoencoder Implementation

In addition to basic neural network functionalities, TensorTango includes an autoencoder implementation. Autoencoders are used for dimensionality reduction and feature learning. An autoencoder built from TensorTango, as well as from Keras, were built to reduce the dimensionality of data.

## WineQuality Dataset

The `data/` directory includes the WineQuality dataset used for training and testing purposes. This dataset contains features related to wine quality, and TensorTango can be used to build models for predicting wine quality based on these features. The dataset can also be found here: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

## XGBoost Classifier

For comparison and ensemble learning, TensorTango includes a custom-built XGBoost classifier class. This class can be used to classify wine quality based on features after reducing dimensionality using both TensorTango's autoencoder and a similar model built with Keras.

## Contributing

Contributions to TensorTango are welcome! Feel free to fork the repository, make improvements, and submit pull requests.
