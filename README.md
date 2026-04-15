Deep Learning Exercises
This repository contains my implementations and coursework for the Deep Learning Exercises course.
The work in this repository starts with Python and NumPy fundamentals, then builds up a small deep learning framework from scratch, extends it with convolutional and recurrent layers, and finally applies PyTorch to a real image classification problem.

Repository Overview
The exercises in this repository cover the following topics:

NumPy fundamentals and array-based image generation
Data handling, batch generation, and data augmentation
Building neural-network components from scratch
Forward and backward propagation
Loss functions and optimizers
Convolutional neural networks
Weight initialization strategies
Regularization methods
Recurrent neural networks
PyTorch-based image classification
Exercise Overview
Exercise 0 – NumPy Tutorial
Implemented basic NumPy-based pattern generation and data handling utilities.

Main tasks included:

Creating image patterns such as checkerboards, circles, and RGB spectra
Implementing visualization methods
Building an ImageGenerator class for loading labeled image data
Supporting batching, resizing, shuffling, mirroring, and rotation-based augmentation
Exercise 1 – Neural Networks
Built the core parts of a simple layer-based deep learning framework from scratch.

Implemented components include:

SGD optimizer
BaseLayer
FullyConnected layer
ReLU
SoftMax
CrossEntropyLoss
NeuralNetwork class for training and testing
This exercise focused on understanding forward propagation, backpropagation, parameter updates, and modular network design.

Exercise 2 – Convolutional Neural Networks
Extended the framework with important CNN building blocks and improved optimization methods.

Implemented components include:

Weight initializers: Constant, UniformRandom, Xavier, and He
Advanced optimizers: SGD with Momentum and Adam
Flatten layer
Convolutional layer
Pooling layer
This exercise focused on adding support for convolutional architectures and improving training stability.

Exercise 3 – Regularization and Recurrent Layers
Expanded the framework with regularization strategies and sequence models.

Implemented components include:

Base optimizer support for regularization
L1 and L2 regularizers
Dropout
Batch Normalization
Activation functions: TanH and Sigmoid
Elman RNN
Optional extensions such as LeNet and LSTM
This exercise focused on improving generalization, handling training/testing phase differences, and learning sequence modeling fundamentals.

Exercise 4 – PyTorch for Classification
Applied deep learning concepts using PyTorch for a real classification task.

Main tasks included:

Building a custom ChallengeDataset
Implementing preprocessing and augmentation pipelines
Creating a ResNet-style architecture
Implementing a training pipeline with validation and early stopping
Training and tuning a model for solar-cell defect classification
This exercise moved from framework-building to practical model development with PyTorch.

Skills Demonstrated
Through these exercises, I worked on:

Numerical programming with NumPy
Object-oriented design in Python
Implementing deep learning layers and training pipelines from scratch
CNN and RNN fundamentals
Regularization and normalization techniques
Model training and debugging
PyTorch-based computer vision workflows
Technologies Used
Python
NumPy
Matplotlib
SciPy
scikit-image
PyTorch
Purpose
This repository documents my hands-on learning process in deep learning, from low-level implementations of neural-network operations to modern PyTorch-based classification workflows.
