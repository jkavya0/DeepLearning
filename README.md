# Deep Learning Exercises

## Overview

This repository demonstrates a complete deep learning workflow, starting from NumPy-based implementations of neural networks to PyTorch-based image classification.

The project focuses on building core deep learning components from first principles (forward/backward propagation, optimizers, CNNs, RNNs) and applying them to a real-world classification problem.

---

## Covered Topics

* NumPy-based numerical computing
* Neural networks from scratch (forward & backward propagation)
* Loss functions and optimizers
* Convolutional Neural Networks (CNNs)
* Regularization techniques (Dropout, BatchNorm, L1/L2)
* Recurrent Neural Networks (RNNs)
* PyTorch-based image classification

---

## Exercises

### Exercise 0 – NumPy & Data Handling

* Generated image patterns (checkerboard, circles, RGB spectrum)
* Built an `ImageGenerator` class
* Implemented batching, resizing, shuffling, and augmentation

---

### Exercise 1 – Neural Networks (From Scratch)

* Implemented:

  * Fully Connected layer
  * ReLU activation
  * SoftMax
  * CrossEntropyLoss
  * SGD optimizer
* Built a modular neural network framework
* Implemented forward and backward propagation

---

### Exercise 2 – CNN & Optimization

* Implemented:

  * Convolutional layer
  * Pooling layer
  * Flatten layer
* Weight initialization:

  * Xavier, He, Uniform, Constant
* Optimizers:

  * SGD with Momentum
  * Adam

---

### Exercise 3 – Regularization & RNN

* Regularization techniques:

  * L1, L2
  * Dropout
  * Batch Normalization
* Activation functions:

  * TanH, Sigmoid
* Implemented:

  * Elman Recurrent Neural Network (RNN)

---

### Exercise 4 – PyTorch Classification

* Built a custom dataset and preprocessing pipeline
* Applied data augmentation
* Implemented a ResNet-style architecture
* Training pipeline:

  * Validation
  * Early stopping
* Application:

  * Solar-cell defect classification

---

## How to Run

### 1. Install dependencies

```
pip install 
```

### 2. Run an exercise

```
python src/main.py
```

*(Modify entry point depending on your implementation)*

---

## Skills Demonstrated

* Deep learning fundamentals (from-scratch implementation)
* CNN and RNN architectures
* Optimization techniques (SGD, Momentum, Adam)
* Regularization and normalization
* PyTorch model development
* Data preprocessing and augmentation

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* SciPy
* scikit-image
* PyTorch

---

## Key Insight

This project emphasizes understanding deep learning internals by implementing core components from first principles before transitioning to high-level frameworks like PyTorch.

---

## Author

Kavya Jayaramaiah
