# Softmax Regression from Scratch

Implementation of Multinomial Logistic Regression (Softmax Regression) from scratch using NumPy for handwritten digit classification.

This project demonstrates the fundamental mechanics of training a neural network without using deep learning frameworks such as TensorFlow or PyTorch.

---

## Project Overview

This project implements a single-layer neural network for multi-class classification using the MNIST dataset.

The model is trained using **Mini-Batch Gradient Descent**, and the following core components are implemented manually:

- Softmax activation function
- Cross-Entropy loss
- One-Hot Encoding
- Gradient-based weight updates
- Training / validation accuracy tracking

The model classifies three digits: **1, 2, and 6**.

---

## Model Architecture

Single-layer neural network:

Input Layer  
- 784 features (28 × 28 grayscale image)

Output Layer  
- 3 classes (digits 1, 2, 6)

Prediction:

argmax(softmax(Wx + b))

---

## Training Details

Dataset: MNIST (subset)

Training samples: provided labeled dataset  
Validation split: 80 / 20  
Test samples: unlabeled dataset

Training method:

- Mini-Batch Gradient Descent
- Batch size: 32
- Maximum epochs: 300
- Learning rate: 1e-6

Loss function:

Cross-Entropy Loss

Evaluation metrics:

- Training Accuracy
- Validation Accuracy

---

## Features

✔ Implemented Softmax function from scratch  
✔ Implemented Cross-Entropy loss manually  
✔ Manual gradient update for weights and biases  
✔ Mini-batch training  
✔ Training / validation loss visualization  
✔ Prediction output generation  

---

## Output

The program outputs:

- Final training epoch
- Learning rate
- Training accuracy
- Validation accuracy
- Prediction results for the test dataset

Example:
End Epoch: 124
Learning rate: 0.000001
Train Accuracy: 93.5
Validation Accuracy: 92.8
