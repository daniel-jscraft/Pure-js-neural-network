# Vanilla Neural Network on the browser with Javascript

This project shows how to train a neural network on the browser using Vanilla Neural Network(No library) 
then we compare its performance with a neural network trained using Tensorflow.js.

The model is trained on the MNIST dataset, which contains images of handwritten digits. 

This project lets you train a handwritten digit recognizer using three different model approaches:
- Fully Connected Neural Network - Vanilla Artificial Neural Network(My own implementation)
- Fully Connected Neural Network (also known as a DenseNet) Using TensorFlow.js
- Convolutional Neural Network(also known as a ConvNet or CNN) Using TensorFlow.js

Note: currently the entire dataset of MNIST images is stored in a PNG image


![Alt text](predictions.png?raw=true "inference digits")


## Getting Started
run `yarn install` to install all the dependencies.

run `yarn watch` to start the development server


## Implementing a Backpropagation algorithm from scratch
In this repository, you will learn how to implement the backpropagation algorithm from scratch using Javascript.

What is Backpropagation? Back-propagation is the essence of neural net training. 
It is the method of fine-tuning the weights of a neural net based on the error rate obtained in the previous epoch 
(i.e., iteration). Proper tuning of the weights allows you to reduce error rates and to make the model reliable by increasing its generalization.

Backpropagation is a short form for "backward propagation of errors." It is a standard method of training artificial neural 
networks. This method helps to calculate the gradient of a loss function with respects to all the weights in the network.

The backpropagation algorithm consists of two phases:

* The forward pass where we pass our inputs through the network to obtain our output classifications.
* The backward pass (i.e., weight update phase) where we compute the gradient of the loss function and use this information to iteratively apply 
the chain rule to update the weights in our network.