---
categories: blog
layout: post
published: false
date: '2017-07-21 23:56 +0200'
excerpt_separator: <!--more-->
title: Neural Network in Pyton using Numpy
---
## Implementing Neural Network in Python using Numpy

Here we will see how to implement Neural Network in Python using only numpy library. 
There are plenty of deep learning frameworks currently available for python. Those include  
"Tensorflow"  
"Lasagne"  
"Torch"  

However it might be a good exercise to implement a Neural Network on our own.

### Implementing fully connected layer

Fully connected layer can be described by the weight matrix, bias vector and activation function. It transforms N dimensional input data into M dimensional output data. New features will be computed as a linear combination of old features passed through a nonlinearity layer.

![FullyConnectedLayer](/assets/img/NumpyNeuralNetwork/NNLayer.png)

It can be easily seen that those operations can be written as a matrix multiplication. It is presented below (Nonlinearity function ommited on this picture)

![VectorizedForm](/assets/img/NumpyNeuralNetwork/VectorForm.png)

That is how activations in fully connected layer are computed for a single input sample. To forward multiple samples at once we will simply forward input matrix instead of a single input vector.

![MatrixForm](/assets/img/NumpyNeuralNetwork/MatrixForm.png)

To train Neural Network we will need to adjust weights **w** and biases **b** such that we increase the performance of our network (minimize loss). Let's assume that our layer got gradient information from the next layer telling how to adjust outputs **O** to decrease this loss function. Our layer will have to compute two things:
1. Gradient of loss with respect to its own parameters **w** and **b**, this will be used to adjust those parameters and decrease the loss.
2. Gradient of loss with respect to its inputs **I**, this information will be send to the previous layer, and used in the same way.

How to compute derivatives with respect to the parameters:

![ParametersGradient](/assets/img/NumpyNeuralNetwork/NNDerivatives.png)

