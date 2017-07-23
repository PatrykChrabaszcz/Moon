---
categories: blog
layout: post
published: false
date: '2017-07-21 23:56 +0200'
excerpt_separator: <!--more-->
title: Neural Network in Pyton using Numpy
---
# Implementing Neural Network in Python using Numpy

Here we will see how to implement Neural Network in Python using only numpy library. 
There are plenty of deep learning frameworks currently available for python. Those include  
"Tensorflow"  
"Lasagne"  
"Torch"  

However it might be a good exercise to implement a Neural Network on our own.

## Deriving equations for Forward and Backward Pass

Fully connected layer can be described by the weight matrix, bias vector and activation function. It transforms N dimensional input data into M dimensional output data. New features will be computed as a linear combination of old features passed through a nonlinearity layer.

![FullyConnectedLayer](/assets/img/NeuralNetworkNumpy/Network.png)

It can be easily seen that those operations can be written as a matrix multiplication. It is presented below (Nonlinearity function ommited on this picture)

![VectorizedForm](/assets/img/NeuralNetworkNumpy/ForwardVector.png)

That is how activations in fully connected layer are computed for a single input sample. To forward multiple samples at once we will simply forward input matrix instead of a single input vector.

![MatrixForm](/assets/img/NeuralNetworkNumpy/ForwardMatrix.png)

To train Neural Network we will need to adjust weights **w** and biases **b** such that we increase the performance of our network (minimize loss). Let's assume that our layer got gradient information from the next layer telling how to adjust outputs **O** to decrease this loss function. Our layer will have to compute two things:
1. Gradient of loss with respect to its own parameters **w** and **b**, this will be used to adjust those parameters and decrease the loss.
2. Gradient of loss with respect to its inputs **I**, this information will be send to the previous layer, and used in the same way.

How to compute derivatives with respect to the parameters. To simplify we consider just one output neuron, later we will generalize to the full layer.

![ParametersGradient](/assets/img/NeuralNetworkNumpy/ParamDerivatives.png)

Using chain rule we derive that:

![ChainRule](/assets/img/NeuralNetworkNumpy/ChainRule.png)

We already have all components to compute gradients. Derivative of **Loss** with respect to output was delivered to us by the next layer in the network, derivative of **f** (activation function) with respect to **x1** can be computed analytically and we know input **I**.

Again we would like to find matrix equations for backpropagation. Starting with an equation for a single input vector (one sample). Dot operator indicates elementwise multiplication:


![VectorDerivative](/assets/img/NeuralNetworkNumpy/ParamVectorGrad.png)

And if we want to forward multiple samples we will have to average gradients over this minibatch:

![VectorDerivative](/assets/img/NeuralNetworkNumpy/ParamMatrixGrad.png)

Now we need to find out how to compute the gradient of **Loss** with respect to the **input**. This information will be passed to the previous layer (backpropagation) and used as a training signal.


![InputDerivative](/assets/img/NeuralNetworkNumpy/InputGradDerivation.png)

We will directly go to the matrix multiplication form:

![InputDerivativeMatrixForm](/assets/img/NeuralNetworkNumpy/InputMatrixGrad.png)


## Implementation of Fully Connected Layer

We will implement fully connected layer as a class.

We want to be able to:

- Stack multiple layers on top of each other to form NeuralNetwork
- Choose activation function 
- Choose parameter initialization function
- Choose between Gradient Descent and Stochastic Gradient Descent

### Constructor

- **activation**: object that represents nonlinearity function. This object will be used to compute **f(a)** as well as **f'(a)**.
- **input_size_info**: describes how many inputs this layer has. We want to simplify usage so we make it possible to derive this information from the previous layer.
- **init_func**: this function is used to initialize parameters in this layer.
- **neurons_num**: number of neurons in this layer

```python

class DenseLayer(object):
    def __init__(self, activation, input_size_info=None, init_func=WeightInit.xavier_init_gauss,
                 neurons_num=100):
        try:
            input_size = input_size_info.neurons_num
        except AttributeError:
            input_size = input_size_info

        self.neurons_num = neurons_num
        self.w = np.zeros(shape=(input_size, neurons_num), dtype='float32')
        self.b = np.zeros(shape=(1, neurons_num), dtype='float32')
        init_func(self)

        self.x = None                       # Input
        self.dx = None                      # Gradient w.r.t. input (List for data chunks)
        self.dw = np.zeros_like(self.w)     # Gradient w.r.t. weights (List for data chunks)
        self.db = np.zeros_like(self.b)     # Gradient w.r.t. biases (List for data chunks)

        # Lists used for gradient averaging
        self.dx_list = []
        self.dw_list = []
        self.db_list = []
        self.s_list = []
        self.a = None   # x @ w
        self.y = None   # Output
        self.activation = activation
        


```

We need to store input value **self.x** when we do forward propagation, it will be later used in backpropagation part. We also store gradients of our parameters **self.dw**, **self.db** after backpropagation,  those will be later used by the optimizer to update the network . 

We might want to use our implementation with SGD (Stochastic Gradient Descent) where we update parameters after each minibatch or with Gradient Descent where we update parameters after passing all the training data. To use Gradient Descent we will have to store history of minibatch gradients and average over that before we perform update (For bigger network and bigger datasets this implementation will not be memory efficient, this can be improved by storing only the number of minibatches and current mean gradient)

### Forward pass 

```python
    def forward_pass(self, x):
        self.x = x
        self.a = x @ self.w + self.b
        self.y = self.activation.f(self.a)
		return self.y

```