---
categories: blog
layout: post
published: true
date: '2018-01-06 16:13 +0100'
excerpt_separator: <!--more-->
title: Implementing a Neural Network using Numpy
---
# Implementing a Neural Network in Python using Numpy

Here we will see how to implement Neural Network in Python using only numpy library. 
There are plenty of deep learning frameworks currently available for python:  
**[Tensorflow](https://www.tensorflow.org/)**  
**[PyTorch](http://pytorch.org/)**

However it might be a good exercise to implement a Neural Network on our own.

## Deriving equations for Forward and Backward Pass

Fully connected layer can be described by the weight matrix, bias vector and activation function. It transforms N dimensional input data into M dimensional output data. New features will be computed as a linear combination of old features passed through a nonlinearity layer.

![FullyConnectedLayer](/assets/img/NeuralNetworkNumpy/Network.png)

It can be easily seen that those operations can be written as a matrix multiplication as presented below:

![VectorizedForm](/assets/img/NeuralNetworkNumpy/ForwardVector.png)

That is how activations in fully connected layer are computed for a single input sample. To forward multiple samples at once we will simply forward input matrix instead of a single input vector.

![MatrixForm](/assets/img/NeuralNetworkNumpy/ForwardMatrix.png)

To train Neural Network we will need to adjust weights **w** and biases **b** such that we increase the performance of our network (minimize the loss). Let's assume that our layer got gradient information from the next layer telling how to adjust outputs **O** to decrease this loss function. Our layer will have to compute two things:
1. Gradient of loss with respect to its own parameters **w** and **b**, this will be used to adjust those parameters and decrease the loss.
2. Gradient of loss with respect to its inputs **I**, this information will be send to the previous layer.

How to compute derivatives with respect to the parameters?  
To simplify we consider just one output neuron, later we will generalize to the full layer.

![ParametersGradient](/assets/img/NeuralNetworkNumpy/ParamDerivatives.png)

Using chain rule we derive that:

![ChainRule](/assets/img/NeuralNetworkNumpy/ChainRule.png)

We already have all components to compute gradients. 
 - derivative of the **loss** function with respect to the output was delivered to us by the next layer in the network
 - derivative of **f** (activation function) with respect to **x1** can be computed analytically
 - we know input **I**.

Again we would like to find matrix equations for backpropagation. We start with an equation for a single input vector (dot operator indicates elementwise multiplication):


![VectorDerivative](/assets/img/NeuralNetworkNumpy/ParamVectorGrad.png)

If we want to forward multiple samples we will have to average gradients over this minibatch:

![MatrixDerivative](/assets/img/NeuralNetworkNumpy/ParamMatrixGrad.png)

Now we need to find out how to compute the gradient of the **loss** function with respect to the **input**. This information will be passed to the previous layer (backpropagation) and used as a training signal.


![InputDerivative](/assets/img/NeuralNetworkNumpy/InputGradDerivation.png)

We will directly go to the matrix multiplication form:

![InputDerivativeMatrixForm](/assets/img/NeuralNetworkNumpy/InputMatrixGrad.png)


## Implementation of Fully Connected Layer

We will implement fully connected layer as a class.

We want to be able to:

- Stack multiple layers on top of each other to form a neural network
- Choose activation function 
- Choose parameter initialization function
- Choose between Gradient Descent and Stochastic Gradient Descent

### Constructor

```python

class DenseLayer(object):
    def __init__(self, activation_func, input_size_info=None, init_func=WeightInit.xavier_init_gauss,
                 neurons_num=100):
        try:
            input_size = input_size_info.neurons_num
        except AttributeError:
            input_size = input_size_info

        self.neurons_num = neurons_num
        self.activation_func = activation_func

        self.w = np.zeros(shape=(input_size, neurons_num), dtype='float32')
        self.b = np.zeros(shape=(1, neurons_num), dtype='float32')
        init_func(self)

        # In some settings we want to use the mean gradient from
        # multiple mini-batches or even from the whole dataset.
        self.grad_w_acc = GradientAccumulator()
        self.grad_b_acc = GradientAccumulator()

        self.x = None   # Input x
        self.a = None   # x @ w
        self.y = None   # Output
        


```

We will store network parameters inside **self.w** and **self.b** variables.  
We accumulate gradients from minibatches using GradientAccumulator class, it is required because in some  training settings we use the mean gradient from the whole dataset to perform one weight update. 

### Forward pass 

```python
    def forward_pass(self, x):
        self.x = x
        self.a = x @ self.w + self.b
        self.y = self.activation.f(self.a)
		return self.y

```

To implement forward pass we simply use matrix form we derived earlier. In this implementation we divide our parameters into weight matrix **self.w** and bias vector **self.b** this way it will easier to implement L2 regularization only to **self.w** matrix. We need to store input **self.x** as it is required later for the **backward_pass** function. Output from the **forward_pass** function can be used as an input for the next layer.

### Backward pass 

```python
    def backward_pass(self, dl):
        grad_a = grad_loss * self.activation_func.f_grad(self.a)
        grad_b = grad_a.mean(axis=0)
        grad_w = self.x.T @ grad_a / self.x.shape[0]
        grad_x = grad_a @ self.w.T

        # Store gradients from batches until next update is called
        self.grad_b_acc.append(grad_b)
        self.grad_w_acc.append(grad_w)

        return grad_x

```

We use matrix form for backpropagation that we derived earlier. We compute gradients of loss with respect to the parameters **self.grad_w**, **self.grad_b** and input **self.grad_x**. Output from this function can be used as an input for the previous layer. 


## Stacking layers, implementation of the Neural Network

Neural network will store all layers in a sequence and forward output from the previous layer to the next layer during forward pass, it will also compute loss function using output from the last layer. During backpropagation it will forward training signal from the next layer to the previous layer.


```python
class Net(object):
    def __init__(self, objective):
        self.layers = []
        self.objective = objective

    def add_layer(self, layer):
        self.layers.append(layer)

    # Forward pass through all layers
    def predict(self, x):
        p = self.layers[0].forward_pass(x)
        for l in self.layers[1:]:
            p = l.forward_pass(p)
        return p

    # Backward pass through all layers
    def backward_pass(self, p, y):
        grad_loss = self.objective.loss_d(pred=p, targ=y)
        for l in self.layers[::-1]:
            grad_loss = l.backward_pass(grad_loss)

    def loss(self, p, y):
        return self.objective.loss(pred=p, targ=y)
```
Solver will be used to compute weight updates based on the gradients.

```python
 class Solver:
    class Base:
        def __init__(self, network):
            self.network = network
            self.t = 0

        def one_step(self, x, y):
            p = self.network.predict(x)
            self.network.backward_pass(p, y)

        def finish_iteration(self):
            self._update()

        def _update(self):
            for l in self.network.layers:
                delta_w, delta_b = self._compute_update(l)
                l.update(delta_w, delta_b)
            self._advance()

        def _advance(self):
            self.t += 1

        def _compute_update(self, layer):
            raise NotImplementedError
```
To train the network we will use the function **one_step** (providing input data together with the corresponding targets), this function will accumulate the gradients from all samples.  When we are ready to perform an update we will call the function **finish_iteration**.


```python
    class Simple(Base):
        (...)

        def _compute_update(self, layer):
            lr = self.decay_algorithm.learning_rate(self.t)
            grad_w = layer.grad_w_acc.mean_gradient()
            grad_b = layer.grad_b_acc.mean_gradient()
            delta_w = -lr * (grad_w + self.alpha*layer.w)
            delta_b = -lr * grad_b

            return delta_w, delta_b

```
Simple procedure for computing weight update is provided inside the **Solver.Simple** class.


## Training 




