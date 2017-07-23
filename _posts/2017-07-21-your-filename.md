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

We will store network parameters inside **self.w** and **self.b** variables. We store gradients from minibatches and average over them before we perform an update. 

### Forward pass 

```python
    def forward_pass(self, x):
        self.x = x
        self.a = x @ self.w + self.b
        self.y = self.activation.f(self.a)
		return self.y

```

To implement forward pass we simply use matrix form we derived earlier. In this implementation we divide our parameters into weight matrix **self.w** and bias vector **self.b** this way it will easier to implement L2 regularization only to **self.w** matrix. We need to store input **self.x** as it is required later in backward_pass function. Output from this function can be used as an input for the next layer.

### Backward pass 

```python
    def backward_pass(self, dl):
        da = dl * self.activation.f_d(self.a)
        self.db = da.mean(axis=0)
        self.dw = self.x.T @ da / self.x.shape[0]
        self.dx = da @ self.w.T
        s = self.x.shape[0]

        # Store gradients from batches until next update is called
        self.db_list.append(self.db)
        self.dw_list.append(self.dw)
        self.dx_list.append(self.dx)
        self.s_list.append(s)

		return self.dx

```

We use matrix form for backpropagation that we derived earlier. We compute gradients of loss **dl** with respect to the parameters **self.dw**, **self.db** and input **self.dx**. Output from this function can be used as an input for the previous layer. 


## Stacking layers, implementation of Neural Network

Neural network will store all layers in a sequence and forward output from the previous layer to the next layer during forward pass, it will also compute loss function using output from the last layer. During backpropagation it will forward training signal from the next layer to the previous layer.


```python
class Net(object):
    def __init__(self, solver, objective):
        self.layers = []
        self.solver = solver
        self.objective = objective

    def add_layer(self, layer):
		self.layers.append(layer)

```
Solver will be used to compute weight updates based on the gradients.

```python
    # One way to train is to call one_interation()
    # Weights will be updated after single batch
    def one_iteration(self, x, y):
        p = self.predict(x)
        self._backward_pass(p, y)
        self._update()

    # Other way to train is to call one_step() until
    # all data is forwarded, then call finish_iteration()
    def one_step(self, x, y):
        p = self.predict(x)
        self._backward_pass(p, y)

    def finish_iteration(self):
        self._update()
        
    def _update(self):
        for l in self.layers:
            l.average_gradients()
			self.solver.update_layer(l)
```
We can train our network updating parameters in each iteration using **one_iteration(...)** or using **one_step** we can compute gradiens for all training data and then average them and apply update using **finish_iteration(...)** function


```python
# Forward pass through all layers
    def predict(self, x):
        p = self.layers[0].forward_pass(x)
        for l in self.layers[1:]:
            p = l.forward_pass(p)
        return p

    # Backward pass through all layers
    def _backward_pass(self, p, y):
        dl = self.objective.loss_d(pred=p, targ=y)
        for l in self.layers[::-1]:
			dl = l.backward_pass(dl)

```

Funcion **predict(...)** will simply return output from the last layer, if we use our network for classification this output will represent logit values for each class. It will be used in **_backward_pass(...)** to compute derivatives using objective loss. For classification we will use softmax cross entropy.













