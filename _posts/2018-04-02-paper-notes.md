---
categories: blog
layout: post
published: false
date: '2018-04-02 18:55 +0200'
excerpt_separator: <!--more-->
title: Paper Notes
---
## RNN Dropout

### Improving neural networks by preventing co-adaptation of feature detectors (3 Jul 2012) (Arxiv)
- Set of possible weights that perform well on the train data is huge, but not many of them will perform well on the test data.
- Can be seen as a low cost method for building an ensemble of networks that share some of the weights.
- Instead of L2 normalization they put an upper bound L2 norm constraint of the incoming weight wector for each hidden unit (I think that by "unit" they mean "layer").
- They fix dropout probability to 0.5 

"Using a constraint" {limit on max L2 norm} "rather than a penalty" {they mean L2 regularization penalty} "prevents weights from growing very large no matter how large the proposed weight-update is. This makes it possible to start with a very large learning rate which decays during learning, thus allowing a far more thorough search of the weight-space than methods that start with small weights and use a small learning rate.
"In networks with a single hidden layer of N units and a “softmax” output layer for computing the probabilities of the class labels, using the mean network is exactly equivalent to taking the geometric mean of the probability distributions over labels predicted by all 2^N possible networks."

Questions:
- When they talk about single hidden layer of N units network do they mean a network without nonlinearity?


### Dropout improves recurrent neural networks for handwriting recognition (5 Nov 2013) (ICFHR 2014)
- Authors claim it's a first work that applies dropout to RNNs.
- Architecture used is a hybrid of CNN and RNN
- For CNNs (low amount of weights) dropout samples more models than dropconnect.
- Claims that in some previous work CNN dropout had to be smaller than 0.5, otherwise performance decreases.
- Claims up to 20% improved performance when dropout applied only to the last LSTM layer.
- Claims that dropout acts as L2 regularization but hyperparameter is easier to tune.
"In  our  approach,  dropout  is  carefully  used  in  the network so that it does not affect the recurrent connections."
"(...) dropout is applied only to feed-forward connections  and  not to recurrent connections"
"In our experiments, however, we find out that ReLU can not give good performance in LSTM cells, hence we keep tanh for the LSTM cells and sigmoid for the gates"
"We found that  dropout  at  3  LSTM  layers  is  generally  helpful,  however the  training  time  is  significantly  longer  both  in  term  of  the number  of  epochs  before  convergence  and  the  CPU  time  for each epoch."
"(...) hypothesis  that  dropout encourages the units to emit stronger activations. Since some units were randomly dropped during  training,  stronger  activations  might  make  the  units more  independently  helpful,  given  the  complex  contexts  of other  hidden  activations."
"The word recognition networks  with dropout at  the topmost layer significantly  reduces  the  CER  and  WER  by  10-20%,  and  the performance  can  be  further  improved  by  30-40%  if  dropout is  applied  at  multiple  LSTM  layers."
"Extensive  experiments  also  provide  evidence  that dropout  behaves  similarly  to  weight  decay,  but  the  dropout hyper-parameter  is  much  easier  to  tune  than  those  of  weight decay.

### Recurrent neural network regularization (8 Sep 2014) (ICLR 2015) Zaremba et al. 
- First approach for using dropout in RNNs (LSTM)
- Dropout applied only to the non recurrent connections
- Applies dropout on the RNN input and output (L+1 dropouts per timestep) 
- Dropout regularized model gives a similar performance to an ensemble of 10 non-regularized models. 
- Same as "Dropout improves recurrent neural networks for handwriting recognition".

"Unfortunately, dropout (...) does not work well with RNNs."
"Bayer et al. (2013) claim that conventional dropout does not work well with RNNs because the recurrence amplifies noise, which in turn hurts learning."
"Standard dropout perturbs the recurrent connections, which makes it difficult for the LSTM to learn to store information for long periods of time."
"By not using dropout on the recurrent connections, the LSTM can benefit from dropout regularization without sacrificing its valuable memorization ability."
"The main idea is to apply the dropout operator only to the non-recurrent connections"

Questions: 
- Is dropout mask different for different timesteps? Probably yes.
    
   

