---
categories: blog
layout: post
published: false
date: '2018-04-02 18:55 +0200'
excerpt_separator: <!--more-->
title: Paper Notes
---
## RNN Dropout

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
    
   
### Dropout improves recurrent neural networks for handwriting recognition (5 Nov 2013