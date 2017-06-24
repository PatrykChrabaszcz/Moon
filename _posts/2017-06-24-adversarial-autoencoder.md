---
categories: blog
layout: post
published: false
date: '2017-06-24 16:30 +0200'
excerpt_separator: <!--more-->
title: Adversarial Autoencoder
---
## Adversarial Autoencoder in Tensorflow

** Github link **
** Dataset link **

Autoencoder neural network consists of two parts.   

- Encoder network: 

	Takes input data *X*, produces its latent representation *z = E(X)*
- Decoder network:

	Takes latent representation *z*, reconstructs input data *X_{r} = D(z) = D(E(X))*
    
Usually dimensionality of *z* is much lower than dimensionality of *X*. In order to have reconstructions that are similar to the input data, encoder network must learn to extract meaningful features from it.

Here we will train autoencoder to compress RGB face images (64x64) into a small vector (128 numbers).

[Adversarial Autoencoder](https://arxiv.org/abs/1511.05644) extends vanilla Autoencoder forcing the distribution of *z=E(X)* to match a prior distribution (We will use gaussian). This way it is possible to use trained autoencoder as a generative model. To do so we have to sample from our prior and pass this sample thorugh Decoder network. 


