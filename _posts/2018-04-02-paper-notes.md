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
- Went from 160 test errors to 110 errors on MNIST using dropout 0.5 on hidden activations and 0.2 on the input.
"Using a constraint" {limit on max L2 norm} "rather than a penalty" {they mean L2 regularization penalty} "prevents weights from growing very large no matter how large the proposed weight-update is. This makes it possible to start with a very large learning rate which decays during learning, thus allowing a far more thorough search of the weight-space than methods that start with small weights and use a small learning rate."
"It is also possible to adapt the individual dropout probability of each hidden or input unit by comparing the average performance on a validation set with the average performance when the unit is present.  This makes the method work slightly better."
"For datasets in which the required input-output mapping has a number of fairly different regimes, performance can probably be further improved by making the dropout probabilities be a learned function of the input, thus creating a statistically efficient “mixture of experts” in which there are combinatorially many experts, but each parameter gets adapted on a large fraction of the training data."
"Dropout can be seen as an extreme form of bagging in which each model is trained on a single case and each parameter of the model is very strongly regularized by sharing it with the corresponding parameter in all the other models. This is a much better regularizer than the standard method of shrinking parameters towards zero."
"One reason why dropout gives major improvements over backpropagation is that it encourages each individual hidden unit to learn a useful feature without relying on specific other hidden units to correct its mistakes."

### Dropout: A simple way to prevent neural networks from overfitting (Nov 2013) (JMLR 2014) Srivastava et al. (Hinton)
"With  unlimited  computation,  the  best  way  to  “regularize”  a  fixed-sized  model  is  to
average the predictions of all possible settings of the parameters, weighting each setting by its posterior probability given the training data. 
- Seems that optimal probability for dropping hidden neurons is close to 0.5 while for the input is closer to 0.0 than 0.5
- About "Fast Dropout Trianing": proposed a method for speeding up dropout by marginalizing dropout noise
- Dropout better than plain network but worse than bayesian network
- Clam that dropout helps also in convolutional layers 
- Show figures with CNN filters that are less noisy and more independent when using dropout.
- For FC networks show that only few neurons have high activations while most of them stay at 0 
- Compare weight averaging to MC sampling, given enough samples MC gets a little bit better, but weight averagining is still a good approximation
- Propose another dropout type: use Gaussian distribution instead of Bernoulli
- When using dropout learning rate should be 10-100x bigger or momentum should be 0.99 instead of 0.9
"One particular form of regularization was found to be especially useful for dropout constraining  the  norm  of  the  incoming  weight  vector  at  each  hidden  unit  to  be  upper bounded by a fixed constant c" {typical values for c 3-4} ". (...). This is also called max-norm regularization since it implies that the maximum value that the norm of any weight can take is c.
" In  dropout,  each  model  is  weighted  equally,  whereas  in  a  Bayesian  neural network each model is weighted taking into account the prior and how well the model fits the data, which is the more correct approach."
"We  found  that  as  a  side-effect  of  doing  dropout,  the  activations  of  the  hidden  units become sparse, even when no sparsity inducing regularizers are present"
"As the size  of  the  data  set  is  increased,  the  gain from doing dropout increases up to a point and then declines." {MNIST experiment}
It can be seen that around k=50, the Monte-Carlo method becomes as good as the approximate method.  Thereafter, the Monte-Carlo method is slightly better than the approximate method but well within one standard deviation of it. This suggests that the weight scaling method is a fairly good approximation of the true model average"
"One of the drawbacks of dropout is that it increases training time.  A dropout network typically takes 2-3 times longer to train than a standard neural network of the same architecture"

### Dropout improves recurrent neural networks for handwriting recognition (5 Nov 2013) (ICFHR 2014) Pham et al.
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

Gal about this paper: "They conclude that dropout in recurrent layers disrupts the RNN’s ability to model sequences, and that dropout should be applied to feed-forward connections and not to recurrent connections."

# Understanding dropout
TODO. Some theory behind dropout 

# Ivestigation of recurrent neural network architectures and learning methods for spoken language understanding
- Bluche claims that in this paper they  demonstrated "that  dropout  can  significantly  increase  the  generalization  capacity  in  architectures  with  recurrent  layers"

# Regularization and nonlinearities for neural language models: when are they needed?
- Gal claims that here authors reason that noise added in the recurrent connections of an RNN leads to model instabilities, and onlz add dropout to the decoding part.

# On fast dropout and its applicability to recurrent networks

Gal about this paper: "They reason that with dropout, the RNN’s dynamics change dramatically, and that dropout should be applied to the <<non-dynamic>> parts of the model – connections feeding from the hidden layer to the output layer"


# Regularization of Neural Networks using DropConnect


### Where to Apply Dropout in Recurrent Neural Networks for Handwriting Recognition? (ICDAR 2015) Bluche et al.
- Same people as "Dropout improves rnn for handwriting recognition"
- Investigates bidirectional LSTM 
- Checks three possible ways to add dropout to the model (input, hidden recurrent, hidden forward)
"In this paper, we show that further improvement can be achieved by implementing  dropout  differently,  more  specifically  by  applying it  at  better  positions  relative  to  the  LSTM  units"
"A   false   rumour   stated that  dropout  in  combination  with  parameterized  convolutions should not work well, but some attempts demonstrated that it actually  is  a  source  of  improvement"
"Moreover, adding dropout before, inside and after the LSTM at the same time is not as good as choosing the right position."
"among   all   relative   positions   to   the   LSTM,   when dropout  is  applied  to  every  LSTM,  placing  it  after was the worst choice in all six configurations"
"It seems like dropout is best close to the inputs and outputs of the network"
Gal about this paper: "They provide mixed results, not showing significant improvement on existing technique"

### Recurrent neural network regularization (8 Sep 2014) (ICLR 2015) Zaremba et al. 
- Authors claim that it is the first approach for using dropout in RNNs (LSTM)
- Dropout applied only to the non recurrent connections
- Applies dropout on the RNN input and output (L+1 dropouts per timestep) 
- Dropout regularized model gives a similar performance to an ensemble of 10 non-regularized models. 
- Same as "Dropout improves recurrent neural networks for handwriting recognition".
- Applied for Language Modeling and Speech Recognition

"Unfortunately, dropout (...) does not work well with RNNs."
"Bayer et al. (2013) claim that conventional dropout does not work well with RNNs because the recurrence amplifies noise, which in turn hurts learning."
"Standard dropout perturbs the recurrent connections, which makes it difficult for the LSTM to learn to store information for long periods of time."
"By not using dropout on the recurrent connections, the LSTM can benefit from dropout regularization without sacrificing its valuable memorization ability."
"The main idea is to apply the dropout operator only to the non-recurrent connections"

Questions: 
- Is dropout mask different for different timesteps? Probably yes.
	Gal says: "In comparison, Zaremba’s dropout variant replaces zx with the time-dependent ztx which is sampled anew every time step"

### A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (16 Dec 2015) (NIPS 2016)
- Dropout as an approximate Bayesian Inference
- Dropout only on inputs and outputs still leads to overfitting
- Proposes variational RNN that uses the same dropout at each timestep
- Improve Zaremba's SOTA results on Penn Treebank.
- Show that MC dropout with 1000 samples is better than weights averaging 
- Also proposes to use dropout on embeddings for word models
- Dropout amkes it possible to use bigger networks 
- Ensembling Variational RNNs futher improves performance
"(...) dropout can be interpreted as a variational approximation to the posterior of a Bayesian neural network (NN)."
"Empirical results have led many to believe that noise added to recurrent layers (connections between RNN units) will be amplified for long sequences, and drown the signal. Consequently, existing research has concluded that the technique should be used with the inputs and outputs of the RNN alone."
"In the new dropout variant, we repeat the same dropout mask at each time step for both inputs, outputs,
and recurrent layers (drop the same network units at each time step). This is in contrast to the existingad hoc techniques where different dropout masks are sampled at each time step for the inputs and
outputs alone (no dropout is used with the recurrent connections since the use of different masks with these connections leads to deteriorated performance."
"(...) we show that it is possible to derive a variational inference based variant of dropout which successfully regularises such parameters" {meaning recurrent} ", by grounding our approach in recent theoretical research."
"Implementing our approximate inference is identical to implementing dropout in RNNs with the same network units dropped at each time step , randomly dropping inputs, outputs, and recurrent connections.  This is in contrast to existing techniques, where different network units would be dropped at different time steps, and no dropout would be applied to the recurrent connections."
"A common approach for regularisation is to reduce model complexity (necessary with the non-regularised LSTM). With the Variational models however, a significant reduction in perplexity is achieved by using larger models."
"Yet it seems that with no embedding dropout, a higher dropout probability within the recurrent layers leads to overfitting! This presumably happens because of the large number of parameters in the embedding layer which is not regularised. Regularising the embedding layer with dropout probability pE= 0.5 we see that a higher recurrent layer dropout probability indeed leads to increased robustness to overfitting, as expected. 
"(...) we assess the importance of weight decay with our dropout variant. Common practice is to remove weight decay with naive dropout.  Our results suggest that weight decay plays an important role with our variant (it corresponds to our prior belief of the distribution over the weights)."

Merity calls it "Variational Dropout"

### Hybrid speech recognition with deep bidirectional LSTM
- Presents noise injection as referenced by Moon in his dropout paper.

Mono says: "In weight noise injection, zero mean Gaussian noise is added to the weights when computing the gradient."


### RnnDrop: A Novel Dropout for RNNs in ASR (Dec 2015) (ASRU 2015) Moon et al. (Samsung)
- Claims that previous dropout only on inputs and outputs is worse than "weight noise injection" regularization.
- Dropout mask for a sequence drawn once for all timesteps
- Different dropout masks for different sequences within minibatch.
- Gal claims it is parallel to their work about variational RNN
- Used for phoneme  recognition (TIMIT) and speech recognition (Wall Street Journal WSJ) 

"(...) there have been some attempts to apply the dropout, a popular  regularization scheme for the feed forward neural networks, to RNNs, but they do not perform as well as other regularization scheme such  as weight noise injection."
"(...) rnnDrop, which drops out hidden nodes instead of some connections. That is, rnnDrop drops both the non-recurrent and recurrent connectionsthat are connected to the dropped nodes.More importantly,the dropout mask is randomly selected for each input sequenceand fixed throughout the sequence. "
"Although DBLSTM networks trained with rnnDropshowed superiorperformances, it took much longer to train them"

Gal about this work: "They randomly drop elements in the LSTM’s internal cell ct and use the same mask at every time step.  This is the closest to our proposed approach (although fundamentally different to the approach we suggest)"
Gal when reproducing results from this work: "Adding our embedding dropout, the model performs much better, but still underperforms compared to applying dropout on the inputs and outputs alone"

### Recurrent dropout without memory loss
- Merity sais it is reminescent to Zoneout 
Merity said: "Another approach is to regularize the network through limiting updates to the RNN’s hidden state. (...) authors drop updates to network units, specifically the input gates of the LSTM, in lieu of the units themselve."

### Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations
Merity sais about this work: "(...) updates to the hidden state may fail to occur for randomly selected neurons"



### Regularizing and Optimizing LSTM Language Models (Aug 2017) (Arxiv) Merity et al. (Salesforce)
- SOTA on Penm Treebank and WikiText-2.
- Adding neural cache to the model futher improves the performance.
- They use  averaged SGD (ASGD) for training. Their method is called NT-ASGD which autotunes T
- Use random sequence length but compare it with very naive implementation where sequence starting point is not randomized. 
- When using longer sequences they use bigger (Need to look at the code to confirm because it is not clearly stated in the paper) learning rate to make influence of short and long sequences equal.
- Uses embedding dropout as Gal 
- Uses weight tying
- Uses Temporal Activation Regularization TAR (also Merity idea) to penalize the model when it produces large changes in hidden state between timesteps.
- AR penalizes hidden activations that are significantly larger than 0
- AR and TAR only applied to the final LSTM output
- They use gradient cliping (0.25)
- For each technique they use they also check the performance when this technique is disabled. The most influencial one is hidden to hidden weight decay.

"We propose the weight-dropped LSTM which uses DropConnect on hidden-to-hidden weights as a form of recurrent regularization"
"(...) we introduce NT-ASGD, a variant of the averaged stochastic gradient method, wherein the averaging trigger is determined using a non-monotonic condition as opposed to being tuned by the user."
"The weight-dropped LSTM applies recurrent regularization through a DropConnect mask on the hidden-to-hidden recurrent weights.  Other strategies include the use of randomized-length backpropagation through time (BPTT), embedding dropout, activation regularization (AR), and temporal activation regularization (TAR)."
"As no modifications are required of the LSTM implementation these regularization strategies are compatible with black box libraries, such as NVIDIA cuDNN, which can be many times faster than naïve LSTM implementations."
" In the context of word-level language modeling, past work has empirically found that SGD outperforms other methods" {Adam, SGD with Momentum etc.} " in not only the final loss but also in the rate of convergence"
"We propose a variant of ASGD where T is determined on the fly through a non-monotonic criterion and show that it achieves better training outcomes compared to SGD."
"As the dropout operation is applied once to the weight matrices, before the forward and backward pass, the impact on training speed is minimal and any standard RNN implementation can be used, including inflexible but highly optimized black box LSTM implementations such as NVIDIA’s cuDNN LSTM"
"As the same weights are reused over multiple timesteps, the same individual dropped weights remain dropped for the entirety of the forward and backward pass"
"(...) we use variational dropout for all other dropout operations, specifically using the same dropout mask for all inputs and outputs of the LSTM within a given forward and backward pass. Each example within the mini-batch uses a unique dropout mask, rather than a single dropout mask being used over all examples, ensuring diversity in the elements dropped out."
"In addition, L2 decay can be used on the individual unit activations and on the difference in outputs of an RNN at different time steps; these strategies labeled as activation regularization (AR) and temporal activation regularization (TAR) respectively"
"In past work, pointer based attention models have been shown to be highly effective in improving language modeling "

### Recurrent dropout without memory loss (Mar 2016) (Arxiv) Semenitua 


# General 
#Maxout networks


# RNNs 
### On the State of the Art of Evaluation in Neural Language Models
- Very nice visualization for hyperparameters
Marity says: "(...) apply extensive hyperparameter search to an LSTM based language modeling implementation, analyzing the sensitivity of RNN based language models to hyperparameters. Unlike our work, they use a modified LSTM, which caps the input gate it to be min(1−ft, it), use Adam withβ1= 0 rather than SGD or ASGD, use skip connections between LSTM layers, and use a black box hyperparametertuner for exploring models and settings."
### Improving neural language models with a continuous cache.
 - To check what is a cache model
 - Merity sais that it can be added on top of already trained model
 
### Martin Sundermeyer, Ralf Schlüter, and Hermann Ney. LSTM neural networks for language modeling. In
INTERSPEECH
, 201

### Nal Kalchbrenner and Phil Blunsom. Recurrent continuous translation models. In
EMNLP
, 2013

#Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In
NIPS
, 2014


# Batch norm etc

Other forms of regularization explicitly act upon activa-
tions such as batch normalization (
Ioffe & Szegedy
,
2015
),
recurrent batch normalization (
Cooijmans et al.
,
2016
), and
layer normalization (
Ba et al.
,
2016
). These all introduce
additional training parameters and can complicate the trai
n-
ing process while increasing the sensitivity of the model.
