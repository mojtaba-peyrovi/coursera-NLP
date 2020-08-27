

## Coursera NLP by deeplearning.ai

### Course 3: Natural Language Processing with Sequence Model:
 __Week1 -  Neural Networks for Sentiment Analysis:__
Neural Network is a computational method which tries to mimic the way human brains find patterns. NNT is a very powerful tool for AI applications.
Here is an example:
<img src="sample-neural-network.JPG">
Then network above has an input layer, two hidden layers, and an three output units.
The input layer has n features ( predictors) we call them x as individuals but also can be shown as a vector of X with the size of n.
Here is how we move forward:
**a superscript i:** the ith computational layer. for example "a" superscript 0 is X. 
**z superscript i:** the calculation of the weighted values which is going to be used in the activation function later. the value can be achieved by the sum of W (weight matrix) of the current layer, multiplied by the "a" matrix of the previous layer.
**W superscript i:** The weight matrix of layer i
** g superscript i:** It is the activation function of layer i
Finally we get the values of layer i by passing z into the activation function g. 
<img src="forward-propagation.JPG">
Here is an example of a neural network being used in sentiment analysis:
<img src="sample-sentiment-analysis-nnt.JPG">

In order to create the sentiment analysis in NNT, first we list all vocabulary in the corpus, then assign an integer to each, then replace the words in each tweet with their integer values, and at the end we add zeros to all tweets to make them as long as the longest tweet. This process is called **Padding.**
<img src="padding.JPG">
### Trax Library for NNT:
Trax is built on the top of Tensorflow and it's simple to use. Here is an example of how to build the architecture:
<img src="trax-architecture.JPG">

[Here](https://trax-ml.readthedocs.io/en/latest/) is the link to Trax documentation.
[Here](https://jax.readthedocs.io/en/latest/index.html "JAX") is the link to JAX documentation.

#### What is JAX?
Jax Trax uses TF as its backend engine, and also uses JAX to speed up the computations. JAX can be considered as an optimized version of Numpy and we even import the numpy from trax like this:
```python
import trax.fastmath.numpy as np
```
Another example is importing numpy.ndarray like this:
```python
jax.interpreters.xla.DeviceArray
```




> Written with [StackEdit](https://stackedit.io/).