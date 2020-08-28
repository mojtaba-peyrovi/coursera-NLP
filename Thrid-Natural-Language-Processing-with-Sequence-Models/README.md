

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

### Dense vs ReLu:
Dense layer means when all nodes are fully connected to the neurons of the next layer.
<img src="dense_layers.JPG">
ReLu layer typically follows a dense layer and transforms any negative values to zero before sending them on to the next layer.
<img src="relu-layer.JPG">

### Serial Layer:
A composition of dense and activation layers (sublayers) that operate in a sequence to implement the forward propagation calculation. we can see it as having the whole model being compressed in one layer.
<img src="serial-layer.JPG">

### Embedding Layer:
In NLP tasks we normally include an embedding layer which takes the vocabulary words with an index assigned to each, and maps it with a set of parameters in a determined dimension. Then we need to train the model to bring the best values for the embeddings to have the best performance. For the embedding layer we have a matrix of weights of size equal **(vocabulary size x embedding dimension)**.
The size of the embedding can be treated as a hyperparameter in the model, (parameters we manually add similar to learning rate)

### Mean Layer:
It simply comes after embedding layer, and calculates the mean of the embedding parameters for each word (row). for example if we have 4 vocabularies, and 100 embedding dimensions, the output of embedding output size is (4x100) and it feeds the mean layer and returns a vector of size 100. 
<img src="mean-layer.JPG">

As a summary:
>Embedding is trainable using an embedding layer
>Mean layer gives a vector representation

### Training the NNT:
For training, we need to calculate the gradient (differential). Trax can do it simply as seen here:
<img src="gradients-in-trax.JPG">
to train the the model we use grad() method. This method takes two sets of parameters. First set is parameters for the grad function, the second set for the function returned by grad. Then all we need to do, is to iterate over grads and each time deduct the weights by the grad times learning rate (alpha) until we get the best convergence. We actually made forward and backward propagation in a single line of code.
<img src="training-grads.JPG">

### Notebook notes:
- There is new way of writing a loop that I didn't know. Instead of saying:
```python
[i for i in range(0,size(a))] # a is a list
```
We can simply say:
```python
[*range(size(a))]
```
- **Shuffling the examples for each epoch is known to reduce variance, making the models more general and overfit less.**





> Written with [StackEdit](https://stackedit.io/).