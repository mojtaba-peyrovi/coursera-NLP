

## Coursera NLP by deeplearning.ai

### Course 2: Natural Language Processing with Probabilistic Model:
 __Week1 -  Autocorrect and Minimum Edit Distance:__
What is autocorrect? An application that changes misspelled words into the correct ones. In this course we don't cover the words spelling correct but not being appropriate in the context. (e.g. Happy birthday my DEER friend, instead of DEAR friend.)

#### Four Steps of Auto Correction:
##### Step1: Identify the misspelled word: 
A misspelled word is a word which is not found in a dictionary. like this:
```python
if word not in vocab:
	misspelled = True
```
##### Step2: Find strings n edit-distance away: 
Find strings that are one, two, three, any number n edit distance away. Edit is an operation performed on a string to change it to another string. edit distance is the count of these operations. better definition is: n edit distance is how many operations needed from a string to be converted to another string. There are four edit types:
1- Insert: add a letter
2- Delete: remove a letter
3- Switch: swap two adjacent letters
4- Replace: change one letter to another
By having these steps and combining them, we find a list of possible strings that are n edits away.
##### Step3: Filter candidates:
Once we have the list of possible strings, we can filter them and keep the ones that are real and correctly spelled words.

##### Step4: Calculate word probabilities:
Finding the most likely words among the candidates. The probability of a words is calculated by dividing the word's frequency in the corpus to the total length of the corpus. For autocorrection we replace the misspelled word with the candidate with the highest probability.

#### The Lab:
- We added all the unique words to a list. When we want to add the frequency of each word, we can create a dictionary having a word and its count as each row of the dict. In order to do this, there is two ways:

First method:
```python
count_a = dict()
for w in words:
count_a[w] = counts_a.get(w,0) + 1
```
Second method:
```python
from collections import Counter
count_b = dict()
count_b = Counter(words)
```
**Numpy intersectld:** it finds the items existing in two numpy arrays:
```python
vocab = ['dean','deer','dear','fries','and','coke']
edits = list(deletes)

print('vocab : ', vocab)
print('edits : ', edits)

candidates=[]

### START CODE HERE ###
#candidates = ??  # hint: 'set.intersection'
import numpy as np
candidates = np.intersect1d(edits, vocab)
### END CODE HERE ###

print('candidate words : ', candidates)

>>> vocab :  ['dean', 'deer', 'dear', 'fries', 'and', 'coke']
edits :  ['earz', 'darz', 'derz', 'deaz', 'dear']
candidate words :  ['dear']
```
### Minimum Edit Distance:
This is a tool to find out how similar two words or strings are. Given two words or strings, minimum edit distance is the minimum number of operations needed to transform one string into another. use cases are: spelling correction, document similarity, and machine translation, DNA sequencing. 

For editing, we use three different operations:
1- Insert
2- Delete
3- Replace
for example for converting "play" to "stay" we need to replace "p" to "s" and "l" to "t". so the number of operations will be two. 
### Operation Cost:
For each type of operation we consider the cost of insert as 1, cost of delete 1, and cost of replace 2. like this:
{ "insert":1, "delete":1, "replace":2}

In finding the minimum edit distance, we are trying to minimize the cost, which is the sum of all operation costs.
<img src="simple_minimum_distance_calculation.JPG">

### Algorithm:
We create a matrix where rows will be the source string and columns will be the target string.
We need to fill out a distance matrix (D) such that the distance matrix D[2,3] is the distance the beginning of the source to character 2, and from the beginning to character 3 in the target string.  We show it in python like this:
```python
D[2,3] = source[:2] -> target[:3]
or
D[i,j] = source[:i] -> target[:j]
```
<img src="distance-matrix.JPG">

In order to fill the matrix with all transformation costs per character, we start from the top left corner, and calculate for the smallest number of characters (# which is zero) then we add 1, then 2, etc. to till the end of it. calculating the diagonal numbers has three ways:
1- delete then insert
2- insert then delete
3- delete and insert at the same time
we can calculate the cost for each method and use the minimum of them as the final cost. then we add one more character and keep doing the same calculations. 

<img src="cost-matrix-simple.JPG">

#### The formulaic approach:
after filling the first four cells on the top left corner, here is what we do for the rest of the cells:
1- **Fillout the top and left edges:** we can use this formula:
```python
D[i,j] = D[i-1,j] + del_cost  # for the left column
D[i,j] = D[i, j-1] + ins_cost  # for the top row 
```
2- **calculate the rest of the cells:** the calculation is like this:
calculate the following three costs:

- D[i-1,j] + del_cost
- D[i,j-1] + ins_cost
- D[i-1,j-1] + rep_cost  // if the characters are the same, the replacement cost will be zero

Then find the minimum of them.
<img src="minimum-edit-distance-formula.JPG">

When we input all numbers and add heatmap effect to it, we see that from the middle of the matrix to the end, we don't need any change. so, the numbers will repeat without any cost being added to them. (the reason is in this example both words end with "ay"

<img src="minimum-edit-dictance-heatmap.JPG">

This method is also called **Levennshtein.**

In order to program this, we use **Dynamic Programming** or **Recursive programming** which solves the problem for a small portion, then uses the result of it to calculate the next iterations.

### Assignment notes:
[THIS](https://norvig.com/spell-correct.html](https://norvig.com/spell-correct.html)) article has the whole code in Python.

The photo below is a good review of list comprehension in Python.

<img src="python-list-comprehension.png">

The image below shows how to write word splits in list comprehension.
<img src=" python-list-comprehension_word_splits.png">

__Week2: Part of Speech Tagging (POS tagging):__

Part of speech refers to the category of words or the lexical term(noun, verb, adjective, etc.)  in the language.
<img src="part-of-speech-example.JPG">

__Some applications of POS tagging:__
1- Named entities like Eifel tower and Paris in the sentence: Eifel tower is located in Paris.
2- Co-reference resolution: when we say Eifel tower is located in Paris. It is 324 meters. This method can tell "it" refers to the Eifel.
3- Speech recognition.



#### The notebook notes:
This [link](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) has the list of all tag abbreviations.

__defaultdict:__ It is a special type of dictionary that returns 0 if the key doesn't exist. 
__string library:__ string.punctuation returns punctuation marks. [(docs)](https://docs.python.org/3/library/string.html)

#### Markov Chains:
When we have a sentence like "Why not learn English?" We want to know what type of word (tag) would be suitable right after a verb. It's more likely to be a noun. So we can say:
>  The likelihood  of the next word's POS in a sentence, tends to depend of the POS of the previous word.

The image below shows the visual representation of the likelihood of the word coming after another word:
<img src="markov-chain-visual-represenetation.JPG">

**What are Markov Chains?** 
A stochastic model that describes a sequence of possible events. For predicting the probability of an event we need to know the state of the previous event. (Stochastic means random)

**Directed Graphs:** In computer science, a graph is a kind of data structure that is visually represented as a set of nodes or circles connected by lines. When the connector lines look like arrows and have directions, we call them directed graphs. 
- The circles represent the states of our model. 
- The connectors are also called __Edges.__
- A state refers to a certain condition of the present moment.
<img src="states-in-graph.JPG">

Now we want to see how POS can be shown as a states. if we show nouns by NN, verbs by VB, and others by O, we can make a graph like this: 
<img src="transition-probabilities.jpg">
The edges of the photo above show the weight probability of each transition from one state to another.
__Markov Property:__ It says the probability of the next event, only depends on the current event. It helps to keep the model simple. It doesn't need the information of all previous states, only the current one.
Having the graph, we can create a table (matrix) that saves the values of the transition probabilities, called __Transition Matrix.__
<img src="transition-matrix.JPG">

The transition matrix is an NxN matrix, where N is the number of states in the graph.
- In the transition matrix, the sum of each row will be equal to 1.
__Initial Probabilities:__ Because the model always looks at the previous word to find the transition probability, for the first word of the sentence, there is no previous word. For having that issue fixed, we add an __Initial State__ and show it like this:
<img src="initial_probabilities.JPG">
So the dimension is not NxN anymore. It is (N+1)x(N)
So far we learned that we can show all of the states as a vector named Q, and we can show the transition matrix as the matrix A.
<img src="markov-chain_summary.JPG">

__Hidden Markov Models:__ It refers to the states that are hidden or not directly observable. For example, if we say Jump, Run, Fly, etc. as humans we know they are verbs, but the machine just sees the text, not the type of them. These three words are called observable to the machine. 
<img src="observable-words.JPG">
__Emission Probabilities:__ They describe the the transition from the hidden states of  the Markov Model to the observables or the words of the corpus.
<img src="emission-probabilities.JPG">
We can also show a tabular representation of the emission probabilities. In this matrix, B, each row is associated with a hidden state, and a column belongs to each of the observables.
<img src="emission-probability-matrix.JPG">

For matrix B also, the summation of each row is 1.

This is the summary:
<img src="summary-of-hidden-markov-models.JPG">
__How to calculate the transition probabilities?__
To calculate the probabilities, we don't care about the words themselves in the corpus, we just need the POS tags. For example, if we want to see the probability of a word followed by a noun:
- We calculate the number of occurrences of the combination. 
- Then calculate the number of times the first tag (verb) occurred. 
- The final probability is the first value divided by the second value. (in the following photo: 2/3)
<img src="calculate-probabilities-visually.JPG">

Here is how we formulate it:
- Define the C function that is the count of each tag pairs.
- The we calculate the P(t) function given the previous tag.
<img src="transition-probability-formula.JPG">

__Corpus Preparation:__ 
1- First we add an "S" tag to the beginning of each sentence.
2- Then we lowercase all characters.

__IMPORTANT:__ In the transition matrix, each row shows the current state, and each column represents the next state.
Here is how to populate the transition matrix:
<img src="populating-transition-matrix.JPG">

Now, after inputting all the count values, we should divide each count by the sum of each row.
<img src="populating-transition-matrix-2.JPG">
To avoid the problem of division by zero, we should __Smooth__ the values like this:
<img src="smoothing-tranistion-probabilities.JPG">

Remember, with smoothing, the total of each row is not exactly 1 anymore.

- In real world examples, we don't apply smoothing to the first row of the matrix, because by doing this we say we don't mind a sentence starting with any POS including punctuation letters.

__How to calculate the emission probabilites?__
For calculating it, we need to do this (because it is the probability of each observable word vs the its tag):
1- first we count each word
2- then count the it's tag in the whole corpus
3- divide first number by the second one. (in the following photo it is 2/3)
<img src="emission-probabilities-calculation.JPG">

Finally in the image below we see how to populate the emission matrix. (V is the size of the vocabulary)
<img src="populating-emission-matrix.JPG">

#### notebook notes:
- sorted() is python a function that can sort any iterable like list in an ascending order.
- np.sum() you can sum up the rows, or columns. parameters are (axis=1, keepdims=True) it will sum up ROWS and result in the same dimensions as the original matrix.
The photo below, shows the difference between keepdims false and true:
<img src="keepdims_false_vs_true.JPG">
- If we want to normalize the matrix we can do this way:
<img src="normalizing_matrix_numpy.JPG">
Notice that first we calculated the sum of each row, then divided each row by the sum of the row. after the normalization, the sum of each row must be 1.
- np.diag() will save the diagonal values of the matrix into a numpy array. notice that the shape of it would be (3,). we can reshape it (because we need to have it to be the same shape as the row sum (3,1) we can reshape like this:
```python
d = np.diag(transition_matrix)
d.shape
> (3,)
d = np.reshape(d, (3,1))
d.shape
>(3,1)
```
- __np.fill_diagonal(transition_matrix, d):__ it will fill the diagonals with the new value of d
- __np.vectorize():__ It performs the vectorized operation (elementwise) e.g. d = d + np.vectorize(math.log)(rows_sum)

### Vitrebi Algorithm:
We learned how to find the most probable WORD given a previous word, using the transition and emission matrices. Vitrebi algorithm will help finding the most probable SENTENCE, given the previous sentence.
The process starts from the initial state, and moves to the first word (I love to learn) that cannot be emitted from any other state. (word I) for calculating its probability we multiply transition value(0.3), by emission value(0.5).
<img src="vitrebi-initial-move.JPG">
The next step is to find the probability for the word love. There are two ways for love to have been seen (one from VB, and one from NN) we calculate both probabilities and pick the higher value (VB in this example)
<img src="vitrebi-word-love.JPG">
The next step is, from the second word to the third one (from Love to To) through O state we can get there. (0.2 x 0.4 = 0.08)
and finally from To to the last word Learn, which is 0.5 x 0.2 = 0.1. 
The total probability is the product of all probabilities. 
<img src="vetrabi-total-probability.JPG">
Viterbi algo, calculates the probabilities for several sentences at the same time using matrix representation of each sequence.

### Steps of Vitrebi:
__1) Initialization:__
We need to make two auxiliary matrices C, D.
Here is how we calculate matrix C:
<img src="vitrebi-matrix-c.JPG">
and here is how we initialize matrix D:
<img src="vetrebi-matrix-d-initialization.JPG">
__2) Forward Pass:__ 
Here is how we do forward pass:
<img src="vitrebi-forward-pass.JPG">
__3) Backward Pass:__
Here is the backward pass:
<img src="vitrebi-backward-pass.JPG">
 

#### Assignment Notebook
- A good way of showing the process of a loop, is to print the iteration number on specific iterations. for example below loops through words and on each 50,000 rows, it prints the row no. 
```python
for word_tag in training_corpus:
        
        # Increment the word_tag count
        i += 1
        
        # Every 50,000 words, print the word count
        if i % 50000 == 0:
            print(f"word count = {i}")
   ```

__N-Grams Model:__
#### Text Corpus: 
A large database of text documents, such as all pages on Wikipedia, all books from one author, or all tweets from one account.
#### Language Model:
A language model is a tool that calculates the probabilities of sentences. We should think of a sentence as a sequence of words. It can also calculate the probability of a word given a history of the previous words. 
N-Grams model can be used to autocomplete a sentence.
<img src="n-grams-flow.JPG">
Language models can be used in different cases:
<img src="n-grams-other-applications.JPG">

### What is an N-Gram?
N-Gram is a sequence of N words. Given a sentence, we can generate Unigrams, Bigrams, and Trigrams which are combinations of one, two, three words next to each other in the same order as the sentence.
<img src="n-grams-1-2-3.JPG">
We can also have sequences with more than three words. Here is how we show them:
<img src="sequence-notation.JPG">
Here is how we calculate the probability of each unigram:
<img src="unigram-probability.JPG">
For calculating the bigrams probability, we can say the probability of  a word to be "am" given the previous word as "I" is the count of times where "I am" repeated in the corpus, divided by count of times where "I" occurred.
<img src="bigrams-probability.JPG">
Similarly, we can calculate the trigrams' probability:
<img src="trigrams-probability.JPG">

### Sequence Probabilities:
What is the probability of a sequence? For doing this, we need to calculate P(A, B) which is the probability of word A and B. which will be the probability of A, times the probability of B give A, like this:
<img src="chain-rule.JPG">
If we extrapolate it to more than two words, we get this:
<img src="sequence-probability.JPG">
Here is an example:
<img src="sequence-probability-example.JPG">

The problem with this approach is, most of the time, the exact sequence doesn't exist in the training corpus. The longer the sentence, the least the likelihood of a sentence to appear in the corpus. For fixing this we should apply **Approximation** which would be using ONLY the probability of the previous word, rather than the whole sequence behind the given word.	
<img src="sequence-probability-approximation.JPG">

Or we can say we just consider the product of the probability of the bigrams.

#### Another Issue:
If we are considering the bigrams only, what happens to the beginning and ending of a sentence?
In order to solve it we simply ad a token to the beginning of the sentence, so that the first word has a predecessor.
<img src="start-of-sentence-token.JPG">
Similarly, for Trigrams, the first two word won't get enough predecessors to have any probability. for this case we add the token twice to the beginning of the sentence.
<img src="starting-token-n-grams.JPG">
For the end of the word, similarly we add </s> token. For </s> we don't need to add more than once for N-grams for any N value.

### Steps:
1- Count matrix:.JPG">

<img src="count-matrix.JPG">

2- probability matrix:
<img src="probability-matrix,JPG">

3- Language model:
<img src="language-model.JPG">

### Notebook Notes:
Here is how we append an element to the end of a tuple:
```python
sample = ('moji','wins')
word = 'always'
new_sample = sample + (word,)
print(new_sample)
>> ('moji','wins','always')
```
For large datasets the split between train/validation/test should be like 98%, 1%, 1%

In NLP we can split the dataset it two ways:
1- picking 80% of the sentences in the corpus for testing, 10% for validation, and 10% for testing.
2- picking random short sequences from the corpus.
<img src="nlp-split-methods.JPG">

**Perplexity Metric:** 
Perplexity is a commonly used metric in language modeling. As perplexed means confused, we can use perplexity model to measure the complexity of a text. This metric helps us figure out whether a sentence is written by a human or by a machine picking random words. A text written by a human usually has a **Lower** perplexity score.

In the other hand we can say:
> Perplexity is the inverse probability of a test set, normalized by the number of words in the test set. So. the better a language model estimates the probability of the test set, the lower the perplexities are going to be.

Perplexity is closely related to entropy which measures uncertainty.
<img src="perplexity.JPG">

Good language models have the perplexity of 20-60 not more than 60. For English language it can be even lower.
<img src="perplexity-example.JPG">

The formula for perplexity for bigrams, is the product of the bigram probabilities of all sentences to the power of minus one over the count of all words in the corpus (m)
If all the sentences in the test set are concatenated, the formula will be the product of the bigram probabilities for the entire test set to the power of minus one over the count of words.
<img src="perplexity-formula.JPG">
**Log Perplexity:** for the ease of computation, some papers offer using log perplexity instead of the m'th root of the probability. Here is the formula:
<img src="log-perplexity.JPG">
Good language models have the log perplexity between 4.3-5.9
The following example shows how using Unigram, Bigram, and Trigrams return different perplexities. The Trigram model returned a better perplexity and the sentence generated by the model makes more sense:
<img src="perplexity-comparison.JPG">
  
 ### Out of Vocabulary Words:
 When the model has only a fixed amount of words, it is called **Closed Vocabulary.** For example chatbots only have a limited set of words. But most of the times your model has to deal with **Open Vocabulary** cases which means the words your model hasn't seen before.
 The unknown words are also called **Out Of Vocabulary (OOV)** words. 
 One way is to tag them with a tag "UNK". (replace them with the word "UNK")
 Now, we can count the probabilities with the word UNK as another word. 
 For example, if we say in the corpus, if a word repeated more than or equal to twice, then it will be added to the vocabulary list. So, any word which happened only once (like people's names) will be tagged as unknown.
 <img src="unknown-word-example.JPG">
 We need to be careful not to use so many unknown words, because the model will generate sequences with so many unknown words with high probabilities.
 Also, we can use perplexity as a metric by comparing Language Models with the same V.
 ### Smoothing:
 When we deal with a limited corpus, sometimes the probability of some words may be skewed. We can fix it with Smoothing method. We use this technique when we have some known words in the vocabulary but still we miss them in the corpus. For example, we may have "John" and "eats" in the corpus but the bigram "John eats" is missing. so the count and probability of this bigram will be zero.
**Add-one Smoothing(Laplacian Smoothing):** We add 1 to the nominator and each occurrence of counts in the denominator summation. This method works well with large enough corpus size, to outweigh the 1 value, otherwise the probability of missing words will be too high.
<img src="smoothing.JPG">
**Backoff Technique:** For each word, if N-gram is missing, we use N-1 gram, and if it is also missing, we use N-2, etc.
<img src="backoff.JPG">

### Notebook Notes:
- Here is how to count the most M common words in a corpus:
```python
from collections import Counter
# the target size of the vocabulary
M = 3

# pre-calculated word counts
# Counter could be used to build this dictionary from the source corpus
word_counts = {'happy': 5, 'because': 3, 'i': 2, 'am': 2, 'learning': 3, '.': 1}

vocabulary = Counter(word_counts).most_common(M)	
```
 __Week4 -  Word embeddings with neural networks:__
This week is about training our own word vectors, or embeddings. The purpose for making word vectors, is to convert words to numerical values, so that we can do mathematics on the words.

**One-Hot Vector:** For each word in the corpus, we create a column of zeros and only the row in which the word is associated with, the value will be 1.
<img src="one-hot-vector.JPG">
The problem with these vectors, is they are so huge and sparse, and take so much processing memory to be digested. Also, this representation doesn't carry the meaning of the word.

**How word embedding works?** 
First, if we have a 1D axis for representing whether each word is negative, or positive, then we can distribute our words along this axis. like this:
<img src="1d-embedding.JPG">
Then we can find another concept and add another axis to represent the second concept, and then we can distribute the words in this 2d surface. 
<img src="2d-embedding.JPG">
By having these two dimensions, we can represent each words as a vector of length two.

**How to Create Word Embeddings?** 
To create a word embedding, we need two things:
1- َ A text corpus: The corpus must be consisting of words about a specific topic. The context of a word refers to other words or a combination of words said to occur around a particular word. The corpus can be related to a specific industry which means for marketing industry, using all words in our contents would be relevant.
2- An embedding method: we use machine learning to train the embeddings.
In order to feed the corpus into the ML model, we need to make a numerical representation of the corpus using the one-hot vector or simple unique integer assignment.

**Word Embedding Methods:**
1- Word2vec: It uses a shallow neural network to generate word embeddings. It uses the following methods;
 - continuous bag of words (CBOW)
 - continuous skpi-gram / skip-gram with negative sampling (SGNS)
2- Global Vectors by Stanford
3- FastText by Facebook: This model supports out of vocabulary words.

More advanced based on deep Learning, contextual embeddings. In the previous methods, a word always has the same embedding, but in deep learning techniques, a word will have different embedding depending on the context.
1- BERT by Google
2- Elmo by Allen Institute for AI
3- GPT-2 by OpenAI

Some useful links here:
[Thai BERT](https://www.slideshare.net/kobkrit/thai-text-processing-by-transfer-learning-using-transformer-bert)
[Another Thai BERT LIB](https://github.com/ThAIKeras/bert)

[Also BERT for THAI](https://nlp.h-its.org/bpemb/th/)


### CBOW (Continuous Bag Of Words) model:
As mentioned, in order to transform the data from the corpus into the embedding ML model, we should handle missing words. It can be predicted based on the surrounding words.
<img src="cbow.JPG">
Here is how it works. If two unique words are regularly surrounded by a similar set of words in various sentences, then those two words tend to be close in their meanings. or we can say **They are related semantically.**
<img src="cbow-concept.JPG">
The training example can be created per word, by calling the main word **"Center Word"**, then we define the word before and after the word as **"Context Words"**, and the total context and center words is called **"Window".** 
<img src="cbow-training-example.JPG">
To train the model, we need a set of examples. Similar to CNN, the window will move word by word and the center word will be learned based on the context.
<img src="cbow-training-process.JPG">
This training will be done by neural network. but before that we need to clean the data and tokenize them. Here are the steps:

Steps Of Cleaning and Tokenization:
---
 - Letter case: we should make sure the model is case insensitive, by converting all letters to either all lowercase or all uppercase.
 - Punctuation: We can replace them all with a specific character such as full stop.
 - Numbers: If the numbers don't mean much in the text, we can just drop them all, if not, we can replace them with a number token.
 - Special Symbols: its usually safer to drop them.
 - Special Words: emojis, and hashtags. we can for example treat an emoji as a new word.
 <img src="cleaning-tokenization.JPG">
 Here is an exmaple of how to deal with emojis:
 <img src="tokenization-example.JPG">
 And below is the python code to handle it:
 First: import libraries:
 <img src="tokenization-python-emoji-libraries.JPG">
 Second: replace all punctuation will . using Regex:
 <img src="tokenization-python-emoji-regex.JPG">
 Third: Tokenization
 <img src="tokenize.JPG">
 Fourth: convert the tokens including emojis into lowercase:
 <img src="lowercase.JPG">
 This corpus now is ready to be used in model training.
 ### Create the sliding window in Python:
 Here is the function which takes two parameters:
 - words: which is an array of tokens (words)
 - C: the **Context Half Size** which means the number of the words we get from each side of the center word.
 <img src="get-window-function.JPG">
 In the function we used **yield** instead of **return** because return will immediately exit the function and return the values, but yield pauses the function and returns the values and will continue running if more values are needed (The object created by yield is generator object)
 Here is the outcome of the function:
 <img src="get-windows-outcome.JPG">
 ### Transforming center words into vectors:
 First, we create a Vocabulary array from corpus which is a set of unique words in the corpus.
 Second: we convert each word as a one-hot vector. and we can input this matrix into the model:
 <img src="center-words-into-vectors.JPG">
 ### Transforming context words into vectors:
 For this, we just calculate the average of hot-vectors for each context.
 <img src="context-to-vector.JPG">
  
Architecture of CBOW Model:
-
We make a neural network with one hidden layer and the Vx1 matrix of all words (formed as vectors) as the input layer, and the output layer is a Vx1 vector that represents the center word.
<img src="cbow-architecture.JPG">
The size of the hidden layer should be the size of the word embedding.

The activation function for the first layer should be Relu, and the hidden layer Softmax.
<img src="cbow-matrix-sizes-explained.JPG">
**Batch Processing:** To have a faster processing, we can feed samples in groups rather than one by one. The batch size is the hyperparameter that we will define as "m" for the algorithm. The input values will be batched m by m as separate matrices. In order to have the next calculation steps, we need to adjust the sizing of b vector to be also in batches of size m. The process of converting a vector to a set of m vectors is called **Broadcasting.** It gets done automatically in Numpy when we convert a matrix to a column vector with the same number of rows.
<img src="cbow-matrix-sizes-batching-explained.JPG">
At the end we will have a mxV matrix as Y hat, and it is the same size as the input matrix X and each X value will get a predicted value as its counterpart Y hat.
<img src="cbow-matrix-sizes-batching-explained-2.JPG">

- About Softmax: It gets the values of the matrix and converts them to values between 0-1 that can be interpreted as probabilities.
<img src="softmax.JPG">
Here is an example of softmax calculation:
<img src="softmax-example.JPG">
**Cost Function:** For classification models we mostly use **cross-entroy loss** which comes most of the time hands in hands with Softmax activation function.
<img src="cross-entroy.JPG">

**How to extract the word embedding from a trained CBOW model?**
 We have three options:
 1- Each column of the matrix W1 corresponds to one word of the input matrix X, in the same order. 
<img src="word-embedding-extraction-op1.JPG">
2- Each row in W2 corresponds to a word in matrix X.
<img src="word-embedding-extraction-op2.JPG">
3- Take the average of values in option1,2
<img src="word-embedding-extraction-op3.JPG">

**Evaluating the Model:**
we can do it in two ways:
1- Intrinsic: evaluates how well the word embedding capture the semantic and syntactic relationship between the words. semantic refers to the meaning of of the words, and syntactic refers to the grammar. 
One way of doing intrisic check, is by **Analogies** which means giving examples and asking the system to bring similar words (tokyo to Japan is similar to Paris to? .....) .This approach has a disadvantage that one scenario can have more than one possible answer.
<img src="intrinsic-analogies.JPG">
The other way is to make a clustering algorithm
<img src="intrinsic-clustering.JPG">
2- Extrinsic: Evaluates the usefulness of a word embedding from an actual external example. Extrinsic methods are the ultimate test to see if the word embedding the the classification are actually useful. 