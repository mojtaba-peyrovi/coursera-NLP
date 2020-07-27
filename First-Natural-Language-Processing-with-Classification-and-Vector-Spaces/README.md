

## Coursera NLP by deeplearning.ai

### Course 1: Classification and Vector Spaces:
- Week1 - Sentiment analysis with logistic regression:
In order to create a sentiment analysis on a tweet using logistic regression, we need to follow the steps below:
1- extract features from the text
2- train the logistic regression classifier based on the features X
3- make the predictions (classify)

 Imagine we have a list of tweets: [tweet1, tweet2, tweet3, ...] , which is called **Corpus** . Now, we need to make a list of all UNIQUE words from all tweets. we call the new list **Vocabulary list*** and we show it by letter **V**.
 Then for each tweet, we can do **Feature Extraction.** to do so, we need to see what words in the vocabulary list appears in the tweet. For each word in V, if the value exist in the tweet, we assign value 1 to it, and if doesn't exist, we assign values of 0. Of course, the matrix containing the features has so many zeros and will be considered a **Sparse** representation.

#### Problems with sparse representation:
We have a large list of 0-1 which is as big as the length of the V vector.</br>

<img src="https://render.githubusercontent.com/render/math?math=[\theta _{1}, \theta _{2}, \theta _{3}, ... , \theta _{n}] , n = \left | V \right |"></br>

it will cause:
1- Large training time
2- Large prediction time
In order to take care of this issue, we will count how many times each word is repeated in a positive class, and how many times repeated in negative class. Using both of these counts, we can use the features to train the classifier. 

Because we are using the training data, we know which tweet is negative and which one is positive. now imagine we have two positive tweets such as :

- I am happy becase I am learning NLP
- I am happy

The whole Vocabulary list has 8 unique words such as : [I,am,happy,because,learning,NLP,sad,not]
Having these positive class tweets and the V list, we can calculate how many times each words has appeared in the positive tweets. PosFreq(1) = [3,3,2,1,1,1,0,0] for example first 3 means the word I, has been repeated three times in positive tweets. and the last 0 means the word not, has not been used in any positive tweet.
We do the same thing for negative classes. and the end result will be a dictionary mapping from word,class to frequency.
 We also call it **Frequency Dictioinary.**

Now for each tweet, we can extract the features like this:


<img src="https://render.githubusercontent.com/render/math?math=X_{m} = [1, \sum_{w}^{}freqs(w,1), \sum_{w}freqs(w,0)]">

it means for tweet m, we have three features, 1 is the bias, sum of the frequency of the words repeated in posFreq, and sum of the frequency of the words repeated in negFreq.
Example: for the tweet: I am sad, I am not learning NLP. given the  posFreq will be 3+3+1+1+0+0  (check the video) and for negFreq we get 11. so for this tweet, the features will be [1,8,11]

#### Preprocessing:
For preprocessing the text, we should do two steps. 

- **Stop words and Punctionation:** for this step we should first find the words and punctuation marks that don't add significant meaning to the sentence. We should compare the sentence with two lists:
1- Stop word
2- Punctuations
and all of the stopwords and punctuations will be removed from the sentence. Most of the time hyperlinks and handles also don't add any meaning to the sentence.
For example, this tweet: 
> @AndrewYNg @YMourri tuning GREAT AI model https://deeplearning.ai

After preprocssing will look like this:
> tuning GREAT AI model

The second thing you have to implement is **stemming** which means transforming each word in its stem. stem is the set of characters that are used to construct the word and its derivatives. for example for the word "tuning" the stem is "tun". because adding e would make the word tune, adding ing makes it tuning, etc.
We do this technique to reduce the vocabulary size.

The third thing we need to do, is to **lowercase** all the words. For example we convert GREAT to great. 
After doing all these steps, the final sentence would be like this:
> [tun, great, ai, model]
> 
### Tokenize the string[](https://iyninneg.coursera-apps.org/notebooks/NLP_C1_W1_lecture_nb_01.ipynb#Tokenize-the-string)

To tokenize means to split the strings into individual words without blanks or tabs. In this same step, we will also convert each word in the string to lower case. The  [tokenize](https://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.casual)  module from NLTK allows us to do these easily:
 
### Conclusion:
Let's put all we learned together. We have a set of m tweets. We will have to implement the preprocessing for each one of the tweets and convert each of them to a list of words. Then, we extract features for each tweet using a **frequency dictionary mapping.**  e.g. "I am happy Because I am learning NLP @Deeplearning" will be converted to [happy, learn, nlp] and the frequency matrix will be [1,40,20]
At the end we will have a matrix of m rows and three columns:
$$
\begin{bmatrix}
1 & X_{1}^{(1)} & X_{2}^{(1)} \\ 
1 & X_{1}^{(2)} & X_{2}^{(2)} \\ 
. & . & .\\ 
 .& . & .\\ 
. &.  & . \\
1 & X_{1}^{(m)} & X_{2}^{(m)}
\end{bmatrix}
$$
Where each row will be associated to a tweet.

### Testing the model:
We imagine we ran the logistic regression model already and calculated parameters (thetas)
Now for each tweet and its theta we have the prediction like this:</br>

<img src="https://render.githubusercontent.com/render/math?math=pred = h(X_{val}, \theta) \geq 0.5" width="200"></br>

because we are using Sigmoid function. For example, if we have h(x) values for each tweet as below: </br>

<img src="https://render.githubusercontent.com/render/math?math=\begin{bmatrix}0.3\\ 0.5\\ 0.7\\ .\\.\\h_{m}\end{bmatrix}"></br>

We compare them one by one against 0.5, and if its bigger we have 1, if smaller 0:

<img src="https://render.githubusercontent.com/render/math?math=\begin{bmatrix}0\\1\\1\\1\\.\\.\\pred_{m}\end{bmatrix}">

Then we compare the prediction values (the above matrix) vs the real values, and we have a new matrix with the comparisons. If the values match, we return 1, if not we return 0. 

<img src="https://render.githubusercontent.com/render/math?math=\sum_{i=1}^{m}\frac{(pred^{(i)} == y_{val}^{(i)})}{m}" width="200">

the sigma above, sums up all zero and ones (showing the total times the prediction was correct), and divides them by total number of tweets, and returns the accuracy percentage.

- Week2 - Naive Bayes:
Imagine we have a corpus of positive and negative tweets. If A is labeled as a positive tweet, the probability of the tweet being positive is:

<img src="https://render.githubusercontent.com/render/math?math=P(A) = P(Positive) = N_{pos} / N" width="300">

means the count of positive labelled tweets, divide by the number of all tweets in the corpus. 

Now, if there is a tweet B that contains the word "Happy" but tagged in both positive and negative tweets, like the photo word-happy-case.jpg, the probability of a word to contain the word "happy" and also being positive is the area of intersection divided by the area of the whole corpus. For example if there is 20 tweets, and three of them contain the word and also labelled positive, the probability will be 3/20 = 0.15

If we just consider the tweets that contain the word "happy" and as we see, three of them are labelled positive, and one is labelled negative, as seen in photo bayes-rules.jpg, then the probability of the tweet being positive and containing the word will be 3/4 = 0.75	
This simply means if we have a tweet, based on this training dataset, the likelihood of a tweet to be positive, if it contains the word "Happy", is 75%.

We can also calculate the probability of a tweet to be contain the word "Happy" if it is positive. photo: bayes-rules-2.jpg
In this case we have 13 tweets in total, and three of them have the word "Happy" so the probability will be 3/13 = 0.231

If we combine these two probabilities we can get the final definition of Bayes Rules seen at photo: bayes-rules-3.jpg

We can make is simpler by algebra and say Bayes Rule as:
P(X|Y) = P(Y|X) * (P(X)/P(Y))
Simply, **Bayes' Rule** is based on mathematical formulation of conditional probabilities. 
Using this formula, we can calculate the probability of X if Y, if we have the probability of Y if X, and the ones of X and Y separately.

It is called Naive because it assumes that the features we used in logistic regression are independent and I think it means it doesn't care for the features just calculates the probabilities. 

To implement it we need to to the following steps:
1- count the unique words in all the corpus
2- count the repetition of each word in both positive and negative tweets.
3- sum up all positive counts and negative counts.
4- then we create a new table called the table of conditional probabilities, in which every word in the vocabulary list, will have a P(word|Positive) and P(word|negative)  as seen in photo bayes-steps-1.jpg. In this table, the sum of positive values will be 1, so will the negative column.
5- In this table, the words with almost equal probability, don't add anything to the sentiment. What are really important to the model, are the words with big difference.
6- Now we should calculate the product of positive over negative of each word and if the final result is bigger than one we say its moer probable for the sentence to be positive than negative. photo: bayes-steps-2.jpg (This value is called **Naive Bayes inference condition rule for binary classification**)

#### One more issue to solve with Naive Bayes:
No probability in the original formula should be zero. In order to avoid zeros, we will slightly change the main formula and it is called **Laplacian Smoothing.** The formula is in the photo laplacian-smoothing.jpg
Now, back to the same example, instead of original naive bayes, we can calculate the probabilities using Laplacian smoothing. photo laplacian-smoothing-implementation.jpg
After laplacian smoothing, no probability will be zero. 

### Log Likelihood: 
#### Ratio of Probability:
IN photo bayes_step_2.jpg, we calculated the Niave bayes inference condition rule for binary classification which is called the **Log Likelihood**. Now, we have a table with a percentage of frequency for each word and since we use Laplacian smoothing, none of the frequency values is zero. 
Now we can calculate the ratio of probability like this: </br>
<img src="https://render.githubusercontent.com/render/math?math=\ratio(W_{i}) = \frac{P(W_{i}|Pos)}{P(W_{i}|Neg)}" width="250" class="center"></br>

This ration will be more than 1 for positive, less than 1 for negative, and equal to one will show neutral words. And in order to avoid division by zero, we can write it like this:
$$
\ratio P(W_{i}) = \frac{P(W_{i}|1) + 1}{P(W_{i}|0) + 1}
$$
The final formula for the naive bayes, is the multiplication of two fractions, one is called **Prior Ratio** and the other one is **Log Likelihood.** photo: final_naive_bayes_formula.jpg

- when we run this algo, we may face the risk of underflow which is the product of numbers less than zero for several time have a very tiny result that computers cannot save it. To avoid this, we use logarithm.
photo: fix_underflow_issue.jpg

- The log of Log Likelihood is called **Lambda** and the photo: calculate_lambda.jpg shows how to calculate it.
- photo log_likelihood_sample_calculation.jpg shows an example of calculating log likelihood.
**IMPORTANT:** The prior ratio for balanced datasets(where the number of pos and neg words is equal) will be zero. but for unbalanced datasets, this value will be important.

### How to test the model?
We simply expose the model to a new tweet which has not been used for training the model. We have lambda values, and the prior ratio, so we can calculate the log likelihood. If a word does not exist in the frequency dictionary, we consider them as neutral.
The photo naive_bayes_testing.jpg has the summary of the steps to calculate the sentiment of the testing tweet.

- Week3 - Vector Space Models:

Vector Space Models help us recognize if two sentences are similar in meaning even if they don't look similar and don't share the same words. It can also capture dependencies between words. 
Vector models are used in feature extraction to answer the questions of who? why? where? when?
This technique captures the context around each word in the text, therefore it captures the relative meaning.
### Co-Occurrence: 
**1- Word by Word approach:**
The co-occurrence of two different words is the number of times they appear together in the corpus within a specific distance K. 
(check this photo: word-by-word-co-occurrence.jpg)
It shows that if we have k=2, the word "data" is related to the word "simple" with number 2 because two times in the corpus simple has been within the distance of 2 words from "data". For the example in the photo, the vector representation of the word "data" is [2, 1, 1, 0]
The vector representation has n entries which can be between one, and the size of the vocabulary. 

**2- Word by document approach:**
For this technique, we count the number of times each word is repeated in docs in the corpus with specific tags. this is the photo: 
(word-by-doc-co-occurrence.jpg)

If we make the co-occurrence matrix for two sample words of "data" and "film" then we can create a vector space with two coordinates, one for each and we can represent the 	matrix values like we see in the image: vector-space-word-to-doc.jpg
The image shows that the ML, and Economy documents are much more similar. but if the dimensions are higher it's not this easy to capture this. Later in this course we learn how to calculate the distance and angle between each vector.

### Euclidean Distance:
The Euclidean distance is the length of the line, connecting the end of two vectors.  photo: eucledean_distance.jpg
For  more than two dimensions, we do the same thing which is taking the square root of the sum of squares. (eucledean_distance_n_dimension.jpg)
In algebra it is called the Norm.
#### Calculating the norm in Py:
very simple:
```python
v = np.array([1,2,3])
w = no.array([4,5,6])
d = np.linalg.norm(v-w)
```
The shorter the d (euclidean distance or the norm) is, the more similar the documents are.

### Cosine Similarity:
Sometimes the euclidean distance doesnt work because of the number of docs are not the same and it makes a problem. But cosine similarity will calculate the cosine of the angle between two vectors. see photo: euclidean-vs-cosine-similarity.jpg

Cosine similarity is not biased by the size of the vector. but Euclidean is.
Here is how we calculate the cosine similarity:
photo: cosine-similarity-calculation.jpg
 > The closer the cosine value between two vectors is to 1, the more similar the vectors are. And the closer it is to zero, the less similar the vectors are.

### Principle Component Analysis (CPA)
Sometimes we are interested to reduce the dimensions of the vector. For doing this, we should find the uncorrelated dimensions to the data, and remove them. 

#### Eigenvector: 
Uncorrelated features to your data,.
#### Eigenvalue:
The amount of information retained by each feature.

- There is another method in Sklearn called t-SNE to reduce the dimensions.

### Idea1 for HL:
We have the word embedding for Thai words. we can calculate the similarity between keywords we have in google for the specific industry, then make recommendation for thai keywords with the most similar meanings to the specific content.


[link]([https://github.com/mmihaltz/word2vec-GoogleNews-vectors](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)) to Google News word embedding.

Another link here:
[https://code.google.com/archive/p/word2vec/](https://code.google.com/archive/p/word2vec/)

Link to the dataset of 100 billion words by Google News
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing


- Week4 - Transforming Word Vectors:

#### Transforming Vectors: 
Here is how we transform a vector to another one:
```python
R = np.array([[2,0],[0,-2]])
x = np.array([[1,1]])
transformed = np.dot(x, R)
>> Result: x is (1x2) and R is (2x2) ==> the result will be (1x2)
```
The transformation matrix is called **"Transformation Matrix".**
A good example of using it is translating a vector for an English word, into a French vector to translate the word from English to French.

We can create a subset of translations between the two languages, then using this transformation matrix, we can find the translation for any word.

photo: transformation_matrix_improvement.jpg shows how to start with a random R matrix, then in a loop try to minimize the loss function but comparing the translation with the real values.  (Transcript F means calculating the norm of the matrix, which is also called ** Frobenius Norm.**)  - photo: frobenius-norm.jpg

Here is how we calculate it in Python:
```python 
A = np.array([[2,2], [2,2]])
A_squared = np.square(A)
A_forbenius = np.sqrt(np.sum(A_squared))
```
By calculating the derivative of the loss function (frobenius norm) we calculate the gradient, which we are aiming to minimize it.
photo: gradient-formula.jpg

### Rotation Matrix:
There is a formula to transform the vector using the transformation matrix in counterclockwise direction having the degree theta. Here is the code:
```python
angle = 100 * (np.pi / 180) #convert degrees to radians

Ro = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle), np.cos(angle)]])

x2 = np.array([2, 2]).reshape(1, -1) # make it a row vector
y2 = np.dot(x2, Ro)

print('Rotation matrix')
print(Ro)
print('\nRotated vector')
print(y2)

print('\n x2 norm', np.linalg.norm(x2))
print('\n y2 norm', np.linalg.norm(y2))
print('\n Rotation matrix norm', np.linalg.norm(Ro))
```
Using this function, we calculated the rotation matrix.

The norm of any R^2 rotation matrix is always 2⎯⎯√=1.414221

- What we need to remember, is using R matrix to translate the English word's vector to the counterpart French word vector, the transformed matrix doesn't have to be necessarily identical to a word in the French corpus. We need to use K-nearest neighbors to find the closest meanings.

### Hash Tables:
In order not to search for all the words list to find the appropriate or closest translation, we can find categories to search within them rather than comparing the sample with the full list. 
For doing this we use hash functions to bucketize the values. 
#### Locality Sensitive Hashing:
We can define has functions which are able to categorize the values which are closer to each other in one group.

In order to do this, we define multiple planes, (dividers) and check if the sample is above each plane or below, and based on that we calculate the total hash. 
check this photo: hash-table-from-multiple-plane.jpg
Here is the code to find the hash value given the planes and a vector:
```python
def hash_multiple_plane(p_l, v):
    hash_value = 0
    for i,p in enumerate(p_l):
        sign = side_of_plane(p, v)
        hash_i = 1 if sign >= 0 else 0
        hash_value += 2**i * hash_i
    return hash_value    
```

side_of_plane function is defined like this:
```python
def side_of_plane(P, v):
    dotproduct = np.dot(P, v.T)
    sign_of_dot_product = np.sign(dotproduct)
    sign_of_dot_product_scalar = np.asscalar(sign_of_dot_product)
    return sign_of_dot_product_scalar
```
- We can convert a document into a vector by summing up element wise, the vectors for each word. Then we can use KNN, to search the document. photo: word-vectors-to-doc-vector.jpg and here  (code-to-doc-vector.jpg) is the code for making the document vector.

KNN can find documents, or sentence with similar meanings.


> Written with [StackEdit](https://stackedit.io/).