

## Coursera NLP by deeplearning.ai

### Course 1: Classification and Vector Spaces:
- Week1:
In order to create a sentiment analysis on a tweet using logistic regression, we need to follow the steps below:
1- extract features from the text
2- train the logistic regression classifier based on the features X
3- make the predictions (classify)

 Imagine we have a list of tweets: [tweet1, tweet2, tweet3, ...] , which is called **Corpus** . Now, we need to make a list of all UNIQUE words from all tweets. we call the new list **Vocabulary list*** and we show it by letter **V**.
 Then for each tweet, we can do **Feature Extraction.** to do so, we need to see what words in the vocabulary list appears in the tweet. For each word in V, if the value exist in the tweet, we assign value 1 to it, and if doesn't exist, we assign values of 0. Of course, the matrix containing the features has so many zeros and will be considered a **Sparse** representation.

#### Problems with sparse representation:
We have a large list of 0-1 which is as big as the length of the V vector.

$$
[\theta _{1}, \theta _{2}, \theta _{3}, ... , \theta _{n}] 
, n = \left | V \right |
$$
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

$$
X_{m} = [1, \sum_{w}^{}freqs(w,1), \sum_{w}freqs(w,0) ]
$$
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
Now for each tweet and its theta we have the prediction like this:
$$
pred = h(X_{val}, \theta) \geq 0.5
$$
because we are using Sigmoid function. For example, if we have h(x) values for each tweet as below:
$$
\begin{bmatrix}
0.3\\ 
0.5\\ 
0.7\\ 
.\\
.\\
h_{m}
\end{bmatrix}
$$
We compare them one by one against 0.5, and if its bigger we have 1, if smaller 0:
$$
\begin{bmatrix}
0\\ 
1\\ 
1\\ 
1\\ 
.\\
.\\
pred_{m}
\end{bmatrix}
$$
Then we compare the prediction values (the above matrix) vs the real values, and we have a new matrix with the comparisons. If the values match, we return 1, if not we return 0. 
$$
\sum_{i=1}^{m}\frac{(pred^{(i)} == y_{val}^{(i)})}{m}
$$
the sigma above, sums up all zero and ones (showing the total times the prediction was correct), and divides them by total number of tweets, and returns the accuracy percentage.








> Written with [StackEdit](https://stackedit.io/).