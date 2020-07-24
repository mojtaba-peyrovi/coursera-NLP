

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