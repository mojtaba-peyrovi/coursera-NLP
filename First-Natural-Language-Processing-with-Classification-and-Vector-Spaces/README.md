

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











> Written with [StackEdit](https://stackedit.io/).