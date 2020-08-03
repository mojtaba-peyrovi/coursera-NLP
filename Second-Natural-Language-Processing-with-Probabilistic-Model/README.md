

## Coursera NLP by deeplearning.ai

### Course 2: Natural Language Processing with Probabilistic Model:

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

> Written with [StackEdit](https://stackedit.io/).