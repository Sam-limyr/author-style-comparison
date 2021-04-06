import numpy as np
import pandas as pd
import re
import os
import sys

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords   # Requires NLTK in the include path.

## List of stopwords
STOPWORDS = stopwords.words('english') # type: list(str)

authors = {"charles_dickens": ["davidc.txt", "greatex.txt", "olivert.txt", "twocities.txt"],
           "fyodor_dostoevsky": ["crimep.txt", "idiot.txt", "possessed.txt"], 
           "leo_tolstoy": ["warap.txt", "annakarenina.txt"], 
           "mark_twain": ["toms.txt", "huckfinn.txt", "connecticutyankee.txt", "princepauper.txt"]}
authorsIndexed = ["charles_dickens", "fyodor_dostoevsky", "leo_tolstoy", "mark_twain"]

num_novels = sum([len(authors[author]) for author in authors])

stats = {}
vocab = {}

for i in range(len(authorsIndexed)):
    author = authorsIndexed[i]
    
    # initialize
    # vocab[author] = {}
    stats[author] = {}
    
    # calculate logprior
    stats[author]["logprior"] = np.log(len(authors[author]) / num_novels)
    
    # to help calc loglikelihood
    total_words_for_author = 0
    stats[author]["loglikelihood"] = {}
    
    for bookName in authors[author]:
        bookFilepath = "novels/" + author + "/" + bookName
        print(bookFilepath)
        # filepath = os.path.join(sys.path[1], bookFilepath)
        # print(filepath)
        
        # read text in book
        with open(bookFilepath, encoding='utf-8', errors='ignore') as f:
            text = f.read()[1:]
        
        words = re.findall(r'\w*’?\w*', text)
        
        for word in words:
            # don't count stopwords
            if word in STOPWORDS:
                continue
            total_words_for_author += 1
            if word == '':
                continue
            # TODO: Handle known words that are in ALL CAPS
            if word.isupper():
                word = word.lower()
            # TODO: Project Gutenberg seems to have italicised? words that are _<word>_, handle.
#             if re.match(r'_\w+_', word):
#                 word = word.replace('_', '')
            # TODO: Handle named entities (people, cities?), count them as the same since not style
            
            # just set everything to lowercase
            word = word.lower()
            if word not in vocab:
                vocab[word] = [0] * len(authorsIndexed)
                vocab[word][i] = 1
            else:
                # counting word occurrence
                vocab[word][i] += 1
    
    # calculate loglikelihood
#     for word, count in vocab[author].items():
#         stats[author]["loglikelihood"][word] = np.log(count / (total_words_for_author - count))
                
print("Unique words in combined vocabulary: {}".format(len(vocab.keys())))

def viewVocabCounts(vocab, num_of_entries):
    counter = 0
    for word, count in vocab.items():
        if (counter < 10):
            counter += 1
            print(word, count)

viewVocabCounts(vocab, 10)

from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

# extract stats for each author
def extractFeatures(stats, scaler, isTestSet):
    features = pd.DataFrame(data=stats)
    
#     if isTestSet:
#         scaled = scaler.transform(features)
#     else:
#         scaler.fit(features)
#         scaled = scaler.fit_transform(features)
    
    return pd.DataFrame(features)

def train_model(model, x_train, y_train):
    ''' TODO: train your model based on the training data '''
    model.fit(x_train, y_train)
    pass

def predict(model, x_test):
    ''' TODO: make your prediction here '''
    return model.predict(x_test)

model = MultinomialNB(fit_prior=False)
scaler = StandardScaler()
            
x_train = extractFeatures(vocab, scaler, False)

# x_train = array of [books, words]
# y_train = books
y_train = authorsIndexed
train_model(model, x_train, y_train)

from test_runner import *
from matplotlib import pyplot

# only count words that are in known vocab, ignore OOV words
def extractWords(queries, vocab):
    query_vocab = {}
    for i in range(len(queries)):
        query = queries[i]
        words = re.findall(r'\w*’?\w*', query)
        
        for word in words:
            if word in vocab:
                if word not in query_vocab:
                    query_vocab[word] = [0] * len(queries)
                    query_vocab[word][i] = 1
                else:
                    query_vocab[word][i] += 1 
                    
    for word, count in vocab.items():
        if word not in query_vocab:
            query_vocab[word] = [0] * len(queries)
    
    features = pd.DataFrame(data=query_vocab)
    return features

test_cases = get_all_tests()

tests = pd.Series(test_cases)
x_test = extractWords(tests, vocab)
print("test queries' stats:")
print(x_test)

output_answers = predict(model, x_test)
check_test_results(output_answers)

# get importance
importance = model.coef_
# summarize feature importance
print("Feature importance for Charles Dickens:")
# plot feature importance
pyplot.bar([x for x in range(len(importance[0]))], importance[0])
pyplot.show()

# print("Feature importance for 0:")
# #     for i,v in enumerate(importance[1]):
# #         print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# pyplot.bar([x for x in range(len(importance[1]))], importance[1])
# pyplot.show()

# print("Feature importance for 1")
# #     for i,v in enumerate(importance[2]):
# #         print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# pyplot.bar([x for x in range(len(importance[2]))], importance[2])
# pyplot.show()