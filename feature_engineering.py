import pandas as pd
import re
import os
import sys

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords   # Requires NLTK in the include path.

## List of stopwords
STOPWORDS = stopwords.words('english') # type: list(str)

authors = {"charles_dickens": ["davidc.txt"], "fyodor_dostoevsky": ["crimep.txt"], 
           "leo_tolstoy": ["warap.txt"], "mark_twain": ["toms.txt"]}

stats = {}

for author in authors:
    
    # Statistics to track per author
    sentence_count = 0
    stopword_count = 0
    total_words = 0
    # punctuation_count?
    
    vocab = {}
    
    for bookFilepath in authors[author]:
        filepath = os.path.join(sys.path[0], bookFilepath)
        
        # read text in book
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            text = f.read()[1:]

        # sentence segment, split based on punctuation
        # if ! or ?, OR .\s[A-Z] OR ."\s[A-Z]
        text = text.replace("\n", " ")
        p_sentence = re.compile(r'([?!]"?)|((?<!Dr|Mr|Ms|Jr|Sr|St)(?<!Mrs|Rev)\."?\s+)')
        sentence_list = re.split(p_sentence, text)
        sentences = []
        for ele in sentence_list:
            if ele == None or len(ele) == 0:
                continue
            # avoid out of range error
            elif (len(sentences) == 0):
                # count statistics
                sentence_count += 1
                words = re.findall(r'\w\'?\w+', ele)
                # contractions treated as one word
                total_words += len(words)
                for word in words:
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        # counting word occurrence because might as well
                        vocab[word] += 1
                    if word in STOPWORDS:
                        stopword_count += 1
                sentences += [ele.strip()]
            elif ele[0] in "?!.":
                # count statistics
                sentences[-1] += ele.strip()
            else:
                # count statistics
                sentence_count += 1
                words = re.findall(r'\w\'?\w+', ele)
                # contractions treated as one word
                total_words += len(words)
                for word in words:
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        # counting word occurrence because might as well
                        vocab[word] += 1
                    if word in STOPWORDS:
                        stopword_count += 1
                sentences += [ele.strip()]

        avg_word_per_sent = total_words / sentence_count
        unique_word_count = len(vocab.keys())
        # store stats in outer dictionary, in order of "authors" dictionary
        if "stopword_count_per_sent" not in stats:
            stats["stopword_count_per_sent"] = [stopword_count/sentence_count]
        else:
            stats["stopword_count_per_sent"] += [stopword_count/sentence_count]
        if "avg_word_per_sentence" not in stats:
            stats["avg_word_per_sentence"] = [avg_word_per_sent]
        else:
            stats["avg_word_per_sentence"] += [avg_word_per_sent]
        if "vocab_word_count" not in stats:
            stats["vocab_word_count"] = [unique_word_count]
        else:
            stats["vocab_word_count"] += [unique_word_count]
        
        print("Number of sentences: {} \nNumber of total words: {} \nNumber of stopwords: {} \nAvg. words per sentence: {} \nWords in vocab: {}".format(sentence_count, total_words, stopword_count, avg_word_per_sent, unique_word_count))
        
print(stats)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

def train_model(model, x_train, y_train):
    ''' TODO: train your model based on the training data '''
    model.fit(x_train, y_train)
    pass

def predict(model, x_test):
    ''' TODO: make your prediction here '''
    return model.predict(x_test)

# extract stats for each author
def extractFeatures(stats, scaler, isTestSet):
    features = pd.DataFrame(data=stats)
    
    if isTestSet:
        scaled = scaler.transform(features)
    else:
        scaler.fit(features)
        mean = scaler.mean_
        scaled = scaler.fit_transform(features)
    
    return pd.DataFrame(scaled)

# Count stats for the queries
def extractStats(queries, scaler):
    features = {}
    vocab = {}
    for query in queries:
        stopword_count = 0
        
        words = re.findall(r'\w\'?\w+', query)
        # contractions treated as one word
        words_in_sent = len(words)
        for word in words:
            if word not in vocab:
                vocab[word] = 1
            else:
                # counting word occurrence because might as well
                vocab[word] += 1
            if word in STOPWORDS:
                stopword_count += 1
        if "stopword_count_per_sent" not in features:
            features["stopword_count_per_sent"] = [stopword_count]
        else:
            features["stopword_count_per_sent"] += [stopword_count]
        if "avg_word_per_sentence" not in features:
            features["avg_word_per_sentence"] = [words_in_sent]
        else:
            features["avg_word_per_sentence"] += [words_in_sent]
        if "vocab_word_count" not in features:
            features["vocab_word_count"] = [len(vocab.keys())]
        else:
            features["vocab_word_count"] += [len(vocab.keys())]
    
    features = pd.DataFrame(data=features)
    scaled = scaler.transform(features)
    return scaled

# Logistic Regression Model

# x_train = array of sentences, from all authors
# y_train = array of authors of corresponding sentence in x_train

# C = regularization strength
# max_iter, default=100, can try increase see got diff anot
# multi_class, can be 'ovr' = binary for each label or 'multinomial'
model = LogisticRegression(penalty='l2', C=0.8, solver='saga', multi_class='multinomial', max_iter=1000)

scaler = StandardScaler()
x_train = extractFeatures(stats, scaler, False)
y_train = [
    "charles_dickens",
    "fyodor_dostoevsky",
    "leo_tolstoy",
    "mark_twain"
]
train_model(model, x_train, y_train)

# Queries
queries = [
    "As he was passing by the house where Jeff Thatcher lived, he saw a new girl in the garden--a lovely little blue-eyed creature with yellowhair plaited into two long-tails, white summer frock and embroidered pan-talettes.",
    "He was crushed by poverty, but the anxieties of his position had of late ceased to weigh upon him.",
    "He had given up attending to matters of practical importance; he had lost all desire to do so.",
    "Nothing that any landlady could do had a real terror for him.",
    "Is the movement of the peoples at the time of the Crusades explained by the life and activity of the Godfreys and the Louis-es and their ladies?",
    "Thoughtfully, for I could not be here once more, and so near Agnes, without the revival of those regrets with which I had so long been occupied"
]
labels = [
    "mark_twain",
    "fyodor_dostoevsky",
    "fyodor_dostoevsky",
    "fyodor_dostoevsky",
    "leo_tolstoy",
    "charles_dickens"
]

queries = pd.Series(queries)
x_test = extractStats(queries, scaler)
print("queries' stats:")
print(x_test)
y_test = pd.Series(labels)

# test your model
y_pred = predict(model, x_test)

print("Predictions: \n")
print(y_pred)

# Use f1-macro as the metric
score = f1_score(y_test, y_pred, average='macro')
print('LR score on validation = {}'.format(score))
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))