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
           #"leo_tolstoy": ["warap.txt", "annakarenina.txt"], 
           "mark_twain": ["toms.txt", "huckfinn.txt", "connecticutyankee.txt", "princepauper.txt"]}

stats = {}

for author in authors:
    
    vocab = {}
    
    for bookName in authors[author]:
        bookFilepath = "novels/" + author + "/" + bookName
        print(bookFilepath)
        
        # read text in book
        with open(bookFilepath, encoding='utf-8', errors='ignore') as f:
            text = f.read()[1:]

        # sentence segment, split based on punctuation
        # if ! or ?, OR .\s[A-Z] OR ."\s[A-Z]
        text = text.replace("\n", " ")
        p_sentence = re.compile(r'([?!]"?)|((?<!Dr|Mr|Ms|Jr|Sr|St)(?<!Mrs|Rev)\."?\s+)')
        sentence_list = re.split(p_sentence, text)
        
        # to reconstruct sentences as string
        sentences = []
        
        # Statistics to track per book
        sentence_count = 0
        stopword_count = 0
        total_words = 0
        capitalized_words = 0
        # punctuation_count
        dash_count = 0 # counting "--"
        comma_count = 0
        italics_count = 0 # _<word>_
        contractions_count = 0 # counting "you'll" etc.
        dialogue_count = 0 # counting ""
        in_dialogue = False
        
        for ele in sentence_list:
            if ele == None or len(ele) == 0:
                continue
            elif len(sentences) != 0 and ele[0] in "?!.":
                # count statistics
                sentences[-1] += ele.strip()
            else:
                # count statistics
                sentence_count += 1
                words = re.findall(r'\w*’?\w*', ele)
                capitalized = re.findall(r'[A-Z]\w+’?\w*', ele)
                dashes = re.findall(r'-+', ele)
                commas = re.findall(r',', ele)
                italics = re.findall(r'_\w+_', ele)
                contractions = re.findall(r'\w+’\w+', ele)
                if not in_dialogue:
                    dialogues = re.findall(r'[‘“].*', ele) # Look for start of dialogue
                    in_dialogue = len(dialogues) > 0
                else:
                    dialogues = re.findall(r'.*[”’]', ele) # Look for end of dialogue
                    in_dialogue = len(dialogues) == 0
                # contractions treated as one word
                total_words += len(words)
                capitalized_words += len(capitalized)
                dash_count += len(dashes)
                comma_count += len(commas)
                italics_count += len(italics)
                contractions_count += len(contractions)
                # Count sentences that have dialogue
                dialogue_count += 1 if in_dialogue else 0
                for word in words:
                    # TODO: Handle known words that are in ALL CAPS
                    # TODO: Handle named entities (people, cities?), count them as the same since not style
                    # TODO: Project Gutenberg seems to have italicised? words that are _<word>_, handle.
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        # counting word occurrence because might as well
                        vocab[word] += 1
                    if word in STOPWORDS:
                        stopword_count += 1
                sentences += [ele.strip()]

        avg_word_per_sent = total_words / sentence_count
        capitalized_per_sent = capitalized_words / sentence_count
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
        if "capitalized_per_sentence" not in stats:
            stats["capitalized_per_sentence"] = [capitalized_per_sent]
        else:
            stats["capitalized_per_sentence"] += [capitalized_per_sent]

        if "dashes_per_sent" not in stats:
            stats["dashes_per_sent"] = [dash_count/sentence_count]
        else:
            stats["dashes_per_sent"] += [dash_count/sentence_count]
        if "comma_count_per_sent" not in stats:
            stats["comma_count_per_sent"] = [comma_count/sentence_count]
        else:
            stats["comma_count_per_sent"] += [comma_count/sentence_count]
        if "italics_per_sent" not in stats:
            stats["italics_per_sent"] = [italics_count/sentence_count]
        else:
            stats["italics_per_sent"] += [italics_count/sentence_count]
        if "contractions_per_sent" not in stats:
            stats["contractions_per_sent"] = [contractions_count/sentence_count]
        else:
            stats["contractions_per_sent"] += [contractions_count/sentence_count]
        if "dialogue_per_sent" not in stats:
            stats["dialogue_per_sent"] = [dialogue_count/sentence_count]
        else:
            stats["dialogue_per_sent"] += [dialogue_count/sentence_count]

        # vocab needs to be able to count rare words and identify them
    #     if "vocab_word_count" not in stats:
    #         stats["vocab_word_count"] = [unique_word_count]
    #     else:
    #         stats["vocab_word_count"] += [unique_word_count]
        
print(stats['stopword_count_per_sent'])

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
        scaled = scaler.fit_transform(features)
    
    return pd.DataFrame(scaled)

# Count stats for the queries
def extractStats(queries, scaler):
    features = {}
    vocab = {}
    for query in queries:
        # split each sentence
        query = query.replace("\n", " ")
        p_sentence = re.compile(r'([?!]"?)|((?<!Dr|Mr|Ms|Jr|Sr|St)(?<!Mrs|Rev)\."?\s+)')
        sentence_list = re.split(p_sentence, query)
        
        sentence_count = 0
        total_words = 0
        capitalized_words = 0
        stopword_count = 0
        in_dialogue = False
        sentences = []
        # punctuation_count
        dash_count = 0 # counting "--"
        comma_count = 0
        italics_count = 0 # _<word>_
        contractions_count = 0 # counting "you'll" etc.
        dialogue_count = 0 # counting ""
    
        for ele in sentence_list:
            if ele == None or len(ele) == 0:
                continue
            elif len(sentences) != 0 and ele[0] in "?!.":
                # count statistics
                sentences[-1] += ele.strip()
            else:
                # count statistics
                sentence_count += 1
                words = re.findall(r'\w*’?\w*', ele)
                capitalized = re.findall(r'[A-Z]\w*’?\w*', ele)
                dashes = re.findall(r'-+', ele)
                commas = re.findall(r',', ele)
                italics = re.findall(r'_\w+_', ele)
                contractions = re.findall(r'\w+’\w+', ele)
                if not in_dialogue:
                    dialogues = re.findall(r'[‘“].*', ele) # Look for start of dialogue
                    in_dialogue = len(dialogues) > 0
                else:
                    dialogues = re.findall(r'.*[”’]', ele) # Look for end of dialogue
                    in_dialogue = len(dialogues) == 0
                # contractions treated as one word
                total_words += len(words)
                capitalized_words += len(capitalized)
                dash_count += len(dashes)
                comma_count += len(commas)
                italics_count += len(italics)
                contractions_count += len(contractions)
                # Count sentences that have dialogue
                dialogue_count += 1 if in_dialogue else 0
                for word in words:
                    # TODO: Handle known words that are in ALL CAPS
                    # TODO: Handle named entities (people, cities?), count them as the same since not style
                    # TODO: Project Gutenberg seems to have italicised? words that are _<word>_, handle.
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        # counting word occurrence because might as well
                        vocab[word] += 1
                    if word in STOPWORDS:
                        stopword_count += 1
                sentences += [ele.strip()]
        
        if "stopword_count_per_sent" not in features:
            features["stopword_count_per_sent"] = [stopword_count/sentence_count]
        else:
            features["stopword_count_per_sent"] += [stopword_count/sentence_count]
        if "avg_word_per_sentence" not in features:
            features["avg_word_per_sentence"] = [total_words/sentence_count]
        else:
            features["avg_word_per_sentence"] += [total_words/sentence_count]
        if "capitalized_per_sentence" not in features:
            features["capitalized_per_sentence"] = [capitalized_words/sentence_count]
        else:
            features["capitalized_per_sentence"] += [capitalized_words/sentence_count]
            
#         if "vocab_word_count" not in features:
#             features["vocab_word_count"] = [len(vocab.keys())]
#         else:
#             features["vocab_word_count"] += [len(vocab.keys())]
            
        if "dashes_per_sent" not in features:
            features["dashes_per_sent"] = [dash_count/sentence_count]
        else:
            features["dashes_per_sent"] += [dash_count/sentence_count]
        if "comma_count_per_sent" not in features:
            features["comma_count_per_sent"] = [comma_count/sentence_count]
        else:
            features["comma_count_per_sent"] += [comma_count/sentence_count]
        if "italics_per_sent" not in features:
            features["italics_per_sent"] = [italics_count/sentence_count]
        else:
            features["italics_per_sent"] += [italics_count/sentence_count]
        if "contractions_per_sent" not in features:
            features["contractions_per_sent"] = [contractions_count/sentence_count]
        else:
            features["contractions_per_sent"] += [contractions_count/sentence_count]
        if "dialogue_per_sent" not in features:
            features["dialogue_per_sent"] = [dialogue_count/sentence_count]
        else:
            features["dialogue_per_sent"] += [dialogue_count/sentence_count]
    
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
print("Class stats distribution")
print(x_train)

# y_train = books
y_train = []
for author in authors:
    y_train += [author] * len(authors[author])
print(y_train)

train_model(model, x_train, y_train)

from test_runner import *
from matplotlib import pyplot

test_cases = get_all_tests()

tests = pd.Series(test_cases)
x_test = extractStats(tests, scaler)
print("test queries' stats:")
print(x_test)

output_answers = predict(model, x_test)
check_test_results(output_answers)

from test_cases import CHARLES_DICKENS_TESTS, FYODOR_DOSTOEVSKY_TESTS, LEO_TOLSTOY_TESTS, MARK_TWAIN_TESTS

# Names of authors
CHARLES_DICKENS_NAME = "charles_dickens"
FYODOR_DOSTOEVSKY_NAME = "fyodor_dostoevsky"
LEO_TOLSTOY_NAME = "leo_tolstoy"
MARK_TWAIN_NAME = "mark_twain"
ALL_AUTHOR_NAMES = [CHARLES_DICKENS_NAME, FYODOR_DOSTOEVSKY_NAME, LEO_TOLSTOY_NAME, MARK_TWAIN_NAME]

# Get F1 Score
correct_answers = [CHARLES_DICKENS_NAME for _ in CHARLES_DICKENS_TESTS] + \
                  [FYODOR_DOSTOEVSKY_NAME for _ in FYODOR_DOSTOEVSKY_TESTS] + \
                  [LEO_TOLSTOY_NAME for _ in LEO_TOLSTOY_TESTS] + \
                  [MARK_TWAIN_NAME for _ in MARK_TWAIN_TESTS]

# Use f1-macro as the metric
score = f1_score(correct_answers, output_answers, average='macro')
print('LR score on validation = {}'.format(score))
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(correct_answers, output_answers))
print(classification_report(correct_answers, output_answers))

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
