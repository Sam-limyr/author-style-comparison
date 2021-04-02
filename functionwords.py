import pandas as pd
import re
import os
import sys
import nltk

from collections import Counter
from nltk.corpus import stopwords   # Requires NLTK in the include path.
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
from sklearn.neighbors import NearestNeighbors

# NUM_FUNCTION_WORDS = 160

CHARLES_DICKENS_NAME = "charles_dickens"
FYODOR_DOSTOEVSKY_NAME = "fyodor_dostoevsky"
LEO_TOLSTOY_NAME = "leo_tolstoy"
MARK_TWAIN_NAME = "mark_twain"

author_names = [CHARLES_DICKENS_NAME, FYODOR_DOSTOEVSKY_NAME, LEO_TOLSTOY_NAME, MARK_TWAIN_NAME]
function_words_pos_tagset = ['DT', 'CC', 'IN', 'PRP', 'PRP$', 'WP', 'WP$']

tbTokenizer = TreebankWordTokenizer()
stopword_list = []

# trim apostrophe shorthands
for word in stopwords.words('english'):
	tokens = tbTokenizer.tokenize(word)
	stopword_list.append(tokens[0])

stopword_set = set(dict.fromkeys(stopword_list).keys()) # remove duplicates
sorted_stopword_list = sorted(stopword_set)

authors = os.listdir("novels")
author_to_alltexts = {}

for author in authors:
	novel_filenames = os.listdir("novels/"+author)

	author_corp = ""

	for novel_filename in novel_filenames:
		filepath = "novels/"+author+"/"+novel_filename
		fp = open(filepath, "r", encoding='utf8')
		text = fp.read()[1:]
		print(novel_filename)

		author_corp += text

	author_to_alltexts[author] = author_corp

# tokenize into words
author_to_tokens = {}

for author, corp in author_to_alltexts.items():
	sentences = sent_tokenize(corp)

	tokensList = []

	for sent in sentences:
		tokens = tbTokenizer.tokenize(sent)
		tokensList.extend(tokens)

	author_to_tokens[author] = tokensList

# count num of stopwords
author_to_stopwordcounter = {}

for author, tokens in author_to_tokens.items():
	stopwords_count = Counter()

	for token in tokens:
		if token in stopword_set:
			stopwords_count[token] += 1

	author_to_stopwordcounter[author] = stopwords_count

# rank top TOP_X_MOST_COMMON stopwords for each author corp, rank starts from 0
author_to_rank = {}

for author, stopword_count in author_to_stopwordcounter.items():
	ranked_stopword_list = stopword_count.most_common() # return all words

	ranked_stopword_dict = {tup[0]: index for index, tup in enumerate(ranked_stopword_list)}
	rank_vector = []

	for word in sorted_stopword_list:
		if word in ranked_stopword_dict:
			rank_vector.append(ranked_stopword_dict[word])
		else:
			rank_vector.append(len(ranked_stopword_list)) # ranked last place if word is not found

	author_to_rank[author] = rank_vector

# print("Stopwordlist: ")
# print(sorted_stopword_list)
# print("Stopword counter: ")
# print(author_to_stopwordcounter['charles_dickens'])
print("Rank vector: ")
print(author_to_rank['charles_dickens'])
print("Rank vector: ")
print(author_to_rank['mark_twain'])

