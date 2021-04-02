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


def read_texts(dir_name):
	authors = os.listdir(dir_name)
	author_to_alltexts = {}
	for author in authors:
		novel_filenames = os.listdir(dir_name + "/" + author)
		author_to_alltexts[author] = {}

		for novel_filename in novel_filenames:
			filepath = dir_name + "/" + author + "/" + novel_filename
			fp = open(filepath, "r", encoding='utf8')
			text = fp.read()[1:]
			print(novel_filename)

			author_to_alltexts[author][novel_filename] = text

	return author_to_alltexts


def parse_tokens(author_to_alltexts):
	# tokenize into words
	author_to_title_to_tokens = {}
	tbTokenizer = TreebankWordTokenizer()

	for author, title_to_text in author_to_alltexts.items():
		title_to_tokens = {}

		for title, text in title_to_text.items():
			sentences = sent_tokenize(text)
			tokensList = []

			for sent in sentences:
				tokens = tbTokenizer.tokenize(sent)
				tokensList.extend(tokens)

			title_to_tokens[title] = tokensList

		author_to_title_to_tokens[author] = title_to_tokens

	return author_to_title_to_tokens


def compute_stopword_freq(stopword_set, author_to_alltexts, author_to_title_to_tokens):
	# count num of stopwords
	author_to_title_to_stopwordcounter = {}

	for author, title_to_text in author_to_alltexts.items():
		title_to_counts = {}

		for title in title_to_text.keys():
			stopwords_count = Counter()
			tokens = author_to_title_to_tokens[author][title]

			for token in tokens:
				if token in stopword_set:
					stopwords_count[token] += 1

			title_to_counts[title] = stopwords_count

		author_to_title_to_stopwordcounter[author] = title_to_counts

	return author_to_title_to_stopwordcounter


def compute_rank_vectors(sorted_stopword_list, author_to_title_to_stopwordcounter):
	# rank top TOP_X_MOST_COMMON stopwords for each author corp, rank starts from 0
	author_to_title_to_vector = {}
	vector_to_authortitle = {}
	all_text_vecs = []
	vec_index = 0

	for author, title_to_stopword_count in author_to_title_to_stopwordcounter.items():
		author_to_title_to_vector[author] = {}

		for title, stopword_count in title_to_stopword_count.items():
			ranked_stopword_list = stopword_count.most_common()  # return all words

			ranked_stopword_dict = {tup[0]: index for index, tup in enumerate(ranked_stopword_list)}
			rank_vector = []

			for word in sorted_stopword_list:
				if word in ranked_stopword_dict:
					rank_vector.append(ranked_stopword_dict[word])
				else:
					rank_vector.append(len(ranked_stopword_list))  # ranked last place if word is not found

			author_to_title_to_vector[author][title] = rank_vector
			vector_to_authortitle[vec_index] = (author, title)
			all_text_vecs.append(rank_vector)

			vec_index += 1

	return all_text_vecs, author_to_title_to_vector, vector_to_authortitle


def generate_stopword_set():
	stopword_list = []
	tbTokenizer = TreebankWordTokenizer()

	# trim apostrophe shorthands
	for word in stopwords.words('english'):
		tokens = tbTokenizer.tokenize(word)
		stopword_list.append(tokens[0])

	stopword_set = set(dict.fromkeys(stopword_list).keys())  # remove duplicates

	return stopword_set


def test_model(nn_model, stopword_set, train_vector_to_authortitle):
	print("Testing model... ")
	# get training data
	author_to_alltexts = read_texts('supplementaryNovels')
	author_to_title_to_tokens = parse_tokens(author_to_alltexts)

	sorted_stopword_list = sorted(stopword_set)

	author_to_title_to_stopwordcounter = compute_stopword_freq(stopword_set, author_to_alltexts, author_to_title_to_tokens)
	test_vecs, author_to_title_to_vector, vector_to_authortitle = compute_rank_vectors(sorted_stopword_list, author_to_title_to_stopwordcounter)

	for index in range(len(test_vecs)):
		test_vector = test_vecs[index]
		dist_arr, point_index_arr = nn_model.kneighbors([test_vector])

		print("Testing "+vector_to_authortitle[index][1] + " by "+vector_to_authortitle[index][0])
		print("Nearest points: ")

		for point_index in range(len(dist_arr[0])):
			print("Point: " + str(train_vector_to_authortitle[point_index_arr[0][point_index]]))
			print("Distance to point: " + str(dist_arr[0][point_index]))

		print(" ")


if __name__ == '__main__':
	print("Building model...")
	# get training data
	author_to_alltexts = read_texts('novels')
	author_to_title_to_tokens = parse_tokens(author_to_alltexts)

	stopword_set = generate_stopword_set()
	sorted_stopword_list = sorted(stopword_set)

	author_to_title_to_stopwordcounter = compute_stopword_freq(stopword_set, author_to_alltexts, author_to_title_to_tokens)
	all_text_vecs, author_to_title_to_vector, vector_to_authortitle = compute_rank_vectors(sorted_stopword_list, author_to_title_to_stopwordcounter)

	# print(author_to_title_to_stopwordcounter['charles_dickens'])
	# print("Stopwordlist: ")
	# print(sorted_stopword_list)
	# print("Stopword counter: ")
	# print(author_to_title_to_stopwordcounter['charles_dickens']['davidc.txt'])
	# print("Rank vector: ")
	# print(author_to_title_to_vector['charles_dickens']['davidc.txt'])

	# build model
	nn_model = NearestNeighbors(n_neighbors=2, metric='manhattan')
	nn_model.fit(all_text_vecs)

	# testing with supp novels
	test_model(nn_model, stopword_set, vector_to_authortitle)