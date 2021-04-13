import os

from collections import Counter
from collections import defaultdict
from nltk.corpus import stopwords   # Requires NLTK in the include path.
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
from sklearn.neighbors import NearestNeighbors

from test_runner import *
from test_cases import *

# NUMBER OF FUNCTION WORDS IN nltk.coprus.stopwords is 160 after collapsing cases
NUM_NEAREST_NEIGHBOUR = 17
NUM_TEXT_SPLIT = 5

def read_texts(dir_name):
	"""
	iterates through text files in subdirectories in dir_name
	subdirectory names are parsed as authors
	returns dict of author to filename to text
	"""
	print("Reading texts...")
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

			segment_len = math.floor(len(text)/NUM_TEXT_SPLIT)

			# author_to_alltexts[author][novel_filename] = text
			for idx in range(0, NUM_TEXT_SPLIT):
				author_to_alltexts[author][novel_filename + str(idx)] = text[segment_len * idx : segment_len*(idx+1)]


	return author_to_alltexts


def parse_tokens(author_to_alltexts):
	"""
	tokenizes text in dict of author to filename to text using nltk TreebankWordTokenizer
	returns dict of author to title to tokens
	"""
	print("Tokenizing...")

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
			# title_to_tokens[title+"1"] = tokensList[:math.floor(len(tokensList)/2)]
			# title_to_tokens[title+"2"] = tokensList[math.floor(len(tokensList)/2):len(tokensList)]

		author_to_title_to_tokens[author] = title_to_tokens

	return author_to_title_to_tokens


def compute_stopword_freq(stopword_set, author_to_alltexts, author_to_title_to_tokens):
	"""
	counts no. of stopwords (given in stopword_set) found in tokens
	returns dict of author to filename to counter for stopword freq
	"""
	print("Computing stopword frequency...")

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
	"""
	ranks stopword frequency in order of sorted_stopword_list (alphabetical order) for each text
	returns list of rank vectors, dict of author to title to vector,
	dict of vector index to (author, title) tup
	"""
	print("Computing rank vectors...")

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
	"""
	Tokenize each stopword to get rid of shorthand forms
	returns set of stopwords
	"""

	stopword_list = []
	tbTokenizer = TreebankWordTokenizer()

	# trim apostrophe shorthands
	for word in stopwords.words('english'):
		tokens = tbTokenizer.tokenize(word)
		stopword_list.append(tokens[0])

	stopword_set = set(dict.fromkeys(stopword_list).keys())  # remove duplicates

	return stopword_set


def run_test_supplementaryNovels_entiretext(nn_model, stopword_set, train_vector_to_authortitle):
	"""
	test model using texts in supplementary novesl
	"""

	print("Testing model on whole texts in supplementaryNovel dataset... ")
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

		# print("Nearest points: ")
		# for point_index in range(len(dist_arr[0])):
		# 	print("Point: " + str(train_vector_to_authortitle[point_index_arr[0][point_index]]))
		# 	print("Distance to point: " + str(dist_arr[0][point_index]))

		nearest_auth, confidence_score = compute_nearest_neighbour(point_index_arr, dist_arr, train_vector_to_authortitle)
		print("Nearest author by majority: "+nearest_auth)
		print("Confidence score: " + str(confidence_score))
		print(" ")

		# print("Most similar author: "+str(train_vector_to_authortitle[point_index_arr[0][0]]))
		#
		# confidence_score = compute_confidence_score(dist_arr[0][0], dist_arr[0][1])
		# print("Confidence score: " + str(confidence_score))


def compute_nearest_neighbour(point_index_arr, dist_arr, train_vector_to_authortitle):
	author_counter = Counter()
	auth_to_point_index = {}

	# count among NUM_NEAREST_NEIGHBOUR neighbours, which is most common
	for point_index in range(NUM_NEAREST_NEIGHBOUR):
		author, title = train_vector_to_authortitle[point_index_arr[0][point_index]]
		author_counter[author] += 1

		if auth_to_point_index.get(author) != None:
			auth_to_point_index[author].append(point_index)
		else:
			auth_to_point_index[author] = [point_index]

	ranked_author_list = author_counter.most_common()
	most_common_auth_count = int(ranked_author_list[0][1])

	nearest_auth = ""
	distance1 = 0
	distance2 = float('inf')

	# case 1: all 3 diff authors, take nearest point
	if most_common_auth_count == 1:
		nearest_auth_tup = train_vector_to_authortitle[point_index_arr[0][0]]
		nearest_auth = nearest_auth_tup[0]
		distance1 = dist_arr[0][0]
		distance2 = dist_arr[0][1] # second nearest point

	# case 2: all same author
	elif most_common_auth_count == NUM_NEAREST_NEIGHBOUR:
		nearest_auth = ranked_author_list[0][0]
		distance1 = dist_arr[0][0]

		diff_auth_index = find_nearest_point_with_diff_auth(point_index_arr, train_vector_to_authortitle, nearest_auth)
		print(diff_auth_index)
		distance2 = dist_arr[0][diff_auth_index]

	# case 3: authorA - 2, authorB - 1, majority wins
	else:
		#  elif: most_common_auth_count >= math.ceil(NUM_NEAREST_NEIGHBOUR/2)
		# distance1, distance2, nearest_auth = find_majority_point(auth_to_point_index, dist_arr, point_index_arr,
		# 												ranked_author_list, train_vector_to_authortitle)

		distance1, distance2, nearest_auth = find_dist_weighted_nearest_author(dist_arr, point_index_arr,
																			   train_vector_to_authortitle)

	confidence_score = compute_confidence_score(distance1, distance2)

	return nearest_auth, confidence_score


def find_majority_point(auth_to_point_index, dist_arr, point_index_arr, ranked_author_list, train_vector_to_authortitle):
	nearest_auth = ranked_author_list[0][0]

	# computing distances
	nearest_point_index = min(auth_to_point_index[nearest_auth])  # authorA: nearest point of majority auth
	distance1 = dist_arr[0][nearest_point_index]

	authorB_index = find_nearest_point_with_diff_auth(point_index_arr, train_vector_to_authortitle, nearest_auth)
	distance2 = dist_arr[0][authorB_index]  # authorB:  nearest point of minority auth

	return distance1, distance2, nearest_auth


def find_dist_weighted_nearest_author(dist_arr, point_index_arr, train_vector_to_authortitle):
	auth_to_distance_sum = Counter()

	for point_index in range(NUM_NEAREST_NEIGHBOUR):
		dist = dist_arr[0][point_index]
		auth, title = train_vector_to_authortitle[point_index_arr[0][point_index]]

		auth_to_distance_sum[auth] += dist

	ranked_auth_list_by_dist_sum = auth_to_distance_sum.most_common()
	nearest_auth = ranked_auth_list_by_dist_sum[0][0]

	# computing distances
	distance1 = ranked_auth_list_by_dist_sum[0][1]
	distance2 = ranked_auth_list_by_dist_sum[1][1]

	return distance1, distance2, nearest_auth


def find_nearest_point_with_diff_auth(point_index_arr, train_vector_to_authortitle, auth):
	for point_index in range(len(point_index_arr[0])):
		test_auth = train_vector_to_authortitle[point_index_arr[0][point_index]][0]

		if test_auth != auth:
			return point_index

	return "help"


def compute_confidence_score(distance1, distance2):
	"""
	case 1: all 3 diff authors, distance1=nearest_point, distance2=sec_nearest_point
	case 2: all 3 same author, distance1=nearest_point, distance2=nearest_point_diff_auth
	case 3: 2 authors, 1 author: distance1=best(author1, author1), distance2=author2

	"""
	return math.fabs(1 - (distance1 / distance2))


def run_test_runner(nn_model, stopword_set, train_vector_to_authortitle):
	print("Running test_runner... ")

	author_to_alltexts = defaultdict(dict)

	for index in range(len(CHARLES_DICKENS_TESTS)):
		author_to_alltexts[0][index] = CHARLES_DICKENS_TESTS[index]

	# for index in range(len(CHARLES_DICKENS_SAME_BOOK_MULTIPLE_PARAGRAPH_TESTS)):
	# 	author_to_alltexts[0][index] = CHARLES_DICKENS_SAME_BOOK_MULTIPLE_PARAGRAPH_TESTS[index]

	for index in range(len(FYODOR_DOSTOEVSKY_TESTS)):
		author_to_alltexts[1][index] = FYODOR_DOSTOEVSKY_TESTS[index]

	for index in range(len(MARK_TWAIN_TESTS)):
		author_to_alltexts[3][index] = MARK_TWAIN_TESTS[index]

	for index in range(len(JANE_AUSTEN_TESTS)):
		author_to_alltexts[4][index] = JANE_AUSTEN_TESTS[index]

	for index in range(len(JOHN_STEINBECK_TESTS)):
		author_to_alltexts[5][index] = JOHN_STEINBECK_TESTS[index]

	author_to_title_to_tokens = parse_tokens(author_to_alltexts)
	sorted_stopword_list = sorted(stopword_set)

	author_to_title_to_stopwordcounter = compute_stopword_freq(stopword_set, author_to_alltexts, author_to_title_to_tokens)
	test_vecs, author_to_title_to_vector, vector_to_authortitle = compute_rank_vectors(sorted_stopword_list, author_to_title_to_stopwordcounter)

	auth_confidence_list = []
	auth_list = []

	for index in range(len(test_vecs)):
		test_vector = test_vecs[index]
		dist_arr, point_index_arr = nn_model.kneighbors([test_vector])

		print("Testing "+str(vector_to_authortitle[index][1]) + " by "+str(vector_to_authortitle[index][0]))
		nearest_auth, confidence_score = compute_nearest_neighbour(point_index_arr, dist_arr,
																   train_vector_to_authortitle)

		# print("Most similar author: "+nearest_auth)
		# print("Confidence score: " + str(confidence_score))
		#
		# print("Nearest points: ")
		# for point_index in range(len(dist_arr[0])):
		# 	print("Point: " + str(train_vector_to_authortitle[point_index_arr[0][point_index]]))
		# 	print("Distance to point: " + str(dist_arr[0][point_index]))

		auth_confidence_list.append((nearest_auth,confidence_score))
		auth_list.append(nearest_auth)

	# print(auth_confidence_list)
	check_test_results(auth_list)

	return auth_confidence_list


def combine_training_data():
	main_texts = read_texts('novels')
	supp_texts = read_texts('supplementaryNovels')

	# combining supp and main
	author_to_alltexts = {}
	for author in main_texts.keys():
		combined = {}
		combined.update(main_texts.get(author))

		if supp_texts.get(author) != None:
			combined.update(supp_texts.get(author))

		author_to_alltexts[author] = combined
	return author_to_alltexts


def main():
	# get training data
	author_to_alltexts = read_texts('novels')
	author_to_title_to_tokens = parse_tokens(author_to_alltexts)

	stopword_set = generate_stopword_set()
	sorted_stopword_list = sorted(stopword_set)

	author_to_title_to_stopwordcounter = compute_stopword_freq(stopword_set, author_to_alltexts, author_to_title_to_tokens)
	all_text_vecs, author_to_title_to_vector, vector_to_authortitle = compute_rank_vectors(sorted_stopword_list, author_to_title_to_stopwordcounter)

	# build model
	print("Building model...")
	nn_model = NearestNeighbors(n_neighbors=19, metric='manhattan')
	nn_model.fit(all_text_vecs)

	# testing
	# test_model(nn_model, stopword_set, vector_to_authortitle)
	return run_test_runner(nn_model, stopword_set, vector_to_authortitle)


if __name__ == '__main__':
	main()