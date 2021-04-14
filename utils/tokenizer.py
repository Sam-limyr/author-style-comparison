import os
import csv
import pandas as pd
import re

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize

authors = os.listdir("../data/train/novels")
author_to_sent = {}
author_to_para = {}

for author in authors:
	novel_filenames = os.listdir("novels/"+author)

	sentence_list = []
	para_list = []

	for novel_filename in novel_filenames:
		filepath = "novels/"+author+"/"+novel_filename
		fp = open(filepath, "r", encoding='utf8')
		text = fp.read()
		print(novel_filename)

		noEmptyParas = [para for para in text.split("\n\n") if para != '']

		sentence_list.extend(sent_tokenize(text))
		para_list.extend(noEmptyParas)

	author_to_sent[author] = sentence_list
	author_to_para[author] = para_list

# for author in author_to_sent.keys():
# 	path = 'tokens/' + author + '.csv'
#
# 	with open(path, 'w', newline='', encoding="utf-8") as csvfile:
# 		csvwriter = csv.writer(csvfile)
#
# 		for sent in author_to_sent[author]:
# 			csvwriter.writerow([sent])

for author in author_to_para.keys():
	path = 'paragraphs/' + author + '.csv'

	with open(path, 'w', newline='', encoding="utf-8") as csvfile:
		csvwriter = csv.writer(csvfile)

		for para in author_to_para[author]:
			csvwriter.writerow([para])

# for reading
# with open('tokens/charles_dickens.csv', 'r', newline='', encoding="utf-8") as readfile:
# 	csvreader = csv.reader(readfile)
# 	list = []
#
# 	for row in csvreader:
# 		list.extend(row)
#
# 	print(list)




# remove text
# dfTrain.Text = [re.sub('\d', '%d', sentence) for sentence in dfTrain.Text]
# dfTest.Text = [re.sub('\d', '%d', sentence) for sentence in dfTest.Text]

# stemming
# ps = PorterStemmer()
#
# sent_list = []
#
# for sentence in dfTrain.Text:
# 	tokens = tb_word_tokenizer.tokenize(sentence)
# 	stemmed_sentence = ""
#
# 	for token in tokens:
# 		stemmed_token = ps.stem(token)
# 		stemmed_sentence += stemmed_token + " "
#
# 	sent_list.append(stemmed_sentence)

# dfTrain.Text = sent_list