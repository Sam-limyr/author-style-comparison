import pandas as pd
import re
import os
import sys

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords   # Requires NLTK in the include path.

## List of stopwords
STOPWORDS = stopwords.words('english') # type: list(str)

authors = {"Charles Dickens": ["davidc.txt"], "Fyodor Dostoevsky": ["crimep.txt"], 
           "Leo Tolstoy": ["warap.txt"], "Mark Twain": ["toms.txt"]}

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
        # store stats in outer dictionary
        stats[author] = {"Sentence count": sentence_count, "Word count": total_words, "Stopword count": stopword_count, "Avg. word per sentence": avg_word_per_sent, "Words in vocab": unique_word_count}
        
        print("Number of sentences: {} \nNumber of total words: {} \nNumber of stopwords: {} \nAvg. words per sentence: {} \nWords in vocab: {}".format(sentence_count, total_words, stopword_count, avg_word_per_sent, unique_word_count))
        
print(stats)