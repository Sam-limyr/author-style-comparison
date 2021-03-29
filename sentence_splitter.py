import pandas as pd
import re
import os
import sys

# Statistics to track
avg_words_per_sent = 0
stopword_count = 0
total_words = 0
unique_vocab_count = 0


filepath = os.path.join(sys.path[0], "davidc.txt")

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
        sentences += [ele.strip()]
    elif ele[0] in "?!.":
        # count statistics
        sentences[-1] += ele.strip()
    else:
        # count statistics
        sentences += [ele.strip()]

print(sentences[0:20])