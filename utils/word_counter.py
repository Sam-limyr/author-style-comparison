### Utility to approximately count the number of words in a text file.

import re
import os
import sys

# name of author and text file to count
author = "mark_twain"
textFile = "princepauper.txt"

def main():
    fileToRead = "data/train/novels/{}/{}".format(author, textFile)
    with open(fileToRead, 'r', errors='ignore') as file:
        text = file.read()
        word_arr = re.split(r"\s+", text)
        print("Approx. number of words in {}: {}".format(textFile, len(word_arr)))

main()