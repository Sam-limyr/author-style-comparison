### Utility to approximately count the number of words in a text file.

import re
import os
import sys

def main():
    textFile = "notesFromUnderground.txt"
    filename = os.path.join(sys.path[0], textFile)
    with open(filename, 'r', errors='ignore') as file:
        text = file.read()
        word_arr = re.split(r"\s+", text)
        print("Approx. number of words in {}: {}".format(textFile, len(word_arr)))

main()