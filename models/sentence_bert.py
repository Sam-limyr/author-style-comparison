from sentence_transformers import SentenceTransformer, util
import numpy as np
from tqdm import tqdm
import pickle
import os


# Corpus encoding parameters
GENERATE_NEW_ENCODINGS = True  # True if you want to overwrite existing encodings; False if you want to use them
PICKLED_ENCODINGS_ADDRESS = "corpus_token_encodings"
NUMBER_OF_BYTES = -1  # Leave as -1 to read all lines; otherwise read this number of bytes per file

# Names of authors
CHARLES_DICKENS_NAME = "charles_dickens"
FYODOR_DOSTOEVSKY_NAME = "fyodor_dostoevsky"
LEO_TOLSTOY_NAME = "leo_tolstoy"
MARK_TWAIN_NAME = "mark_twain"

author_names = [CHARLES_DICKENS_NAME, FYODOR_DOSTOEVSKY_NAME, LEO_TOLSTOY_NAME, MARK_TWAIN_NAME]

corpus = {}
corpus_embeddings = {}
average_cosine_scores = {}


# Initialize model

print("Initializing model...")
model = SentenceTransformer('paraphrase-distilroberta-base-v1')
print("Model initialized.")

queries = [
    "He had become so completely absorbed in himself, and isolated from his fellows that he dreaded meeting.",
    "He was crushed by poverty, but the anxieties of his position had of late ceased to weigh upon him.",
    "He had given up attending to matters of practical importance; he had lost all desire to do so.",
    "Nothing that any landlady could do had a real terror for him."
]


if GENERATE_NEW_ENCODINGS:

    # Read data from csv files
    for author in tqdm(author_names, desc="Reading data from csv files..."):
        with open("tokens/" + author + '.csv', encoding='utf-8') as file:
            corpus[author] = file.readlines(NUMBER_OF_BYTES)

    # Encode all sentences
    for author, sentences in tqdm(corpus.items(), desc="Encoding sentence embeddings..."):
        corpus_embeddings[author] = model.encode(sentences, show_progress_bar=True)

    # Save embeddings to file via pickle module
    with open(PICKLED_ENCODINGS_ADDRESS, 'wb') as encodings_file:
        pickle.dump(corpus_embeddings, encodings_file)

else:

    if not os.path.isfile(os.path.join(os.getcwd(), PICKLED_ENCODINGS_ADDRESS)):
        raise Exception("Pickled corpus encodings do not exist at specified address.")

    print("Retrieving saved corpus embeddings...")
    with open(PICKLED_ENCODINGS_ADDRESS, 'rb') as encodings_file:
        corpus_embeddings = pickle.load(encodings_file)


query_embeddings = model.encode(queries, convert_to_tensor=True)


# Compute cosine-similarities for each corpus sentence with each query sentence
# Calculate the average cosine similarity score across all query sentences for each author

for author, sentences in tqdm(corpus_embeddings.items(), desc="Calculating average similarity scores..."):
    average_cosine_scores[author] = np.average(util.pytorch_cos_sim(sentences, query_embeddings))

for author in sorted(average_cosine_scores.keys(), key=average_cosine_scores.get, reverse=True):
    print("Author: {} \t\t | \t\t Average Similarity Score: {}"
          .format(author, round(float(average_cosine_scores[author]), 5)))



# TODO: Explore util.paraphrase_mining().
#       Problem: It only calculates a corpus's scores with itself.

