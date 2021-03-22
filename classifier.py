# Code credit: https://www.sbert.net/docs/usage/semantic_textual_similarity.html

from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('paraphrase-distilroberta-base-v1')

CHARLES_DICKENS_NAME = "charles_dickens"
FYODOR_DOSTOEVSKY_NAME = "fyodor_dostoevsky"
LEO_TOLSTOY_NAME = "leo_tolstoy"
MARK_TWAIN_NAME = "mark_twain"

corpus = {
    CHARLES_DICKENS_NAME: [
        'I had a chicken',
        'potato'
    ],
    FYODOR_DOSTOEVSKY_NAME: [
        'magician'
    ],
    LEO_TOLSTOY_NAME: [
        'hey there',
        'this is a test',
        'I like animals'
    ],
    MARK_TWAIN_NAME: [
        'horsing around',
        'music is the best'
    ]
}

corpus_embeddings = {
    CHARLES_DICKENS_NAME: None,
    FYODOR_DOSTOEVSKY_NAME: None,
    LEO_TOLSTOY_NAME: None,
    MARK_TWAIN_NAME: None
}

queries = ['Here is an equine creature',
           'The person is doing the thing',
           'A person is using an instrument.',
           'A female is using an instrument.'
           ]

# Encode all sentences
for author, sentences in corpus.items():
    corpus_embeddings[author] = model.encode(sentences, convert_to_numpy=True)
query_embeddings = model.encode(queries, convert_to_numpy=True)


# Compute cosine-similarities for each corpus sentence with each query sentence
# Calculate the average cosine similarity score across all query sentences for each author

average_cosine_scores = {
    CHARLES_DICKENS_NAME: None,
    FYODOR_DOSTOEVSKY_NAME: None,
    LEO_TOLSTOY_NAME: None,
    MARK_TWAIN_NAME: None
}

for author, sentences in corpus_embeddings.items():
    average_cosine_scores[author] = np.average(util.pytorch_cos_sim(sentences, query_embeddings))

for author in sorted(average_cosine_scores.keys(), key=average_cosine_scores.get, reverse=True):
    print("Author: {} \t\t | \t\t Average Similarity Score: {}".format(author, average_cosine_scores[author]))



# TODO: Explore util.paraphrase_mining().
#       Problem: It only calculates a corpus's scores with itself.

