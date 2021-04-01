# Code based off of :https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html

import gensim
import smart_open
import csv
import os

# Names of authors
CHARLES_DICKENS_NAME = "charles_dickens"
FYODOR_DOSTOEVSKY_NAME = "fyodor_dostoevsky"
LEO_TOLSTOY_NAME = "leo_tolstoy"
MARK_TWAIN_NAME = "mark_twain"

author_names = [CHARLES_DICKENS_NAME, FYODOR_DOSTOEVSKY_NAME, LEO_TOLSTOY_NAME, MARK_TWAIN_NAME]
training_files = ['paragraphs/' + author + '.csv' for author in author_names]

# Set file address for test data
TEST_DIRECTORY = 'test'
test_files = os.listdir(os.path.join(os.getcwd(), TEST_DIRECTORY))
test_files = [os.path.join(TEST_DIRECTORY, test_file) for test_file in test_files]
EXPECTED_DOC_IDS = ["0: Charles Dickens",
                    "1: Fyodor Dostoevsky",
                    "2: Leo Tolstoy",
                    "3: Mark Twain",
                    "Non-matching"]

# Training hyperparameters
EPOCHS = 50
VECTOR_SIZE = 200
MIN_WORD_FREQ = 1  # minimum frequency of words for them to be considered; set to 1 for default

# Diagnostic parameters
TEXT_HEAD_LEN = 10


def read_corpus(fname, index_tag=None):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        reader = csv.reader(f)
        tokens = []
        for row in reader:
            for line in row:
                tokens += gensim.utils.simple_preprocess(line)
        if index_tag is None:
            return tokens
        else:
            return gensim.models.doc2vec.TaggedDocument(tokens, [index_tag])


print("Reading data from corpus...")
train_corpus = [read_corpus(training_file, index) for index, training_file in enumerate(training_files)]
test_corpus = [read_corpus(test_file) for test_file in test_files]

print("Training model...")
model = gensim.models.doc2vec.Doc2Vec(vector_size=VECTOR_SIZE, epochs=EPOCHS, min_count=MIN_WORD_FREQ)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

print("Running sanity checks on training data...")
for train_doc_id, _ in enumerate(train_corpus):
    inferred_train_vector = model.infer_vector(train_corpus[train_doc_id].words)
    sims = model.dv.most_similar([inferred_train_vector], topn=len(model.dv))

    # Ensure that the closest match for a training vector is with itself
    assert train_doc_id == sims[0][0], "Sanity check failed for document with ID {}.".format(train_doc_id)

print("Testing model...")
for test_doc_id, _ in enumerate(test_corpus):
    inferred_test_vector = model.infer_vector(test_corpus[test_doc_id])
    sims = model.dv.most_similar([inferred_test_vector], topn=len(model.dv))

    # Compare and print the scores of all documents against the training corpus
    print('\nTesting Document ({}): «{} ...»'.format(test_doc_id, ' '.join(test_corpus[test_doc_id][:TEXT_HEAD_LEN])))
    print('EXPECTED: {}'.format(EXPECTED_DOC_IDS[test_doc_id]))
    for train_doc_id, score in sims:
        print('DOC_ID {}, SCORE {}: «{} ...»'.format(train_doc_id, score,
                                                     ' '.join(train_corpus[train_doc_id].words[:TEXT_HEAD_LEN])))
