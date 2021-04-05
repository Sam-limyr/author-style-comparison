# Code based off of :https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html

import gensim
import smart_open
import csv
from test_runner import get_all_tests, check_test_results, ALL_AUTHOR_NAMES, AUTHOR_ID_TO_NAME_MAPPINGS


training_files = ['paragraphs/' + author + '.csv' for author in ALL_AUTHOR_NAMES]

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


def process_test_case(test_case_string):
    return gensim.utils.simple_preprocess(test_case_string)


print("Reading data from corpus...")
train_corpus = [read_corpus(training_file, index) for index, training_file in enumerate(training_files)]

print("Training model...")
model = gensim.models.doc2vec.Doc2Vec(vector_size=VECTOR_SIZE, epochs=EPOCHS, min_count=MIN_WORD_FREQ, seed=0)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

print("Running sanity checks on training data...")
for train_doc_id, _ in enumerate(train_corpus):
    inferred_train_vector = model.infer_vector(train_corpus[train_doc_id].words)
    sims = model.dv.most_similar([inferred_train_vector], topn=len(model.dv))

    # Ensure that the closest match for a training vector is with itself
    assert train_doc_id == sims[0][0], "Sanity check failed for document with ID {}.".format(train_doc_id)

print("Testing model...")
test_corpus = get_all_tests()
test_corpus = [process_test_case(test_case) for test_case in test_corpus]
output_answers = []
for test_doc_id in range(len(test_corpus)):
    inferred_test_vector = model.infer_vector(test_corpus[test_doc_id])
    sims = model.dv.most_similar([inferred_test_vector], topn=len(model.dv))

    answer = sims[0][0]
    output_answers.append(AUTHOR_ID_TO_NAME_MAPPINGS[answer])

check_test_results(output_answers)
