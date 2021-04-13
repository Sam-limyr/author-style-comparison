# Code based off of :https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html

import gensim
import smart_open
import csv
from test_runner import get_all_tests, check_test_results, ALL_AUTHOR_NAMES, AUTHOR_ID_TO_NAME_MAPPINGS, \
    AUTHOR_NAME_TO_ID_MAPPINGS


training_files = [('paragraphs/' + author + '.csv', author) for author in ALL_AUTHOR_NAMES]

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


def predict_doc2vec_results():
    print("Running doc2vec model...")
    print("Reading data from corpus...")
    train_corpus = [read_corpus(training_file, AUTHOR_NAME_TO_ID_MAPPINGS[author_name])
                    for training_file, author_name in training_files]

    print("Training model...")
    model = gensim.models.doc2vec.Doc2Vec(vector_size=VECTOR_SIZE, epochs=EPOCHS, min_count=MIN_WORD_FREQ, seed=0)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    print("Running sanity checks on training data...")
    for train_doc_id, tagged_document in enumerate(train_corpus):
        inferred_train_vector = model.infer_vector(train_corpus[train_doc_id].words)
        sims = model.dv.most_similar([inferred_train_vector], topn=len(model.dv))

        # Ensure that the closest match for a training vector is with itself
        author_id = tagged_document.tags[0]
        assert author_id == sims[0][0], "Sanity check failed for document with ID {}.\nExpected {}, Received {}"\
            .format(train_doc_id, author_id, sims[0][0])

    print("Testing model...")
    test_corpus = get_all_tests()
    test_corpus = [process_test_case(test_case) for test_case in test_corpus]
    output_answers = []
    for test_doc_id in range(len(test_corpus)):
        inferred_test_vector = model.infer_vector(test_corpus[test_doc_id])
        sims = model.dv.most_similar([inferred_test_vector], topn=len(model.dv))

        answer = sims[0][0]

        # The confidence score is defined as the difference between the similarity scores of the first and second choices.
        #       This is defined semi-arbitrarily, but it allows comparison between models.
        confidence_score = sims[0][1] - sims[1][1]

        output_answers.append((AUTHOR_ID_TO_NAME_MAPPINGS[answer], confidence_score))

    return output_answers
