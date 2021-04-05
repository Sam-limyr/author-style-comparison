


"""

    Important method imports:
        - get_all_tests():
                Returns a list of sample test queries.
        - check_test_results(your_answer_list):
                Accepts a list of string answers. Each answer is expected to be one of the NAME globals defined below.

    Important constant imports:
        - CHARLES_DICKENS_NAME and other NAME constants:
                Standardized names for the four authors in the training corpus.
        - CHARLES_DICKENS_INDEX and other INDEX constants:
                Standardized indices for the four authors.
        - AUTHOR_NAME_TO_ID_MAPPINGS and AUTHOR_ID_TO_NAME_MAPPINGS:
                Standardized mappings between the author names and indices, if necessary.


    Example use case:

            from test_runner import *

            test_cases = get_all_tests()
            output_answers = []

            for test in test_cases:

                ## obtain answer from your model here.
                ## use the name/id constants defined if necessary.

                output_answers.append(answer)

            check_test_results(output_answers)

"""


from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from test_cases import CHARLES_DICKENS_TESTS, FYODOR_DOSTOEVSKY_TESTS, LEO_TOLSTOY_TESTS, MARK_TWAIN_TESTS


# Names of authors
CHARLES_DICKENS_NAME = "charles_dickens"
FYODOR_DOSTOEVSKY_NAME = "fyodor_dostoevsky"
LEO_TOLSTOY_NAME = "leo_tolstoy"
MARK_TWAIN_NAME = "mark_twain"
ALL_AUTHOR_NAMES = [CHARLES_DICKENS_NAME, FYODOR_DOSTOEVSKY_NAME, LEO_TOLSTOY_NAME, MARK_TWAIN_NAME]

# Author indices
CHARLES_DICKENS_INDEX = 0
FYODOR_DOSTOEVSKY_INDEX = 1
LEO_TOLSTOY_INDEX = 2
MARK_TWAIN_INDEX = 3

# Mappings of author names to IDs
AUTHOR_NAME_TO_ID_MAPPINGS = {
    CHARLES_DICKENS_NAME: CHARLES_DICKENS_INDEX,
    FYODOR_DOSTOEVSKY_NAME: FYODOR_DOSTOEVSKY_INDEX,
    LEO_TOLSTOY_NAME: LEO_TOLSTOY_INDEX,
    MARK_TWAIN_NAME: MARK_TWAIN_INDEX
}

AUTHOR_ID_TO_NAME_MAPPINGS = {
    CHARLES_DICKENS_INDEX: CHARLES_DICKENS_NAME,
    FYODOR_DOSTOEVSKY_INDEX: FYODOR_DOSTOEVSKY_NAME,
    LEO_TOLSTOY_INDEX: LEO_TOLSTOY_NAME,
    MARK_TWAIN_INDEX: MARK_TWAIN_NAME
}


# Methods for testing

def get_all_tests():
    all_tests = CHARLES_DICKENS_TESTS + FYODOR_DOSTOEVSKY_TESTS + LEO_TOLSTOY_TESTS + MARK_TWAIN_TESTS
    return all_tests


def check_test_results(results_list, show_details=True, show_matrix=True):
    correct_answers = [CHARLES_DICKENS_NAME for _ in CHARLES_DICKENS_TESTS] + \
                      [FYODOR_DOSTOEVSKY_NAME for _ in FYODOR_DOSTOEVSKY_TESTS] + \
                      [LEO_TOLSTOY_NAME for _ in LEO_TOLSTOY_TESTS] + \
                      [MARK_TWAIN_NAME for _ in MARK_TWAIN_TESTS]

    assert len(results_list) == len(correct_answers), "Input and expected lists do not have the same length!"

    print("Checking test results...")

    if show_details:
        print_results(results_list, correct_answers)

    if show_matrix:
        show_confusion_matrix(results_list, correct_answers)


def show_confusion_matrix(predicted_results, ground_truth):
    y_actu = pd.Series(ground_truth, name='Ground Truth')
    y_pred = pd.Series(predicted_results, name='Predicted')
    dataframe_confusion = pd.crosstab(y_actu, y_pred, rownames=['Ground Truth'], colnames=['Predicted'])

    plt.imshow(dataframe_confusion)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(dataframe_confusion.columns))
    plt.xticks(tick_marks, dataframe_confusion.columns)
    plt.yticks(tick_marks, dataframe_confusion.index)
    plt.tight_layout()
    plt.ylabel(dataframe_confusion.index.name)
    plt.xlabel(dataframe_confusion.columns.name)
    plt.show()


def print_results(predicted_results, ground_truth):
    scores = Counter()
    for i in range(len(predicted_results)):
        expected_answer = ground_truth[i]
        actual_answer = predicted_results[i]
        if actual_answer != expected_answer:
            print("\nTest Case {}\nWRONG: Ground Truth <{}>, Predicted <{}>"
                  .format(i+1, expected_answer, actual_answer))
        else:
            scores[expected_answer] += 1

    print("""

    Total score: {}/{}
    {} score: {}/{}
    {} score: {}/{}
    {} score: {}/{}
    {} score: {}/{}

    """.format(sum(scores.values()), len(ground_truth),
          CHARLES_DICKENS_NAME, scores[CHARLES_DICKENS_NAME], len(CHARLES_DICKENS_TESTS),
          FYODOR_DOSTOEVSKY_NAME, scores[FYODOR_DOSTOEVSKY_NAME], len(FYODOR_DOSTOEVSKY_TESTS),
          LEO_TOLSTOY_NAME, scores[LEO_TOLSTOY_NAME], len(LEO_TOLSTOY_TESTS),
          MARK_TWAIN_NAME, scores[MARK_TWAIN_NAME], len(MARK_TWAIN_TESTS)))

