


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
import seaborn as sns
from sklearn.metrics import f1_score
from test_cases import CHARLES_DICKENS_TESTS, FYODOR_DOSTOEVSKY_TESTS, MARK_TWAIN_TESTS, JANE_AUSTEN_TESTS, JOHN_STEINBECK_TESTS


# Names of authors
CHARLES_DICKENS_NAME = "charles_dickens"
FYODOR_DOSTOEVSKY_NAME = "fyodor_dostoevsky"
LEO_TOLSTOY_NAME = "leo_tolstoy"
MARK_TWAIN_NAME = "mark_twain"
JANE_AUSTEN_NAME = "jane_austen"
JOHN_STEINBECK_NAME = "john_steinbeck"
ALL_AUTHOR_NAMES = [CHARLES_DICKENS_NAME,
                    FYODOR_DOSTOEVSKY_NAME,
                    MARK_TWAIN_NAME]
                    # JANE_AUSTEN_NAME,
                    # JOHN_STEINBECK_NAME]

# Author indices
CHARLES_DICKENS_INDEX = 0
FYODOR_DOSTOEVSKY_INDEX = 1
LEO_TOLSTOY_INDEX = 2
MARK_TWAIN_INDEX = 3
JANE_AUSTEN_INDEX = 4
JOHN_STEINBECK_INDEX = 5

# Mappings of author names to IDs
AUTHOR_NAME_TO_ID_MAPPINGS = {
    CHARLES_DICKENS_NAME: CHARLES_DICKENS_INDEX,
    FYODOR_DOSTOEVSKY_NAME: FYODOR_DOSTOEVSKY_INDEX,
    LEO_TOLSTOY_NAME: LEO_TOLSTOY_INDEX,
    MARK_TWAIN_NAME: MARK_TWAIN_INDEX,
    JANE_AUSTEN_NAME: JANE_AUSTEN_INDEX,
    JOHN_STEINBECK_NAME: JOHN_STEINBECK_INDEX
}

AUTHOR_ID_TO_NAME_MAPPINGS = {
    CHARLES_DICKENS_INDEX: CHARLES_DICKENS_NAME,
    FYODOR_DOSTOEVSKY_INDEX: FYODOR_DOSTOEVSKY_NAME,
    LEO_TOLSTOY_INDEX: LEO_TOLSTOY_NAME,
    MARK_TWAIN_INDEX: MARK_TWAIN_NAME,
    JANE_AUSTEN_INDEX: JANE_AUSTEN_NAME,
    JOHN_STEINBECK_INDEX: JOHN_STEINBECK_NAME
}


# Methods for testing

def get_all_tests():
    print("Getting tests...")
    all_tests = CHARLES_DICKENS_TESTS + FYODOR_DOSTOEVSKY_TESTS + MARK_TWAIN_TESTS + JANE_AUSTEN_TESTS + JOHN_STEINBECK_TESTS
    return all_tests


def check_test_results(results_list_and_confidence_scores, show_details=True, show_matrix=True):
    """
    Accepts a result list and checks it against the correct answers. Provides a confusion matrix, as well as basic
    scoring. The input results list should be a list of tuples (predicted_answer, confidence_score). The
    predicted_answer will be used for grading, and the confidence_score will be used for tiebreaking.
    :param results_list_and_confidence_scores: A list of tuples (predicted_answer, confidence_score) of type (str, int)
    :param show_details: A boolean to determine whether basic score totals will be printed out.
    :param show_matrix: A boolean to determine whether the confidence matrix will be displayed.
    """
    correct_answers = [CHARLES_DICKENS_NAME for _ in range(len(CHARLES_DICKENS_TESTS))] + \
                      [FYODOR_DOSTOEVSKY_NAME for _ in range(len(FYODOR_DOSTOEVSKY_TESTS))] + \
                      [MARK_TWAIN_NAME for _ in range(len(MARK_TWAIN_TESTS))] + \
                      [JANE_AUSTEN_NAME for _ in range(len(JANE_AUSTEN_TESTS))] + \
                      [JOHN_STEINBECK_NAME for _ in range(len(JOHN_STEINBECK_TESTS))]

    results_list = [result[0] for result in results_list_and_confidence_scores]
    confidence_scores = [result[1] for result in results_list_and_confidence_scores]

    assert len(results_list) == len(correct_answers), "Input and expected lists do not have the same length!"

    print("\nChecking test results...")

    if show_details:
        print_results(results_list, correct_answers)

    if show_matrix:
        show_confusion_matrix(results_list, correct_answers)


def show_confusion_matrix(predicted_results, ground_truth):
    x_axis = pd.Series(predicted_results)
    y_axis = pd.Series(ground_truth)
    confusion_matrix_data = pd.crosstab(y_axis, x_axis)

    plot = plt.subplot()
    sns.heatmap(confusion_matrix_data, annot=True, fmt='g', ax=plot)

    plot.set_xlabel("Predicted")
    plot.set_ylabel("Ground Truth")
    plot.set_title("Confusion matrix")
    plot.xaxis.set_ticklabels(confusion_matrix_data.columns)
    plot.yaxis.set_ticklabels(confusion_matrix_data.index)

    plt.show()


def print_results(predicted_results, ground_truth):
    scores = Counter()
    for i in range(len(predicted_results)):
        expected_answer = ground_truth[i]
        actual_answer = predicted_results[i]
        if actual_answer == expected_answer:
            scores[expected_answer] += 1
        # else:
        #     print("\nTest Case {}\nWRONG: Ground Truth <{}>, Predicted <{}>"
        #           .format(i+1, expected_answer, actual_answer))

    print("""

    Total score: {}/{}
    {} score: {}/{}
    {} score: {}/{}
    {} score: {}/{}
    {} score: {}/{}
    {} score: {}/{}

    """.format(sum(scores.values()), len(ground_truth),
               CHARLES_DICKENS_NAME, scores[CHARLES_DICKENS_NAME], len(CHARLES_DICKENS_TESTS),
               FYODOR_DOSTOEVSKY_NAME, scores[FYODOR_DOSTOEVSKY_NAME], len(FYODOR_DOSTOEVSKY_TESTS),
               MARK_TWAIN_NAME, scores[MARK_TWAIN_NAME], len(MARK_TWAIN_TESTS),
               JANE_AUSTEN_NAME, scores[JANE_AUSTEN_NAME], len(JANE_AUSTEN_TESTS),
               JOHN_STEINBECK_NAME, scores[JOHN_STEINBECK_NAME], len(JOHN_STEINBECK_TESTS)))

    score = f1_score(ground_truth, predicted_results, average='micro')
    print("F1 score: {}".format(round(float(score), 4)))


