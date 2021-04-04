


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


def check_test_results(results_list):
    correct_answers = [CHARLES_DICKENS_NAME for _ in CHARLES_DICKENS_TESTS] + \
                      [FYODOR_DOSTOEVSKY_NAME for _ in FYODOR_DOSTOEVSKY_TESTS] + \
                      [LEO_TOLSTOY_NAME for _ in LEO_TOLSTOY_TESTS] + \
                      [MARK_TWAIN_NAME for _ in MARK_TWAIN_TESTS]

    assert len(results_list) == len(correct_answers), "Input and expected lists do not have the same length!"

    print("Checking test results...")
    score = 0
    for i in range(len(results_list)):
        print("\nTest Case {}".format(i+1))
        expected_answer = correct_answers[i]
        actual_answer = results_list[i]
        if actual_answer != expected_answer:
            print("WRONG: Expected <{}>, Actual <{}>".format(expected_answer, actual_answer))
        else:
            print("CORRECT")
            score += 1

    print("\n\nTotal score: {}/{}".format(score, len(correct_answers)))


