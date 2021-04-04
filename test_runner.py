


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


# Tests, from the respective authors

CHARLES_DICKENS_TESTS = [
    """‘When he
    bought the house, he liked to think that there were rooks about it.’
    
    The evening wind made such a disturbance just now, among some tall old
    elm-trees at the bottom of the garden, that neither my mother nor Miss
    Betsey could forbear glancing that way."
    "As the elms bent to one another,
    like giants who were whispering secrets, and after a few seconds of such
    repose, fell into a violent flurry, tossing their wild arms about, as if
    their late confidences were really too wicked for their peace of mind,
    some weatherbeaten ragged old rooks’-nests, burdening their higher
    branches, swung like wrecks upon a stormy sea."
    ‘Where are the birds?’ asked Miss Betsey.
    ‘The--?’ My mother had been thinking of something else.
    ‘The rooks--what has become of them?’ asked Miss Betsey.
    "‘There have not been any since we have lived here,’ said my mother."""
]

FYODOR_DOSTOEVSKY_TESTS = [
    """But in spite of this scornful reflection, he was by now looking cheerful
    as though he were suddenly set free from a terrible burden: and he gazed
    round in a friendly way at the people in the room. But even at that
    moment he had a dim foreboding that this happier frame of mind was also
    not normal."""
]

LEO_TOLSTOY_TESTS = [
    """She raised herself on the sofa on which she had been
    lying and replied through the closed door that she did not mean to go
    away and begged to be left in peace."
    "The windows of the room in which she was lying looked westward. She
    lay on the sofa with her face to the wall, fingering the buttons of the
    leather cushion and seeing nothing but that cushion, and her confused
    thoughts were centered on one subject—the irrevocability of death and
    her own spiritual baseness, which she had not suspected, but which had
    shown itself during her father’s illness. She wished to pray but did not
    dare to, dared not in her present state of mind address herself to God.
    She lay for a long time in that position."
    "The sun had reached the other side of the house, and its slanting rays
    shone into the open window, lighting up the room and part of the morocco
    cushion at which the girl was looking. The flow of her thoughts
    suddenly stopped. Unconsciously she sat up, smoothed her hair, got up,
    and went to the window, involuntarily inhaling the freshness of the
    clear but windy evening."""
]

MARK_TWAIN_TESTS = [
    """When we saw him last, royalty was just beginning to have a bright side
    for him.  This bright side went on brightening more and more every
    day: in a very little while it was become almost all sunshine and
    delightfulness.  He lost his fears; his misgivings faded out and died;
    his embarrassments departed, and gave place to an easy and confident
    bearing.  He worked the whipping-boy mine to ever-increasing profit."""
]


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


