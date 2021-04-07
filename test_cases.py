import os


def get_tests_from_book(filename, number_of_tests, minimum_test_length):
    """
    Parses tests from a given text. This can be used to scale up the number of test cases easily.
    :param filename: The filename containing the file to read.
    :param number_of_tests: The number of test cases to obtain.
    :param minimum_test_length: The minimum length of each test case, in characters.
    :return: Returns all requested test cases, in a list.
    """

    """
    Open file object.
    Count number of lines.
    Partition based on the number of tests.
    Open the file object again.
    At each test case line, keep adding lines until the minimum test length has been reached.
    """

    with open(filename, 'r', encoding='utf-8') as file:
        all_file_lines = list(file.readlines())
        number_of_lines_in_file = len(all_file_lines)
        partition_size = int(number_of_lines_in_file/number_of_tests)

        all_test_cases = []
        test_case_starting_line_numbers = [partition_size * partition_number
                                           for partition_number in range(number_of_tests)]

        for starting_line_number in test_case_starting_line_numbers:
            test_case_text = ""
            for line in all_file_lines[starting_line_number: starting_line_number + partition_size]:
                if len(test_case_text) >= minimum_test_length:
                    break
                test_case_text += line
            all_test_cases.append(test_case_text)

    for test_case in all_test_cases:
        assert len(test_case) >= minimum_test_length, "A test case is shorter than expected: {}".format(test_case)

    return all_test_cases

# File addresses for test novels
ROOT = "supplementaryNovels"
CHARLES_DICKENS_FOLDER = "charles_dickens"
FYODOR_DOSTOEVSKY_FOLDER = "fyodor_dostoevsky"
MARK_TWAIN_FOLDER = "mark_twain"

CHARLES_DICKENS_TEST_FILES = [os.path.join(ROOT, CHARLES_DICKENS_FOLDER, filename)
                              for filename in os.listdir(os.path.join(ROOT, CHARLES_DICKENS_FOLDER))]
FYODOR_DOSTOEVSKY_TEST_FILES = [os.path.join(ROOT, FYODOR_DOSTOEVSKY_FOLDER, filename)
                                for filename in os.listdir(os.path.join(ROOT, FYODOR_DOSTOEVSKY_FOLDER))]
MARK_TWAIN_TEST_FILES = [os.path.join(ROOT, MARK_TWAIN_FOLDER, filename)
                         for filename in os.listdir(os.path.join(ROOT, MARK_TWAIN_FOLDER))]


# Combined tests

CHARLES_DICKENS_TESTS = get_tests_from_book('supplementaryNovels/charles_dickens/barnabyRudge.txt', 25, 1000)
# CHARLES_DICKENS_TESTS = CHARLES_DICKENS_DIFFERENT_BOOK_MULTIPLE_PARAGRAPH_TESTS
                        # CHARLES_DICKENS_SAME_BOOK_SINGLE_SENTENCE_TESTS + \
                        # CHARLES_DICKENS_SAME_BOOK_MULTIPLE_PARAGRAPH_TESTS


FYODOR_DOSTOEVSKY_TESTS = []
# FYODOR_DOSTOEVSKY_TESTS = FYODOR_DOSTOEVSKY_DIFFERENT_BOOK_MULTIPLE_PARAGRAPH_TESTS
                          # FYODOR_DOSTOEVSKY_SAME_BOOK_SINGLE_SENTENCE_TESTS + \
                          # FYODOR_DOSTOEVSKY_SAME_BOOK_SINGLE_PARAGRAPH_TESTS + \
                          # FYODOR_DOSTOEVSKY_SAME_BOOK_MULTIPLE_PARAGRAPH_TESTS

# LEO_TOLSTOY_TESTS = LEO_TOLSTOY_SAME_BOOK_SINGLE_SENTENCE_TESTS + \
#                     LEO_TOLSTOY_SAME_BOOK_MULTIPLE_PARAGRAPH_TESTS
LEO_TOLSTOY_TESTS = []

MARK_TWAIN_TESTS = []
# MARK_TWAIN_TESTS = MARK_TWAIN_DIFFERENT_BOOK_MULTIPLE_PARAGRAPH_TESTS
                   # MARK_TWAIN_SAME_BOOK_SINGLE_SENTENCE_TESTS + \
                   # MARK_TWAIN_SAME_BOOK_SINGLE_PARAGRAPH_TESTS + \
                   # MARK_TWAIN_SAME_BOOK_MULTIPLE_PARAGRAPH_TESTS

