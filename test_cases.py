import os
import math


def get_tests_from_folder(folder, minimum_total_number_of_tests, minimum_test_length):
    number_of_books = len(os.listdir(folder))
    number_of_tests_per_book = math.ceil(minimum_total_number_of_tests/number_of_books)

    all_folder_tests = []
    for filename in os.listdir(folder):
        all_book_tests = get_tests_from_book(os.path.join(folder, filename), number_of_tests_per_book,
                                             minimum_test_length)
        all_folder_tests += all_book_tests

    return all_folder_tests


def get_tests_from_book(filename, number_of_tests, minimum_test_length):
    """
    Parses tests from a given text. This can be used to scale up the number of test cases easily.
    :param filename: The filename containing the file to read.
    :param number_of_tests: The number of test cases to obtain.
    :param minimum_test_length: The minimum length of each test case, in characters.
    :return: Returns all requested test cases, in a list.
    """

    with open(filename, 'r', encoding='utf-8') as file:
        all_file_lines = list(file.readlines())
        number_of_lines_in_file = len(all_file_lines)
        partition_size = math.floor(number_of_lines_in_file/number_of_tests)

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
        assert len(test_case) >= minimum_test_length, "\n\nA test case is shorter than expected. " \
                                                      "This is most likely because there is insufficient content " \
                                                      "in one of the test novels: {}".format(test_case)

    return all_test_cases


# File addresses for test novels
ROOT = "supplementaryNovels"
CHARLES_DICKENS_FOLDER = os.path.join(ROOT, "charles_dickens")
FYODOR_DOSTOEVSKY_FOLDER = os.path.join(ROOT, "fyodor_dostoevsky")
MARK_TWAIN_FOLDER = os.path.join(ROOT, "mark_twain")


# Combined tests

CHARLES_DICKENS_TESTS = get_tests_from_folder(CHARLES_DICKENS_FOLDER, 150, 1000)

FYODOR_DOSTOEVSKY_TESTS = get_tests_from_folder(FYODOR_DOSTOEVSKY_FOLDER, 150, 1000)

LEO_TOLSTOY_TESTS = []

MARK_TWAIN_TESTS = get_tests_from_folder(MARK_TWAIN_FOLDER, 150, 1000)

