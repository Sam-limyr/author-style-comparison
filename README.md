# cs4248-style-comparison
A Natural Language Processing project that aims to isolate and compare the writing styles of authors.

## Project Purpose

This project uses several models to perform stylometric analysis on Victorian-era authors.
The end goal of the project is to understand whether style can be used to:
- Identify an author when presented with test excerpts,
- Identify similarities between different authors, given test excerpts from an unknown author.

A more detailed analysis can be found in the file `Project Report.pdf`, which contains a report
on the data and findings of this project.

## Project Contents

### Entrypoint

The intended entrypoint of this project is `ensemble.py`. Different combinations of the three models can be run by
commenting out the respective sections in the program.

### `models` directory

Contains 5 machine learning source code files:
- `doc2vec.py`: The implementation of the Doc2vec model.
- `feature_engineering.py`: An implementation of Feature Engineering using Logistic Regression.
- `knn.py`: An implementation of function word rank vector analysis using k-Nearest Neighbors.
- `naive_bayes.py`: A baseline tf-idf implementation using Naive Bayes.
- `sentence_bert.py`: An outdated model that seeks to use sentenceBERT.

### `utils` directory

Contains utility programs used to run tests and handle text data:
- `test_runner.py`: Responsible for handling and grading test cases.
- `test_cases.py`: Responsible for extracting test cases from text files.
- `tokenizer.py`: Used to tokenize text from training data.
- `book_splitter.py`: Used to split training data into segments, to analyze the effect of training sample count
on model performance.
- `word_counter.py`: Used to count words in training data.

### `data` directory

Contains training and test data:
- `docs`: Contains further information about the data used.
- `train`: Contains different forms of training data.
- `test`: Contains different forms of testing data.
- `results`: Contains confusion matrices of models derived from test data.
