from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd
import re
import os
import sys

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords   # Requires NLTK in the include path.

## List of stopwords
STOPWORDS = stopwords.words('english') # type: list(str)

from test_runner import *
from matplotlib import pyplot

from test_cases import CHARLES_DICKENS_TESTS, FYODOR_DOSTOEVSKY_TESTS, JANE_AUSTEN_TESTS, JOHN_STEINBECK_TESTS, MARK_TWAIN_TESTS

# Names of authors
CHARLES_DICKENS_NAME = "charles_dickens"
FYODOR_DOSTOEVSKY_NAME = "fyodor_dostoevsky"
JANE_AUSTEN_NAME = "jane_austen"
JOHN_STEINBECK_NAME = "john_steinbeck"
MARK_TWAIN_NAME = "mark_twain"
ALL_AUTHOR_NAMES = [CHARLES_DICKENS_NAME, FYODOR_DOSTOEVSKY_NAME, JANE_AUSTEN_NAME, JOHN_STEINBECK_NAME, MARK_TWAIN_NAME]

# Get F1 Score
correct_answers = [CHARLES_DICKENS_NAME for _ in CHARLES_DICKENS_TESTS] + \
                  [FYODOR_DOSTOEVSKY_NAME for _ in FYODOR_DOSTOEVSKY_TESTS] + \
                  [JANE_AUSTEN_NAME for _ in JANE_AUSTEN_TESTS] + \
                  [JOHN_STEINBECK_NAME for _ in JOHN_STEINBECK_TESTS] + \
                  [MARK_TWAIN_NAME for _ in MARK_TWAIN_TESTS]

authors = {"charles_dickens": ["davidc.txt", "greatex.txt", "olivert.txt", "twocities.txt"],
           "fyodor_dostoevsky": ["crimep.txt", "idiot.txt", "possessed.txt"], 
           #"jane_austen": ["emma.txt", "ladySusan.txt", "northangerAbbey.txt", "prideAndPrejudice.txt"],
           #"leo_tolstoy": ["warap.txt", "annakarenina.txt"], 
           "mark_twain": ["toms.txt", "huckfinn.txt", "connecticutyankee.txt", "princepauper.txt"]}

authorsIndexed = ["charles_dickens", "fyodor_dostoevsky", "mark_twain"]
num_novels = {}
    
def extract_features():
    print("Extracting stats from novels...")
    
    stats = {}
    
    # initialize statistics arrays
    stats["personalpronoun_tags_per_sent"] = []
    stats["determiner_tags_per_sent"] = []
    stats["adjective_tags_per_sent"] = []
    stats["adverb_tags_per_sent"] = []
    stats["verb_past_tags_per_sent"] = []
    stats["noun_tags_per_sent"] = []
    stats["interjection_tags_per_sent"] = []
    stats["modal_tags_per_sent"] = []
    stats["foreign_word_tags_per_sent"] = []
    stats["stopword_count_per_sent"] = []
    stats["avg_word_per_sentence"] = []
    stats["avg_word_length"] = []
    stats["capitalized_per_sentence"] = []
    stats["dashes_per_sent"] = []
    stats["comma_count_per_sent"] = []
    #stats["exclamation_marks_per_sent"] = []
    #stats["question_marks_per_sent"] = []
    stats["italics_per_sent"] = []
    #stats["contractions_per_sent"] = []
    stats["dialogue_per_sent"] = []
    # stats["vocab_word_count"] = []
    
    dir_name = "split_novels"
    
    for author in authors:
        # currently unused
        vocab = {}
        novel_filenames = os.listdir(dir_name + "/" + author)
        #print(novel_filenames)
        num_novels[author] = len(novel_filenames)

        for novel_filename in novel_filenames:
                                     
        #for bookName in authors[author]:
            bookFilepath = dir_name + "/" + author + "/" + novel_filename
            #print(bookFilepath)

            # read text in book
            with open(bookFilepath, encoding='utf-8', errors='ignore') as f:
                text = f.read()[1:]

            # sentence segment, split based on punctuation
            # if ! or ?, OR .\s[A-Z] OR ."\s[A-Z]
            text = text.replace("\n", " ")
            p_sentence = re.compile(r'([?!]"?)|((?<!Dr|Mr|Ms|Jr|Sr|St)(?<!Mrs|Rev)\."?\s+)')
            sentence_list = re.split(p_sentence, text)

            # to reconstruct sentences as string
            sentences = []

            # Statistics to track per book
            tags = {}
            pp_tags = 0
            dt_tags = 0
            adj_tags = 0
            adv_tags = 0
            vbd_tags = 0
            noun_tags = 0
            interjection_tags = 0
            modal_tags = 0
            foreign_word_tags = 0
            sentence_count = 0
            stopword_count = 0
            total_words = 0
            character_count = 0
            capitalized_words = 0
            # punctuation_count
            dash_count = 0 # counting "--"
            comma_count = 0
            exclamation_marks = 0 
            question_marks = 0
            italics_count = 0 # _<word>_
            contractions_count = 0 # counting "you'll" etc.
            dialogue_count = 0 # counting ""
            in_dialogue = False

            for ele in sentence_list:
                if ele == None or len(ele) == 0:
                    continue
                elif len(sentences) != 0 and ele[0] in "?!.":
                    # count statistics
                    if ele[0] == '!':
                        exclamation_marks += 1
                    elif ele[0] == '?':
                        question_marks += 1
                    sentences[-1] += ele.strip()
                else:
                    # POS tagging
                    tokens = nltk.word_tokenize(ele)
                    words_and_tags = nltk.pos_tag(tokens)
                    word_position = 0
                    for word, tag in words_and_tags:
                        if word_position not in tags:
                            tags[word_position] = {}
                        # count tags
                        if tag not in tags[word_position]:
                            tags[word_position][tag] = 1
                        else:
                            tags[word_position][tag] += 1
                        
                        # should be able to be access after processing book through tags[position]['PRP']
                        if tag == 'PRP':
                            pp_tags += 1
                        elif tag == 'DT':
                            dt_tags += 1
                        elif tag == 'JJ':
                            adj_tags += 1
                        elif tag == 'RB':
                            adv_tags += 1
                        elif tag == 'VBD':
                            vbd_tags += 1
                        elif tag[0:2] == 'NN':
                            noun_tags += 1
                        elif tag == 'UH':
                            interjection_tags += 1
                        elif tag == 'MD':
                            modal_tags += 1
                        elif tag == 'FW':
                            foreign_word_tags += 1
                        word_position += 1
                        
                    # count statistics
                    sentence_count += 1
                    words = re.findall(r'\w+’?\w*', ele)
                    # count capitalized words except start of novel and sentences 
                    # (assume sentence starts with a space after previous sentence's fullstop)
                    capitalized = re.findall(r'(?<!\.\s)(?<!\A)([A-Z]\w+’?\w*)', ele) 
                    dashes = re.findall(r'-+', ele)
                    commas = re.findall(r',', ele)
                    italics = re.findall(r'_\w+_', ele)
                    contractions = re.findall(r'\w+’\w+', ele)
                    if not in_dialogue:
                        dialogues = re.findall(r'[‘“].*', ele) # Look for start of dialogue
                        in_dialogue = len(dialogues) > 0
                    else:
                        dialogues = re.findall(r'.*[”’]', ele) # Look for end of dialogue
                        in_dialogue = len(dialogues) == 0
                    # contractions treated as one word
                    total_words += len(words)
                    no_spaces = ''.join(words)
                    character_count += len(no_spaces)
                    capitalized_words += len(capitalized)
                    dash_count += len(dashes)
                    comma_count += len(commas)
                    italics_count += len(italics)
                    contractions_count += len(contractions)
                    # Count sentences that have dialogue
                    dialogue_count += 1 if in_dialogue else 0
                    for word in words:
                        # TODO: Handle known words that are in ALL CAPS
                        # TODO: Handle named entities (people, cities?), count them as the same since not style
                        # TODO: Project Gutenberg seems to have italicised? words that are _<word>_, handle.
                        if word not in vocab:
                            vocab[word] = 1
                        else:
                            # counting word occurrence because might as well
                            vocab[word] += 1
                        if word in STOPWORDS:
                            stopword_count += 1
                    sentences += [ele.strip()]

            unique_word_count = len(vocab.keys())
            
            # store stats in outer dictionary, in order of "authors" dictionary
            stats["personalpronoun_tags_per_sent"].append(np.log(1+pp_tags/sentence_count))
            stats["determiner_tags_per_sent"].append(np.log(1+dt_tags/sentence_count))
            stats["adjective_tags_per_sent"].append(np.log(1+adj_tags/sentence_count))
            stats["adverb_tags_per_sent"].append(np.log(1+adv_tags/sentence_count))
            stats["verb_past_tags_per_sent"].append(np.log(1+vbd_tags/sentence_count))
            stats["noun_tags_per_sent"].append(np.log(1+noun_tags/sentence_count))
            stats["interjection_tags_per_sent"].append(np.log(1+interjection_tags/sentence_count))
            stats["modal_tags_per_sent"].append(np.log(1+modal_tags/sentence_count))
            stats["foreign_word_tags_per_sent"].append(np.log(1+foreign_word_tags/sentence_count))
            stats["stopword_count_per_sent"].append(np.log(1+stopword_count/sentence_count))
            stats["avg_word_per_sentence"].append(np.log(1+total_words/sentence_count))
            stats["avg_word_length"].append(np.log(1+character_count/total_words))
            stats["capitalized_per_sentence"].append(np.log(1+capitalized_words/sentence_count))
            stats["dashes_per_sent"].append(np.log(1+dash_count/sentence_count))
            stats["comma_count_per_sent"].append(np.log(1+comma_count/sentence_count))
            #stats["exclamation_marks_per_sent"].append(np.log(1+exclamation_marks/sentence_count))
            #stats["question_marks_per_sent"].append(np.log(1+question_marks/sentence_count))
            stats["italics_per_sent"].append(np.log(1+italics_count/sentence_count))
            #stats["contractions_per_sent"].append(np.log(1+contractions_count/sentence_count))
            stats["dialogue_per_sent"].append(np.log(1+dialogue_count/sentence_count))
            
            # vocab needs to be able to count rare words and identify them
            # stats["vocab_word_count"].append(unique_word_count)
            
            #print(tags)
    return stats

def train_model(model, x_train, y_train):
    ''' TODO: train your model based on the training data '''
    model.fit(x_train, y_train)
    pass

def predict(model, x_test):
    ''' TODO: make your prediction here '''
    return model.predict_proba(x_test)

# extract stats for each author
def scale_features(stats, scaler, isTestSet):
    features = pd.DataFrame(data=stats)
    
    if isTestSet:
        scaled = scaler.transform(features)
    else:
        scaler.fit(features)
        scaled = scaler.fit_transform(features)
    
    return pd.DataFrame(scaled)

# Count stats for the queries
def extract_test_features(queries, scaler):
    
    stats = {}
    
    # initialize statistics arrays
    stats["personalpronoun_tags_per_sent"] = []
    stats["determiner_tags_per_sent"] = []
    stats["adjective_tags_per_sent"] = []
    stats["adverb_tags_per_sent"] = []
    stats["verb_past_tags_per_sent"] = []
    stats["noun_tags_per_sent"] = []
    stats["interjection_tags_per_sent"] = []
    stats["modal_tags_per_sent"] = []
    stats["foreign_word_tags_per_sent"] = []
    stats["stopword_count_per_sent"] = []
    stats["avg_word_per_sentence"] = []
    stats["avg_word_length"] = []
    stats["capitalized_per_sentence"] = []
    stats["dashes_per_sent"] = []
    stats["comma_count_per_sent"] = []
    #stats["exclamation_marks_per_sent"] = []
    #stats["question_marks_per_sent"] = []
    stats["italics_per_sent"] = []
    #stats["contractions_per_sent"] = []
    stats["dialogue_per_sent"] = []
    # stats["vocab_word_count"] = []
    
    vocab = {}
    for query in queries:
        # split each sentence
        query = query.replace("\n", " ")
        p_sentence = re.compile(r'([?!]"?)|((?<!Dr|Mr|Ms|Jr|Sr|St)(?<!Mrs|Rev)\."?\s+)')
        sentence_list = re.split(p_sentence, query)
        
        pp_tags = 0
        dt_tags = 0
        adj_tags = 0
        adv_tags = 0
        vbd_tags = 0
        noun_tags = 0
        interjection_tags = 0
        modal_tags = 0
        foreign_word_tags = 0
        sentence_count = 0
        total_words = 0
        character_count = 0
        capitalized_words = 0
        stopword_count = 0
        in_dialogue = False
        sentences = []
        # punctuation_count
        dash_count = 0 # counting "--"
        comma_count = 0
        exclamation_marks = 0
        question_marks = 0
        italics_count = 0 # _<word>_
        contractions_count = 0 # counting "you'll" etc.
        dialogue_count = 0 # counting ""
    
        for ele in sentence_list:
            if ele == None or len(ele) == 0:
                continue
            elif len(sentences) != 0 and ele[0] in "?!.":
                # count statistics
                if ele[0] == '!':
                    exclamation_marks += 1
                elif ele[0] == '?':
                    question_marks += 1
                sentences[-1] += ele.strip()
            else:
                # POS tagging
                tokens = nltk.word_tokenize(ele)
                words_and_tags = nltk.pos_tag(tokens)
                for word, tag in words_and_tags:
                    # count tags
                    if tag == 'PRP':
                        pp_tags += 1
                    elif tag == 'DT':
                        dt_tags += 1
                    elif tag == 'JJ':
                        adj_tags += 1
                    elif tag == 'RB':
                        adv_tags += 1
                    elif tag == 'VBD':
                        vbd_tags += 1
                    elif tag[0:2] == 'NN':
                        noun_tags += 1
                    elif tag == 'UH':
                        interjection_tags += 1
                    elif tag == 'MD':
                        modal_tags += 1
                    elif tag == 'FW':
                        foreign_word_tags += 1
                        
                # count statistics
                sentence_count += 1
                words = re.findall(r'\w+’?\w*', ele)
                # count capitalized words except start of novel and sentences 
                # (assume sentence starts with a space after previous sentence's fullstop)
                capitalized = re.findall(r'(?<!\.\s)(?<!\A)([A-Z]\w+’?\w*)', ele) 
                dashes = re.findall(r'-+', ele)
                commas = re.findall(r',', ele)
                italics = re.findall(r'_\w+_', ele)
                contractions = re.findall(r'\w+’\w+', ele)
                if not in_dialogue:
                    dialogues = re.findall(r'[‘“].*', ele) # Look for start of dialogue
                    in_dialogue = len(dialogues) > 0
                else:
                    dialogues = re.findall(r'.*[”’]', ele) # Look for end of dialogue
                    in_dialogue = len(dialogues) == 0
                # contractions treated as one word
                total_words += len(words)
                no_spaces = ''.join(words)
                character_count += len(no_spaces)
                capitalized_words += len(capitalized)
                dash_count += len(dashes)
                comma_count += len(commas)
                italics_count += len(italics)
                contractions_count += len(contractions)
                # Count sentences that have dialogue
                dialogue_count += 1 if in_dialogue else 0
                for word in words:
                    # TODO: Handle known words that are in ALL CAPS
                    # TODO: Handle named entities (people, cities?), count them as the same since not style
                    # TODO: Project Gutenberg seems to have italicised? words that are _<word>_, handle.
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        # counting word occurrence because might as well
                        vocab[word] += 1
                    if word in STOPWORDS:
                        stopword_count += 1
                sentences += [ele.strip()]
        
        stats["personalpronoun_tags_per_sent"].append(np.log(1+pp_tags/sentence_count))
        stats["determiner_tags_per_sent"].append(np.log(1+dt_tags/sentence_count))
        stats["adjective_tags_per_sent"].append(np.log(1+adj_tags/sentence_count))
        stats["adverb_tags_per_sent"].append(np.log(1+adv_tags/sentence_count))
        stats["verb_past_tags_per_sent"].append(np.log(1+vbd_tags/sentence_count))
        stats["noun_tags_per_sent"].append(np.log(1+noun_tags/sentence_count))
        stats["interjection_tags_per_sent"].append(np.log(1+interjection_tags/sentence_count))
        stats["modal_tags_per_sent"].append(np.log(1+modal_tags/sentence_count))
        stats["foreign_word_tags_per_sent"].append(np.log(1+foreign_word_tags/sentence_count))
        stats["stopword_count_per_sent"].append(np.log(1+stopword_count/sentence_count))
        stats["avg_word_per_sentence"].append(np.log(1+total_words/sentence_count))
        stats["avg_word_length"].append(np.log(1+character_count/total_words))
        stats["capitalized_per_sentence"].append(np.log(1+capitalized_words/sentence_count))
        stats["dashes_per_sent"].append(np.log(1+dash_count/sentence_count))
        stats["comma_count_per_sent"].append(np.log(1+comma_count/sentence_count))
        #stats["exclamation_marks_per_sent"].append(np.log(1+exclamation_marks/sentence_count))
        #stats["question_marks_per_sent"].append(np.log(1+question_marks/sentence_count))
        stats["italics_per_sent"].append(np.log(1+italics_count/sentence_count))
        #stats["contractions_per_sent"].append(np.log(1+contractions_count/sentence_count))
        stats["dialogue_per_sent"].append(np.log(1+dialogue_count/sentence_count))

        # vocab needs to be able to count rare words and identify them
        # stats["vocab_word_count"].append(len(vocab.keys()))
        
    features = pd.DataFrame(data=stats)
    scaled = scaler.transform(features)
    return scaled

def get_model():
    
    # Logistic Regression Model

    # C = regularization strength
    # max_iter, default=100, can try increase see got diff anot
    # multi_class, can be 'ovr' = binary for each label or 'multinomial'
    model = LogisticRegression(penalty='l2', C=0.8, solver='saga', multi_class='multinomial', max_iter=1000)
    return model

def get_trained_model(model, scaler, stats):
    # x_train = array of sentences, from all authors
    # y_train = array of authors of corresponding sentence in x_train
    x_train = scale_features(stats, scaler, False)
#     print("Class features distribution:")
#     print(x_train)

    # y_train = books
    y_train = []
    for author in authors:
        y_train += [author] * num_novels[author]
    # print(num_novels)

    train_model(model, x_train, y_train)

def predict_feature_engineering(isDebugging=False):
    stats = extract_features()
    model = get_model()
    scaler = StandardScaler()
    get_trained_model(model, scaler, stats)

    test_cases = get_all_tests()

    tests = pd.Series(test_cases)
    x_test = extract_test_features(tests, scaler)

    output_probs = predict(model, x_test)
    output_answers = []
    output_confidence = []
    output_authors = []
    # try to compare confidence respective to other authors
    diff_to_next_highest = []
    for row in output_probs:
        highest = max(row)
        index = [np.where(row == highest)[0]][0][0]
        author = authorsIndexed[index]

        if index == 0:
            diff_to_next_highest += [highest - row[1]] if highest - row[1] <= highest - row[2] else [highest - row[2]]
        elif index == 1:
            diff_to_next_highest += [highest - row[0]] if highest - row[0] <= highest - row[2] else [highest - row[2]]
        elif index == 2:
            diff_to_next_highest += [highest - row[0]] if highest - row[0] <= highest - row[1] else [highest - row[1]] 

        output_confidence += [highest]
        output_answers.append( (author, highest) )
        output_authors.append(author)
    if isDebugging:
        check_test_results(output_answers)
        generate_F1_score(model, output_authors)
    else:
        return output_answers

def generate_F1_score(model, output_authors):
    # Use f1-macro as the metric
    score = f1_score(correct_answers, output_authors, average='macro')
    print('LR score on validation = {}'.format(score))
    from sklearn.metrics import confusion_matrix, classification_report
    print(classification_report(correct_answers, output_authors))

    # get importance
    importance = model.coef_
    # summarize feature importance
    print("Feature importance for Mark Twain:")
    # plot feature importance
    pyplot.bar([x for x in range(len(importance[2]))], importance[2])
    pyplot.show()

    # print("Feature importance for 0:")
    # #     for i,v in enumerate(importance[1]):
    # #         print('Feature: %0d, Score: %.5f' % (i,v))
    # # plot feature importance
    # pyplot.bar([x for x in range(len(importance[1]))], importance[1])
    # pyplot.show()

    # print("Feature importance for 1")
    # #     for i,v in enumerate(importance[2]):
    # #         print('Feature: %0d, Score: %.5f' % (i,v))
    # # plot feature importance
    # pyplot.bar([x for x in range(len(importance[2]))], importance[2])
    # pyplot.show()
