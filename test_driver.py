from classifier_gensim import predict_doc2vec_results
from feature_engineering import predict_feature_engineering
from functionwords import predict_k_nearest_neighbours_results
from collections import Counter
from test_runner import check_test_results


def main():

    print("\n\nGetting doc2vec predictions...\n")
    doc2vec_results = predict_doc2vec_results()
    print("\n\nGetting feature engineering predictions...\n")
    feature_engineering_results = predict_feature_engineering()
    print("\n\nGetting k-Nearest-Neighbours predictions...\n")
    k_nearest_neighbours_results = predict_k_nearest_neighbours_results()

    combined_results = list(zip(doc2vec_results, feature_engineering_results, k_nearest_neighbours_results))
    print("\n\nCombining results into ensemble...\n")
    ensemble_results = [determine_ensemble_answer(result) for result in combined_results]

    check_test_results(ensemble_results)


def determine_ensemble_answer(*args):
    answers = Counter([element[0] for element in args])

    # If more than 50% of the models have the same answer, return that answer
    most_common_answer, most_common_answer_frequency = answers.most_common(1)[0]
    if most_common_answer_frequency > 0.5 * len(args):
        return most_common_answer

    # Else, if there is a clear plurality winner, return that answer
    _, second_most_common_answer_frequency = answers.most_common(2)[1]
    if most_common_answer_frequency > second_most_common_answer_frequency:
        return most_common_answer

    # Else, select the answer with the highest confidence value
    sorted_answers = sorted(args, key=lambda x: x[1], reverse=True)
    return sorted_answers[0][0]


if __name__ == "__main__":
    main()