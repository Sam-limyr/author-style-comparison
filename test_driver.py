from classifier_gensim import predict_doc2vec_results
from feature_engineering import predict_feature_engineering
from functionwords import predict_k_nearest_neighbours_results
from collections import Counter
from test_runner import check_test_results
from sklearn.preprocessing import scale


def main():

    print("\n\nGetting doc2vec predictions...\n")
    normalized_doc2vec_results = normalize_results(predict_doc2vec_results())
    print("\n\nGetting feature engineering predictions...\n")
    normalized_feature_engineering_results = normalize_results(predict_feature_engineering())
    print("\n\nGetting k-Nearest-Neighbours predictions...\n")
    normalized_k_nearest_neighbours_results = normalize_results(predict_k_nearest_neighbours_results())

    combined_results = list(zip(
        normalized_doc2vec_results
        ,        normalized_feature_engineering_results
        ,        normalized_k_nearest_neighbours_results
    ))

    print("\n\nCombining results into ensemble...\n")
    ensemble_results = [determine_ensemble_answer(result) for result in combined_results]

    check_test_results(ensemble_results)


def determine_ensemble_answer(*args):
    results = args[0]
    answers = Counter([element[0] for element in results])

    # If more than 50% of the models have the same answer, return that answer
    most_common_answer, most_common_answer_frequency = answers.most_common(1)[0]
    if most_common_answer_frequency > 0.5 * len(results):
        print("Majority pick: {}".format(most_common_answer))
        return most_common_answer

    # Else, if there is a clear plurality winner, return that answer
    _, second_most_common_answer_frequency = answers.most_common(2)[1]
    if most_common_answer_frequency > second_most_common_answer_frequency:
        print("Plurality pick: {}".format(most_common_answer))
        return most_common_answer

    # Else, select the answer with the highest confidence value
    sorted_answers = sorted(results, key=lambda x: x[1], reverse=True)
    print("Tiebreak pick: {} | Score {} against {}".format(sorted_answers[0][0], round(float(sorted_answers[0][1]), 4),
                                                           round(float(sorted_answers[1][1]), 4)))
    return sorted_answers[0][0]


def normalize_results(results_list_and_confidence_scores):
    predictions = [element[0] for element in results_list_and_confidence_scores]
    confidence_scores = [element[1] for element in results_list_and_confidence_scores]
    normalized_scores = normalize_confidence_score(confidence_scores)
    normalized_results = list(zip(predictions, normalized_scores))
    return normalized_results


def normalize_confidence_score(scores):
    return scale(scores, axis=0, with_mean=True, with_std=True, copy=True)


if __name__ == "__main__":
    main()