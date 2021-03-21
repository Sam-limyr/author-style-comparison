# Code credit: https://www.sbert.net/docs/usage/semantic_textual_similarity.html

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.'
          ]

queries = ['Here is an equine creature',
           'The person is doing the thing',
           'A person is using an instrument.',
           'A female is using an instrument.'
           ]

#Encode all sentences
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
query_embeddings = model.encode(queries, convert_to_tensor=True)

#Compute cosine-similarities for each corpus sentence with each query sentence
cosine_scores = util.pytorch_cos_sim(corpus_embeddings, query_embeddings)

#Find the pairs with the highest cosine similarity scores
pairs = []
for i in range(len(cosine_scores[0])):
    pairs.append([])
    for j in range(len(cosine_scores)):
        pairs[i].append({'index': [j, i], 'score': cosine_scores[j][i]})

#Sort scores in decreasing order
sorted_pairs = []
for query_pairs in pairs:
    sorted_pairs.append(sorted(query_pairs, key=lambda x: x['score'], reverse=True))

print("Format:\nQuery Sentence \t\t | \t Most Similar Corpus Sentence \t | \t\t Score\n")

TOP_K_SCORES_TO_PRINT = 1
for query_pairs in sorted_pairs:
    for value in query_pairs[0:TOP_K_SCORES_TO_PRINT]:
        i, j = value['index']
        print("{} \t\t {} \t\t Score: {:.4f}".format(queries[j], corpus[i], value['score']))

# TODO: Explore util.paraphrase_mining().
#       Problem: It only calculates a corpus's scores with itself.

