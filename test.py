import numpy as np
import load_word2vec
# def fit_to_size(matrix, shape):
#     res = np.zeros(shape)
#     slices = [slice(0,min(dim,shape[e])) for e, dim in enumerate(matrix.shape)]
#     res[slices] = matrix[slices]
#     return res

# a = np.matrix([[12,3,4,5],[121,4,45,75],[112,36,43,85]])
# print fit_to_size(a, (6,4))

glove_file = "/home/shikher/course_things/cs512-news_clustering/datasets/glove.6B/glove.6B.50d.txt"
google_vocab = load_word2vec.WordEmbedding( path = glove_file )
google_vocab.load( vector_type = "glove" )

print google_vocab.transform("the best way to sleep well is not sleep, at all.")
print google_vocab.transform("Lets see if that works and/or not.")[1]



# Testing
# sentence2sequence("Maurita and Jade both were at the scene of the car crash.")[1]

evidences = ["Maurita and Jade both were at the scene of the car crash."]
hypotheses = ["two people saw the car crash."]

sentence1 = [fit_to_size(np.vstack(google_vocab.transform(evidence)[0]),
                         (30, 50)) for evidence in evidences_test]

sentence2 = [fit_to_size(np.vstack(google_vocab.transform(hypothesis)[0]),
                         (30,50)) for hypothesis in hypotheses_test]

prediction = sess.run(classification_scores, feed_dict={hyp: (sentence1 * N),
                                                        evi: (sentence2 * N),
                                                        y: [[0,0,0]]*N})
print( ["Positive", "Neutral", "Negative"][np.argmax(prediction[0])] + " entailment" )

sess.close()