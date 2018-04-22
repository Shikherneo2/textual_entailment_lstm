# -*- coding: UTF-8 -*-

import os
import random
import numpy as np
import load_word2vec
from tqdm import tqdm
import tensorflow as tf

root_dir = "/home/shikher/workspace/nlp_project/textual_entailment"
vectors_file = "GoogleNews-vectors-negative300.bin"
glove_file = "/home/shikher/course_things/cs512-news_clustering/datasets/glove.6B/glove.6B.100d.txt"

snli_test_file = "snli_1.0/snli_1.0_test.txt"

#Load the vocabulary
google_vocab = load_word2vec.WordEmbedding( path = glove_file )
google_vocab.load( vector_type = "glove" )

batch_size = 32
hidden_size = 64
vector_size = 100
lstm_size = hidden_size
input_p, output_p = 0.7, 0.7
max_hypothesis_length, max_evidence_length = 25, 25

convert_dict = {
      'entailment': 0,
      'neutral': 1,
      'contradiction': 2
}

# Remove rows/columns or padd with 0 to get the shape required
def fit_to_size(matrix, shape):
    res = np.zeros(shape)
    slices = [slice(0,min(dim,shape[e])) for e, dim in enumerate(matrix.shape)]
    res[slices] = matrix[slices]
    return res


def split_data_into_scores(limit=50000):
    import csv
    iteration = 0
    with open( os.path.join(root_dir, snli_test_file), "r" ) as data:
        train = csv.DictReader(data, delimiter='\t')
        evi_sentences = []
        hyp_sentences = []
        labels = []

        for row in train:
            # Append zeros if the size is not a fixed size.
            hyp_sentences.append( fit_to_size( np.vstack( google_vocab.transform(row["sentence1"].lower())[0] ), (max_hypothesis_length, vector_size) ) )
            evi_sentences.append( fit_to_size( np.vstack( google_vocab.transform(row["sentence2"].lower())[0] ), (max_hypothesis_length, vector_size) ) )
            labels.append(row["gold_label"])

        hyp_sentences = np.array(hyp_sentences)
        evi_sentences = np.array(evi_sentences)
        return (hyp_sentences, evi_sentences), labels
    
print "Loaded all data. Testing now..."

data_feature_list, labels = split_data_into_scores()

l_h, l_e = max_hypothesis_length, max_evidence_length
N, D, H = batch_size, vector_size, hidden_size
l_seq = l_h + l_e

google_vocab.reset()
tf.reset_default_graph()
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

# Add tensorflow's default dropout. Can not simply add dropout to all hidden units in a LSTM, as they have relationships with each other
lstm_drop =  tf.contrib.rnn.DropoutWrapper(lstm, input_p, output_p)

hyp = tf.placeholder(tf.float32, [N, l_h, D], 'hypothesis')
evi = tf.placeholder(tf.float32, [N, l_e, D], 'evidence')
# y = tf.placeholder(tf.float32, [N, 3], 'label')

# lstm_size: the size of the gates in the LSTM, as in the first LSTM layer's initialization.
# The LSTM used for looking backwards through the sentences, similar to lstm.
lstm_back = tf.contrib.rnn.BasicLSTMCell(lstm_size)

# A dropout wrapper for lstm_back, like lstm_drop.
lstm_drop_back = tf.contrib.rnn.DropoutWrapper(lstm_back, input_p, output_p)

# Initial values for the fully connected layer's weights.
fc_initializer = tf.random_normal_initializer(stddev=0.1) 
fc_weight = tf.get_variable('fc_weight', [2*hidden_size, 3])
fc_bias = tf.get_variable('bias', [3])


# x: the inputs to the bidirectional_rnn
x = tf.concat([hyp, evi], 1)
# Permuting batch_size and n_steps
x = tf.transpose(x, [1, 0, 2]) # (Le+Lh), N, d
# Reshaping to (n_steps*batch_size, n_input)
x = tf.reshape(x, [-1, vector_size]) # (Le+Lh)*N, d
# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
x = tf.split(x, l_seq,)

rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm, lstm_back, x, dtype=tf.float32)


classification_scores = tf.matmul(rnn_outputs[-1], fc_weight) + fc_bias
# The scores are relative certainties for how likely the output matches
#   a certain entailment: 
#     0: Positive entailment
#     1: Neutral entailment
#     2: Negative entailment

 
# Initialize variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess, "model.ckpt")


preds = []
training_iterations = range(0, data_feature_list[0].shape[0])
training_iterations = tqdm(training_iterations)
for i in training_iterations:
	prediction = sess.run(classification_scores, feed_dict={hyp: ([data_feature_list[0][i]] * N),
	                                                        evi: ([data_feature_list[1][i]] * N)})
	                                                        # y: [[0,0,0]]*N})
	preds.append(np.argmax(prediction[0]))

accuracy = []
for i in range(len(preds)):
	if labels[i] in convert_dict:
		if preds[i]==convert_dict[labels[i]]:
			accuracy.append(1)
		else:
			accuracy.append(0)
print sum(accuracy)*100/float(len(accuracy))

avg_acc_null_hypo = []
for times in range(5):
	accuracy = []
	null_hypo = [ random.randint(0,2) for i in range(len(labels))]
	for i in range(len(null_hypo)):
		if labels[i] in convert_dict:
			if null_hypo[i]==convert_dict[labels[i]]:
				accuracy.append(1)
			else:
				accuracy.append(0)
	avg_acc_null_hypo.append(sum(accuracy)/float(len(accuracy)))		
print np.mean(avg_acc_null_hypo)*100	
# evidences_test = ["Maurita and Jade both were at the scene of the car crash.", "No one saw the bear anywhere", "President Obama visited Greece today to sign a trade deal."]
# hypotheses_test = ["Two people both saw the car crash.", "He saw a big bear in the forest.", "We play cricket every week."]

# sentence1 = [fit_to_size(np.vstack(google_vocab.transform(evidence)[0]),
#                          (30, 50)) for evidence in evidences_test]

# sentence2 = [fit_to_size(np.vstack(google_vocab.transform(hypothesis)[0]),
#                          (30,50)) for hypothesis in hypotheses_test]

# print len(sentence1)
# for i in range(len(sentence1)):
# 	prediction = sess.run(classification_scores, feed_dict={hyp: ([sentence1[i]] * N),
# 	                                                        evi: ([sentence2[i]] * N),
# 	                                                        y: [[0,0,0]]*N})
# 	print( ["Positive", "Neutral", "Negative"][np.argmax(prediction[0])] + " entailment" )

sess.close()