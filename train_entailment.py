# -*- coding: UTF-8 -*-

import os
import sys
import numpy as np
import load_word2vec
from config import *
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

use_this_dataset = snli_dev_file

def score_setup(row):
    convert_dict = {
      'entailment': 0,
      'neutral': 1,
      'contradiction': 2
    }
    score = np.zeros((3,))
    for x in range(1,6):
        tag = row["label"+str(x)]
        if tag in convert_dict: 
            score[convert_dict[tag]] += 1
    return score / (1.0*np.sum(score))

# Remove rows/columns or padd with 0 to get the shape required
def fit_to_size(matrix, shape):
    res = np.zeros(shape)
    slices = [slice(0,min(dim,shape[e])) for e, dim in enumerate(matrix.shape)]
    res[slices] = matrix[slices]
    return res

def split_data_into_scores():
    import csv
    iteration = 0
    #Load the vocabulary
    google_vocab = load_word2vec.WordEmbedding( path = glove_file )
    google_vocab.load( vector_type = "glove" )
    
    with open( os.path.join(root_dir, use_this_dataset), "r" ) as data:
        train = csv.DictReader(data, delimiter='\t')
        evi_sentences = []
        hyp_sentences = []
        labels = []
        scores = []

        for row in train:
            # Append zeros if the size is not a fixed size.
            hyp_sentences.append( fit_to_size( np.vstack( google_vocab.transform(row["sentence1"].lower())[0] ), (max_hypothesis_length, vector_size) ) )
            evi_sentences.append( fit_to_size( np.vstack( google_vocab.transform(row["sentence2"].lower())[0] ), (max_hypothesis_length, vector_size) ) )
            labels.append(row["gold_label"])
            scores.append(score_setup(row))

        # Do not need the vocab anymore. Free memory
        google_vocab.reset()
        hyp_sentences = np.array(hyp_sentences)
        evi_sentences = np.array(evi_sentences)
        return (hyp_sentences, evi_sentences), labels, np.array(scores)
    

data_feature_list, correct_values, correct_scores = split_data_into_scores()

l_h, l_e = max_hypothesis_length, max_evidence_length
N, D, H = batch_size, vector_size, hidden_size
l_seq = l_h + l_e

tf.reset_default_graph()

# With both those out of the way, we can define our LSTM using TensorFlow as follows:
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)


# Add tensorflow's default dropout. Can not simply add dropout to all hidden units in a LSTM, as they have relationships with each other
lstm_drop =  tf.contrib.rnn.DropoutWrapper(lstm, input_p, output_p)


# N: The number of elements in each batches, 
# l_h: The maximum length of a hypothesis. Training an RNN is extraordinarily difficult without rolling it out to a fixed length.
# l_e: The maximum length of evidence, the first sentence.
# D: The size of vectors.

# Where the hypotheses will be stored during training.
hyp = tf.placeholder(tf.float32, [N, l_h, D], 'hypothesis')
# Where the evidences will be stored during training.
evi = tf.placeholder(tf.float32, [N, l_e, D], 'evidence')
# Where correct scores will be stored during training.
y = tf.placeholder(tf.float32, [N, 3], 'label')

# lstm_size: the size of the gates in the LSTM, as in the first LSTM layer's initialization.
# The LSTM used for looking backwards through the sentences, similar to lstm.
lstm_back = tf.contrib.rnn.BasicLSTMCell(lstm_size)


# input_p: the probability that inputs to the LSTM will be retained.
# output_p: the probability that outputs from the LSTM will be retained.

# A dropout wrapper for lstm_back, like lstm_drop.
lstm_drop_back = tf.contrib.rnn.DropoutWrapper(lstm_back, input_p, output_p)


# Initial values for the fully connected layer's weights.
fc_initializer = tf.random_normal_initializer(stddev=0.1) 

# hidden_size: the size of the outputs from each lstm layer multiplied by 2 to account for the two LSTMs.
# Storage for the fully connected layer's weights.
fc_weight = tf.get_variable('fc_weight', [2*hidden_size, 3], initializer = fc_initializer)

# Storage for the fully connected layer's bias.
fc_bias = tf.get_variable('bias', [3])

# tf.GraphKeys.REGULARIZATION_LOSSES:  A key to a collection in the graph
#   designated for losses due to regularization.
#   In this case, this portion of loss is regularization on the weights
#   for the fully connected layer.
tf.add_to_collection( tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(fc_weight) ) 

# x: the inputs to the bidirectional_rnn
x = tf.concat([hyp, evi], 1) # N, (Lh+Le), d
# Permuting batch_size and n_steps
x = tf.transpose(x, [1, 0, 2]) # (Le+Lh), N, d
# Reshaping to (n_steps*batch_size, n_input)
x = tf.reshape(x, [-1, vector_size]) # (Le+Lh)*N, d
# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
x = tf.split(x, l_seq,)

# tf.contrib.rnn.static_bidirectional_rnn: Runs the input through
#   two recurrent networks, one that runs the inputs forward and one
#   that runs the inputs in reversed order, combining the outputs.
rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm, lstm_back, x, dtype=tf.float32)

# rnn_outputs: the list of LSTM outputs, as a list. 
# What we want is the latest output, rnn_outputs[-1]

# The scores are relative certainties for how likely the output matches
#   a certain entailment: 
#     0: Positive entailment
#     1: Neutral entailment
#     2: Negative entailment
classification_scores = tf.matmul(rnn_outputs[-1], fc_weight) + fc_bias
 
# Since we have both classification scores and optimal scores, the choice here is using a variation on softmax loss from Tensorflow:
# tf.nn.softmax_cross_entropy_with_logits.
with tf.variable_scope('Accuracy'):
    predicts = tf.cast(tf.argmax(classification_scores, 1), 'int32')
    y_label = tf.cast(tf.argmax(y, 1), 'int32')
    corrects = tf.equal(predicts, y_label)
    num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

with tf.variable_scope("loss"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits( logits = classification_scores, labels = y )
    loss = tf.reduce_mean(cross_entropy)
    total_loss = loss + weight_decay * tf.add_n( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) )

# optimizer = tf.train.AdamOptimizer(learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
opt_op = optimizer.minimize(total_loss)

# Train the model
# Initialize variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()
# Launch the Tensorflow session
sess = tf.Session()
sess.run(init)

accuracies = []
training_iterations = range(0, training_iterations_count,batch_size)
training_iterations = tqdm(training_iterations)

print "Everything good. Running Training now."

for i in training_iterations:
    # Select indices for a random data subset
    batch = np.random.randint( data_feature_list[0].shape[0], size=batch_size )
    
    # Use the selected subset indices to initialize the graph's placeholder values
    hyps, evis, ys = (data_feature_list[0][batch,:],
                      data_feature_list[1][batch,:],
                      correct_scores[batch])
    
    # Run the optimization with these initialized values
    sess.run( [opt_op], feed_dict={hyp: hyps, evi: evis, y: ys} )
    # check_accuracy_step: how often the accuracy and loss should be tested and displayed.
    if (i/batch_size) % check_accuracy_step == 0:
        # Calculate batch accuracy
        acc = sess.run( accuracy, feed_dict={hyp: hyps, evi: evis, y: ys} )
        # # Calculate batch loss
        tmp_loss = sess.run( loss, feed_dict={hyp: hyps, evi: evis, y: ys} )
        # Display results
        accuracies.append( acc )

# Save the learned weights and bias
save_path = saver.save(sess, os.path.join( model_dir, "model.ckpt"))
sess.close()

# ----------------------------------------------------------------------------------------------------------------------
print "Maximum is : " + str(max(accuracies))        
plt.plot([j for j in range(len(accuracies))], accuracies)
plt.show()