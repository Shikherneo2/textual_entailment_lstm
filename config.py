# Configuration variables used in training and testing

vectors_file = "GoogleNews-vectors-negative300.bin"
root_dir = "/home/shikher/workspace/nlp_project/textual_entailment"
model_dir = "/home/shikher/workspace/nlp_project/textual_entailment/models" 
glove_file = "/home/shikher/course_things/cs512-news_clustering/datasets/glove.6B/glove.6B.100d.txt"

snli_test_file = "snli_1.0/snli_1.0/snli_1.0_test.txt"
snli_dev_file = "snli_1.0/snli_1.0/snli_1.0_dev.txt"
snli_train_file = "snli_1.0/snli_1.0/snli_1.0_train.txt"

batch_size = 32
vector_size = 100
max_hypothesis_length, max_evidence_length = 25, 25

hidden_size = 128
learning_rate = 1
weight_decay = 0.0001
lstm_size = hidden_size
check_accuracy_step = 100
input_p, output_p = 0.7, 0.7
training_iterations_count = 500000