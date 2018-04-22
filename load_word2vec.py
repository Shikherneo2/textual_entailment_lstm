import time
import gensim
import string
import numpy as np
from nltk import word_tokenize

class WordEmbedding(object):
    #logic for loading Google News vectors and transforming strings to word indices

    def __init__(self, path='./GoogleNews-vectors-negative300.bin'):
        self.path = path
        self.model = None
        self.vocab = {}


    def load(self, vector_type = "google"):
        if( vector_type == "google" ):

            t0 = time.time()
            self.model = gensim.models.Word2Vec.load_word2vec_format(self.path,unicode_errors='ignore', binary=True)
            print str(time.time() - t0 ) + " to load the model"
            for i in range(len(self.model.index2word)):
                self.vocab[self.model.index2word[i]]=i
        
        elif( vector_type == "glove" ):
            with open(self.path, "r") as glove:
                for line in glove:
                    name, vector = tuple(line.split(" ", 1))
                    self.vocab[name] = np.fromstring(vector, sep=" ")
            print "Vocabulary created"            

    """
        Turns an input sentence into an (n,d) matrix, 
        where n is the number of tokens in the sentence
        and d is the number of dimensions each word vector has.
        NOTE -- consider using more sophisticated tokenizers here.
        NOTE -- what about names? can we use NER?
    """
    def transform(self, sentence):
        tokens = word_tokenize(sentence.lower())
        tokens = [i for i in tokens if i not in string.punctuation]
        # tokens = sentence.lower().split(" ")
        rows = []
        words = []
        #Greedy search for tokens
        for token in tokens:
            i = len(token)
            while len(token) > 0 and i > 0:
                word = token[:i]
                if word in self.vocab:
                    rows.append(self.vocab[word])
                    words.append(word)
                    token = token[i:]
                    i = len(token)
                else:
                    i = i-1
        return rows, words

    def reset(self):
        self.vocab = {}    