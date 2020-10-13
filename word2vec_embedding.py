#!/usr/bin/env python
# -*- coding: utf-8 -*-
import word2vec
import os, pickle
import sys 

class Word2VecEmbedding:
    def load_model(self, binpath):
        self.model  = word2vec.load(binpath, encoding="latin-1")
        #self.model  = word2vec.load(binpath,encoding = "utf-8")

    def word_encoding_2_id(self, word):
        return self.model.vocab_hash[bytes(word,"utf8").decode("iso-8859-1")]

    def word_decoding_2_word(self, word_index):
        return bytes(self.model.vocab[word_index],'iso-8859-1').decode('utf8')

    def word_encoding_2_vector(self,word):
        try:
            return self.model.vectors[self.word_encoding_2_id(word)]
        except:
            return None

    def save_embedding_to_pickle_file(self,output_path=None):
        embedding_index = {}
        for ln,line in enumerate(self.model.vocab):
            word = self.model.vocab[ln]
            word_vector = self.model.vectors[ln]
            embedding_index[word]=word_vector
        if not os.path.exists('pickled_data/'):
        	os.makedirs('./pickled_data/')
        if output_path is None:
            output_path = "word2vec.pkl"
        with open('./pickled_data/'+output_path, 'wb') as f:
            pickle.dump(embedding_index, f)
        return embedding_index
    def embedding_index(self,output_path="None"):
        embedding_index = {}
        for ln,line in enumerate(self.model.vocab):
            word = self.model.vocab[ln]
            word_vector = self.model.vectors[ln]
            embedding_index[word]=word_vector
        return embedding_index

"""
def word_encoding(word):

def word_decoding(word_encoding):


def dataset_encoding(x_sentence):
    return ""

def dataset_decoding(x_sentence_encoding):
"""

binpath = "/home/lengocluyen/word_embedding_french/word2vec/frWac2Vec/01_frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.bin"
#print("Shape model:",model.vectors.shape)
#ix,
#words
"""
word_embdding = Word2VecEmbedding()
word_embdding.load_model(binpath)

text=""
for i in range(0,200):
    text = text + "\t" + bytes(word_embdding.word_decoding_2_word(i),'iso-8859-1').decode('utf8')
print(text)
print(word_embdding.model.get_vector("tôt"))"""
#word_embdding.save_embedding_to_pickle_file()
