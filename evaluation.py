from keras.models import load_model
import pickle
import numpy as np
import os 
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras_contrib.layers import CRF
from model import ModelTraining
with open('./models/dataset.pkl', 'rb') as f:
	X_train, Y_train, word_encoding, word_decoding, tag_encoding, tag_decoding = pickle.load(f)

	del X_train
	del Y_train

# sentence = 'john is expected to race tomorrow'.split()
# np bez vbn in nn nn

# sentence = 'send me some photos of that tree'.split()
# vb
# ppo
# dti
# nns
# in
# pp$
# nn

sentence = 'je veux un chien et un canard'.split()
# NP
# (S\NP)/NP
# NP/NP
# NP/NP
# NP

tokenized_sentence = []

for word in sentence:
	tokenized_sentence.append(word_encoding[word])

tokenized_sentence = np.asarray([tokenized_sentence])
padded_tokenized_sentence = pad_sequences(tokenized_sentence, maxlen=200)

print('The sentence is ', sentence)
print('The tokenized sentence is ',tokenized_sentence)
print('The padded tokenized sentence is ', padded_tokenized_sentence)
root = "/home/lengocluyen/word_embedding_french/"
root_directory = root + "ccgtagging/bilstm-crf/"
#model = load_keras_model(os.path.join(root_directory,'/models/model.h5'))
model_train = ModelTraining()
model_file = os.path.join(root_directory,'models/model.h5')
print(model_file)
model = model_train.load_keras_model(model_file)
prediction = model.predict(padded_tokenized_sentence)

print(prediction.shape)

for i, pred in enumerate(prediction[0]):
    try:
        print(sentence[199-i], ' : ', tag_decoding.get(np.argmax(pred)))
    except:
        pass
    #    print('NA')
    #print(i)
    #print(sentence[i])
    #print(tag_decoding.get(np.argmax(pred)))
    #print(pred)