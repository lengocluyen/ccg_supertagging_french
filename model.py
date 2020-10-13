import numpy as np
import pickle, sys, os

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding,Add
from keras.layers import Dense, Input
from keras.layers import TimeDistributed,concatenate
from keras.layers import LSTM, Bidirectional,SpatialDropout1D
from keras.models import Model

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import load_model
from keras.layers import Masking,Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from word2vec_embedding import Word2VecEmbedding
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from keras.callbacks import ModelCheckpoint



class ModelTraining():
    def __init__(self,max_sequence_length=100, embedding_dim=200,test_split=0.2, dev_split=0.2, batch_size=32):
        print("Set parametres for Model Training")
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.test_split = test_split
        self.dev_split = dev_split
        self.batch_size = batch_size

    def building_model_full_merge(self,voca2id_datasets, embedding_matrix,rapport_save_path,number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[0]
        lemma2ids = voca2id_datasets[1]
        upostag2ids = voca2id_datasets[2]
        deprel2ids = voca2id_datasets[3]
        ##################### word ###############################
        embedding_word_layer = Embedding(len(word2ids)+1, self.embedding_dim,
                                    weights=[embedding_matrix[0]],
                                    input_length=self.max_sequence_length,
                                    trainable=False)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        #l_lstm_word = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_word_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        #preds_word = TimeDistributed(Dense(nb_output_class+1,activation='relu'))(l_lstm_word)

        ##################### lemma ###############################
        embedding_lemma_layer = Embedding(len(lemma2ids)+1, self.embedding_dim,
                                    weights=[embedding_matrix[1]],
                                    input_length=self.max_sequence_length,
                                    trainable=False)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)
        #l_lstm_lemma = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_lemma_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        #preds_lemma = TimeDistributed(Dense(nb_output_class+1,activation='relu'))(l_lstm_lemma)
        ##################### Upostag ###############################
        embedding_upostag_layer = Embedding(input_dim = len(upostag2ids)+1, output_dim = len(upostag2ids)+1,\
                                            input_length=self.max_sequence_length,trainable=True)
        sequence_upostag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_upostag_sequences = embedding_upostag_layer(sequence_upostag_input)
        #l_lstm_upostag = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_upostag_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        #preds_upostag = TimeDistributed(Dense(nb_output_class+1,activation='relu'))(l_lstm_upostag)
        ##################### Deprel ###############################
        embedding_deprel_layer = Embedding(input_dim = len(deprel2ids)+1, output_dim = len(deprel2ids)+1,\
                                            input_length=self.max_sequence_length,trainable=True)
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)
        #l_lstm_deprel = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_deprel_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        #preds_deprel = TimeDistributed(Dense(nb_output_class+1,activation='relu'))(l_lstm_deprel)
        #####################Dense ########################

        #added = Add()([preds_word, preds_lemma])
        #Sl_lstm = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_lemma_sequences)
        merge_out =concatenate([embedded_word_sequences, embedded_lemma_sequences,embedded_upostag_sequences,embedded_deprel_sequences])
        #merge_dropout =  SpatialDropout1D(0.3)(merge_out)
        main_lstm = Bidirectional(LSTM(units=number_layer, return_sequences=True,
                               recurrent_dropout=0.1))(merge_out)
        out = TimeDistributed(Dense(nb_output_class, activation="relu"))(main_lstm)

        ################# Output Layer#######################
        crf = CRF(nb_output_class,sparse_target=False)
        #model = Model(sequence_input,preds)
        model = Model(inputs=[sequence_word_input,sequence_lemma_input,sequence_upostag_input,sequence_deprel_input], outputs=crf(out))
        model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        #model.compile(loss=crf_loss,optimizer='rmsprop',metrics=[crf_viterbi_accuracy])

        #model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])

        sumary = model.summary()
        self.write_rapport_file(str(sumary),rapport_save_path)
        print("Model fitting - Bidirectional LSTM and CRF")
        return model
    def building_model_full_merge_con(self,voca2id_datasets, embedding_matrix,rapport_save_path,number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[0]
        lemma2ids = voca2id_datasets[1]
        upostag2ids = voca2id_datasets[2]
        deprel2ids = voca2id_datasets[3]
        ##################### word ###############################
        embedding_word_layer = Embedding(len(word2ids)+1, self.embedding_dim,
                                    weights=[embedding_matrix[0]],
                                    input_length=self.max_sequence_length,
                                    trainable=False)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        #l_lstm_word = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_word_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        #preds_word = TimeDistributed(Dense(nb_output_class+1,activation='relu'))(l_lstm_word)

        ##################### lemma ###############################
        embedding_lemma_layer = Embedding(len(lemma2ids)+1, self.embedding_dim,
                                    weights=[embedding_matrix[1]],
                                    input_length=self.max_sequence_length,
                                    trainable=False)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)
        #l_lstm_lemma = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_lemma_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        #preds_lemma = TimeDistributed(Dense(nb_output_class+1,activation='relu'))(l_lstm_lemma)
        ##################### Upostag ###############################
        embedding_upostag_layer = Embedding(input_dim = len(upostag2ids)+1, output_dim = len(upostag2ids)+1,\
                                            input_length=self.max_sequence_length,trainable=True)
        sequence_upostag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_upostag_sequences = embedding_upostag_layer(sequence_upostag_input)
        #l_lstm_upostag = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_upostag_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        #preds_upostag = TimeDistributed(Dense(nb_output_class+1,activation='relu'))(l_lstm_upostag)
        ##################### Deprel ###############################
        embedding_deprel_layer = Embedding(input_dim = len(deprel2ids)+1, output_dim = len(deprel2ids)+1,\
                                            input_length=self.max_sequence_length,trainable=True)
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)
        #l_lstm_deprel = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_deprel_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        #preds_deprel = TimeDistributed(Dense(nb_output_class+1,activation='relu'))(l_lstm_deprel)
        #####################Dense ########################

        #added = Add()([preds_word, preds_lemma])
        #Sl_lstm = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_lemma_sequences)
        merge_out =concatenate([embedded_word_sequences, embedded_lemma_sequences,embedded_upostag_sequences,embedded_deprel_sequences])
        new_embedding = self.embedding_dim + self.embedding_dim +  len(upostag2ids) +len(deprel2ids)+2
        reshape = Reshape((self.max_sequence_length,new_embedding,1))(merge_out)
        filter_sizes = [3,4,5]
        num_filters = 512
        drop = 0.5
        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], new_embedding), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], new_embedding), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], new_embedding), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(self.max_sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(self.max_sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(self.max_sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop)(flatten)

        #merge_dropout =  SpatialDropout1D(0.3)(merge_out)
        main_lstm = Bidirectional(LSTM(units=number_layer, return_sequences=True,
                               recurrent_dropout=0.1))(dropout)
        out = TimeDistributed(Dense(nb_output_class, activation="relu"))(main_lstm)

        ################# Output Layer#######################
        crf = CRF(nb_output_class,sparse_target=False)
        #model = Model(sequence_input,preds)
        model = Model(inputs=[sequence_word_input,sequence_lemma_input,sequence_upostag_input,sequence_deprel_input], outputs=crf(out))
        model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        #model.compile(loss=crf_loss,optimizer='rmsprop',metrics=[crf_viterbi_accuracy])

        #model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])

        sumary = model.summary()
        self.write_rapport_file(str(sumary),rapport_save_path)
        print("Model fitting - Bidirectional LSTM and CRF")
        return model
    def building_model_full(self,voca2id_datasets, embedding_matrix,rapport_save_path,number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[0]
        lemma2ids = voca2id_datasets[1]
        upostag2ids = voca2id_datasets[2]
        deprel2ids = voca2id_datasets[3]
        ##################### word ###############################
        embedding_word_layer = Embedding(len(word2ids)+1, self.embedding_dim,
                                    weights=[embedding_matrix[0]],
                                    input_length=self.max_sequence_length,
                                    trainable=False)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        l_lstm_word = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_word_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        preds_word = TimeDistributed(Dense(nb_output_class,activation='relu'))(l_lstm_word)

        ##################### lemma ###############################
        embedding_lemma_layer = Embedding(len(lemma2ids)+1, self.embedding_dim,
                                    weights=[embedding_matrix[1]],
                                    input_length=self.max_sequence_length,
                                    trainable=False)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)
        l_lstm_lemma = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_lemma_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        preds_lemma = TimeDistributed(Dense(nb_output_class,activation='relu'))(l_lstm_lemma)
        ##################### Upostag ###############################
        embedding_upostag_layer = Embedding(input_dim = len(upostag2ids)+1, output_dim = len(upostag2ids)+1,\
                                            input_length=self.max_sequence_length,trainable=True)
        sequence_upostag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_upostag_sequences = embedding_upostag_layer(sequence_upostag_input)
        l_lstm_upostag = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_upostag_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        preds_upostag = TimeDistributed(Dense(nb_output_class,activation='relu'))(l_lstm_upostag)
        ##################### Deprel ###############################
        embedding_deprel_layer = Embedding(input_dim = len(deprel2ids)+1, output_dim = len(deprel2ids)+1,\
                                            input_length=self.max_sequence_length,trainable=True)
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)
        l_lstm_deprel = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_deprel_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        preds_deprel = TimeDistributed(Dense(nb_output_class,activation='relu'))(l_lstm_deprel)
        #####################Dense ########################

        #added = Add()([preds_word, preds_lemma])
        #Sl_lstm = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_lemma_sequences)
        merge_out =concatenate([preds_word, preds_lemma,preds_upostag,preds_deprel])
        #merge_dropout =  SpatialDropout1D(0.3)(merge_out)
        #main_lstm = Bidirectional(LSTM(units=number_layer, return_sequences=True,
        #                       recurrent_dropout=0.6))(merge_dropout)
        main_lstm = Bidirectional(LSTM(units=number_layer, return_sequences=True,
                               recurrent_dropout=0.1))(merge_out)
        out = TimeDistributed(Dense(nb_output_class, activation="relu"))(main_lstm)

        ################# Output Layer#######################
        crf = CRF(nb_output_class)
        #model = Model(sequence_input,preds)
        model = Model(inputs=[sequence_word_input,sequence_lemma_input,sequence_upostag_input,sequence_deprel_input], outputs=crf(out))
        model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        
        #model.compile(loss=crf_loss,optimizer='rmsprop',metrics=[crf_viterbi_accuracy])

        #model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])

        sumary = model.summary()
        self.write_rapport_file(str(sumary),rapport_save_path)
        print("Model fitting - Bidirectional LSTM and CRF")
        return model

    def building_model_full_test(self,voca2id_datasets, embedding_matrix,rapport_save_path,number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        upostag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        char2ids = voca2id_datasets[0]
        ##################### word ###############################
        embedding_word_layer = Embedding(len(word2ids)+2, self.embedding_dim,
                                    weights=[embedding_matrix[1]],
                                    input_length=self.max_sequence_length,
                                    trainable=False)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        #l_lstm_word = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_word_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        #preds_word = TimeDistributed(Dense(nb_output_class,activation='relu'))(l_lstm_word)

        max_word_length=30
        word_lemma_char_input = Input(shape=(self.max_sequence_length,max_word_length,))
        embedding_word_char_layer = TimeDistributed(Embedding(input_dim=len(char2ids) + 2, output_dim=max_word_length,
                           input_length=max_word_length, mask_zero=True))(word_lemma_char_input)
        
        
        # character LSTM to get word encodings by characters
        word_char_encoding = TimeDistributed(LSTM(units=50, return_sequences=False,
                                        recurrent_dropout=0.5))(embedding_word_char_layer) 
       
        ##################### lemma ###############################
        embedding_lemma_layer = Embedding(len(lemma2ids)+2, self.embedding_dim,
                                    weights=[embedding_matrix[2]],
                                    input_length=self.max_sequence_length,
                                    trainable=False)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)
        #l_lstm_lemma = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_lemma_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        #preds_lemma = TimeDistributed(Dense(nb_output_class,activation='relu'))(l_lstm_lemma)
       
        ################ Char for lemma
        word_lemma_char_input = Input(shape=(self.max_sequence_length,max_word_length,))
        embedding_lemma_char_layer = TimeDistributed(Embedding(input_dim=len(char2ids) + 2, output_dim=max_word_length,
                           input_length=max_word_length, mask_zero=True))(word_lemma_char_input)
        # character LSTM to get word encodings by characters
        lemma_char_encoding = TimeDistributed(LSTM(units=50, return_sequences=False,
                                        recurrent_dropout=0.5))(embedding_lemma_char_layer) 

        
       
       ##################### Upostag ###############################
        embedding_upostag_layer = Embedding(input_dim = len(upostag2ids)+1, output_dim = len(upostag2ids)+1,\
                                            input_length=self.max_sequence_length,trainable=True)
        sequence_upostag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_upostag_sequences = embedding_upostag_layer(sequence_upostag_input)
        #l_lstm_upostag = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_upostag_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        #preds_upostag = TimeDistributed(Dense(nb_output_class,activation='relu'))(l_lstm_upostag)
        ##################### Deprel ###############################
        embedding_deprel_layer = Embedding(input_dim = len(deprel2ids)+1, output_dim = len(deprel2ids)+1,\
                                            input_length=self.max_sequence_length,trainable=True)
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)
        #l_lstm_deprel = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_deprel_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        #preds_deprel = TimeDistributed(Dense(nb_output_class,activation='relu'))(l_lstm_deprel)
        #####################Dense ########################

        embedded_word_char_sequences = concatenate([embedded_word_sequences, word_char_encoding])
        embedded_word_char_sequences = SpatialDropout1D(0.3)(embedded_word_sequences)

        embedded_lemma_char_sequences = concatenate([embedded_lemma_sequences, lemma_char_encoding])
        embedded_lemma_char_sequences = SpatialDropout1D(0.3)(embedded_lemma_sequences)

        x = concatenate([embedded_word_char_sequences,embedded_lemma_char_sequences,embedded_upostag_sequences,embedded_deprel_sequences])
        
        #added = Add()([preds_word, preds_lemma])
        #Sl_lstm = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_lemma_sequences)
        #merge_out = concatenate([l_lstm_word,l_lstm_lemma,l_lstm_upostag,l_lstm_deprel])
        #merge_out =concatenate([preds_word, preds_lemma,preds_upostag,preds_xpostag,preds_deprel])
        
        x_dropout =  SpatialDropout1D(0.3)(x)
        main_lstm = Bidirectional(LSTM(units=number_layer, return_sequences=True,
                               recurrent_dropout=0.6))(x_dropout)
        """main_lstm = Bidirectional(LSTM(units=number_layer, return_sequences=True,
                               recurrent_dropout=0.1))(merge_out)
        out = TimeDistributed(Dense(nb_output_class, activation="relu"))(main_lstm)"""

        ################# Output Layer#######################
        crf = CRF(nb_output_class)
        #model = Model(sequence_input,preds)
        model = Model(inputs=[word_lemma_char_input,sequence_word_input,sequence_lemma_input,sequence_upostag_input,sequence_deprel_input], outputs=crf(main_lstm))
        model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        
        #model.compile(loss=crf_loss,optimizer='rmsprop',metrics=[crf_viterbi_accuracy])

        #model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])

        sumary = model.summary()
        self.write_rapport_file(str(sumary),rapport_save_path)
        print("Model fitting - Bidirectional LSTM and CRF")
        return model
    def building_model_word_lemma(self, voca2id_datasets, embedding_matrix,rapport_save_path,number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[0]
        lemma2ids = voca2id_datasets[1] 
        #####################word###############################
        embedding_word_layer = Embedding(len(word2ids)+1, self.embedding_dim,
                                    weights=[embedding_matrix[0]],
                                    input_length=self.max_sequence_length,
                                    trainable=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        l_lstm_word = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_word_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        preds_word = TimeDistributed(Dense(nb_output_class,activation='relu'))(l_lstm_word)

        #####################lemma###############################
        embedding_lemma_layer = Embedding(len(lemma2ids)+1, self.embedding_dim,
                                    weights=[embedding_matrix[1]],
                                    input_length=self.max_sequence_length,
                                    trainable=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)
        l_lstm_lemma = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_lemma_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmax'))(l_lstm)
        preds_lemma = TimeDistributed(Dense(nb_output_class,activation='relu'))(l_lstm_lemma)
        #####################Dense ########################

        #added = Add()([preds_word, preds_lemma])
        #Sl_lstm = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_lemma_sequences)
        #main_lstm = Bidirectional(LSTM(units=number_layer, return_sequences=True,
        #                       recurrent_dropout=0.6))(SpatialDropout1D(0.3)(concatenate([preds_word, preds_lemma])))
        main_lstm = Bidirectional(LSTM(units=number_layer, return_sequences=True,
                               recurrent_dropout=0.1))(concatenate([preds_word, preds_lemma]))
        
        out = TimeDistributed(Dense(nb_output_class, activation="relu"))(main_lstm)

        ################# Output Layer#######################
        crf = CRF(nb_output_class)
        #model = Model(sequence_input,preds)
        model = Model(inputs=[sequence_word_input,sequence_lemma_input], outputs=crf(out))
        model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        #model.compile(loss=crf_loss,optimizer='rmsprop',metrics=[crf_viterbi_accuracy])
        #model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])

        sumary = model.summary()
        self.write_rapport_file(str(sumary),rapport_save_path)
        print("Model fitting - Bidirectional LSTM and CRF")
        return model
    def building_model_word_feature(self,voca2id_datasets, embedding_matrix,rapport_save_path,number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[0]
        lemma2ids = voca2id_datasets[1]
        #####################word###############################
        embedding_word_layer = Embedding(len(word2ids)+1, self.embedding_dim,
                                    weights=[embedding_matrix[0]],
                                    input_length=self.max_sequence_length,
                                    trainable=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        l_lstm_word = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(embedded_word_sequences)
        #preds = TimeDistributed(Dense(nb_output_class+1,activation='softmamerge_dropoutmerge_dropoutmerge_dropoutx'))(l_lstm)
        preds_word = TimeDistributed(Dense(nb_output_class,activation='relu'))(l_lstm_word)

        ################# Output Layer#######################
        crf = CRF(nb_output_class)
        #model = Model(sequence_input,preds)
        model = Model(inputs=[sequence_word_input], outputs=crf(preds_word))
        model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        #model.compile(loss=crf_loss,optimizer='rmsprop',metrics=[crf_viterbi_accuracy])
        
        #model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])

        sumary = model.summary()
        self.write_rapport_file(str(sumary),rapport_save_path)
        print("Model fitting - Bidirectional LSTM and CRF")
        return model

    # Three models based on the features: Word(w)
    def building_model_feature_w_by_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                  output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            Dropout(drop)(embedded_word_sequences))
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=sequence_word_input,
                      outputs=crf(preds_main))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM with building_model_feature_w_by_bilstm_crf")
        return model, callbacks_list
    def building_model_feature_w_by_2bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                  output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            Dropout(drop)(embedded_word_sequences))
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        # BiLSTM Layer
        l_lstm_main2 = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            Dropout(drop)(preds_main))
        preds_main2 = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main2))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=sequence_word_input,
                      outputs=crf(preds_main2))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM with building_model_feature_w_by_2bilstm_crf")
        return model, callbacks_list

    def building_model_feature_w_by_bilstm(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                              output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]

        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            Dropout(drop)(embedded_word_sequences))
        preds_main = TimeDistributed(Dense(nb_output_class, activation='softmax'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        model = Model(inputs=sequence_word_input, outputs=preds_main)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM with building_model_feature_w_by_bilstm")
        return model, callbacks_list

    # Three models based on the features: Word, Suffix(ws)
    def building_model_feature_ws_by_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                 output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        # Suffix Feature
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[-2]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_suffix_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_suffix_input], outputs=crf(preds_main))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_ws_by_bilstm_crf")
        return model, callbacks_list
    def building_model_feature_ws_by_2bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                 output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        # Suffix Feature
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[-2]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_suffix_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        # BiLSTM Layer
        l_lstm_main2 = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            preds_main)
        preds_main2 = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main2))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_suffix_input], outputs=crf(preds_main2))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_ws_by_2bilstm_crf")
        return model, callbacks_list
    def building_model_feature_ws_by_bilstm(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                             output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        # Suffix Feature
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[-2]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)

        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_suffix_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='softmax'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        model = Model(inputs=[sequence_word_input, sequence_suffix_input], outputs=preds_main)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM with building_model_feature_ws_by_bilstm")
        return model, callbacks_list

    def building_model_feature_ws_by_multi_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                       output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        # Suffix Feature
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[-2]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)


        # Level 1
        # Merge word + suffix
        embedding_word_suffix_sequence = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_suffix_sequences]))
        l_lstm_word_suffix = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedding_word_suffix_sequence)
        preds_word_suffix = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_suffix)


        # Merge Final
        embedded_merge = Dropout(drop)(concatenate(
            [embedded_word_sequences, embedded_suffix_sequences, preds_word_suffix]))
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_merge)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_main)

        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_suffix_input], outputs=crf(preds_main))
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # Checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_ws_by_multi_bilstm_crf")
        return model, callbacks_list
    # Three models based on the features: Word, Suffix, Cap (wsc)
    def building_model_feature_cwsc_by_bilstm_crf(self,voca2id_datasets, embedding_matrix,rapport_save_path,output_directory,number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        char2ids = voca2id_datasets[0]
        word2ids = voca2id_datasets[1]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids)+2, self.embedding_dim,
                                    weights=[embedding_matrix[1]],
                                    input_length=self.max_sequence_length,
                                    trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        #char feature
        max_word_length=30
        word_word_char_input = Input(shape=(self.max_sequence_length,max_word_length,))
        embedding_word_char_layer = TimeDistributed(Embedding(input_dim=len(char2ids), output_dim=max_word_length,\
                                        weights=[embedding_matrix[0]],
                                        input_length=max_word_length, mask_zero=True))(word_word_char_input)
        
        
        # character LSTM to get word encodings by characters
        word_char_encoding = TimeDistributed(Bidirectional(LSTM(units=number_layer, return_sequences=False,
                                        recurrent_dropout=0.5)))(embedding_word_char_layer) 

        # Suffix Feature
        embedding_suffix_layer = Embedding(input_dim = len(suffix2ids), output_dim = len(suffix2ids),\
                                    weights=[embedding_matrix[-2]],
                                    input_length=self.max_sequence_length,
                                    trainable=True,mask_zero=True)
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # Cap Feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim = len(cap2ids), output_dim = len(cap2ids),\
                                    weights=[embedding_matrix[-1]],
                                    input_length=self.max_sequence_length,
                                    trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge layer
        merge_layer = concatenate([embedded_word_sequences,word_char_encoding,embedded_suffix_sequences, embedded_cap_sequences])
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=0.1))(merge_layer)
        preds_main = TimeDistributed(Dense(number_layer,activation='softmax'))(l_lstm_main)

        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input,sequence_suffix_input, sequence_cap_input ], outputs=crf(preds_main))
        #model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        #
        model.compile(loss=crf_loss,optimizer='rmsprop',metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath=output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary),rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_cwsc_by_bilstm_crf")
        return model, callbacks_list

    def building_model_feature_wsc_by_bilstm_crf(self,voca2id_datasets, embedding_matrix,rapport_save_path,output_directory,number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop=0.3
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids)+2, self.embedding_dim,
                                    weights=[embedding_matrix[1]],
                                    input_length=self.max_sequence_length,
                                    trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        # Suffix Feature
        embedding_suffix_layer = Embedding(input_dim = len(suffix2ids), output_dim = len(suffix2ids),
                                    weights=[embedding_matrix[-2]],
                                    input_length=self.max_sequence_length,
                                    trainable=True,mask_zero=True)
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # Cap Feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim = len(cap2ids), output_dim = len(cap2ids),
                                    weights=[embedding_matrix[-1]],
                                    input_length=self.max_sequence_length,
                                    trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge layer
        merge_layer = Dropout(drop)(concatenate([embedded_word_sequences,embedded_suffix_sequences, embedded_cap_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=drop))(merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class,activation='relu'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input,sequence_suffix_input, sequence_cap_input ], outputs=crf(preds_main))
        #model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss,optimizer='rmsprop',metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath=output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True, mode='max')
        #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary),rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wsc_by_bilstm_crf")
        return model, callbacks_list
    def building_model_feature_wsc_by_2bilstm_crf(self,voca2id_datasets, embedding_matrix,rapport_save_path,output_directory,number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop=0.3
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids)+2, self.embedding_dim,
                                    weights=[embedding_matrix[1]],
                                    input_length=self.max_sequence_length,
                                    trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        # Suffix Feature
        embedding_suffix_layer = Embedding(input_dim = len(suffix2ids), output_dim = len(suffix2ids),
                                    weights=[embedding_matrix[-2]],
                                    input_length=self.max_sequence_length,
                                    trainable=True,mask_zero=True)
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # Cap Feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim = len(cap2ids), output_dim = len(cap2ids),
                                    weights=[embedding_matrix[-1]],
                                    input_length=self.max_sequence_length,
                                    trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge layer
        merge_layer = Dropout(drop)(concatenate([embedded_word_sequences,embedded_suffix_sequences, embedded_cap_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=drop))(merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class,activation='relu'))(Dropout(drop)(l_lstm_main))
        # BiLSTM Layer
        l_lstm_main2 = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=drop))(preds_main)
        preds_main2 = TimeDistributed(Dense(nb_output_class,activation='relu'))(Dropout(drop)(l_lstm_main2))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input,sequence_suffix_input, sequence_cap_input ], outputs=crf(preds_main2))
        #model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss,optimizer='rmsprop',metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath=output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True, mode='max')
        #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary),rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wsc_by_2bilstm_crf")
        return model, callbacks_list
    def building_model_feature_wsc_by_bilstm(self,voca2id_datasets, embedding_matrix,rapport_save_path,output_directory,number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop=0.3
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids)+2, self.embedding_dim,
                                    weights=[embedding_matrix[1]],
                                    input_length=self.max_sequence_length,
                                    trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        # Suffix Feature
        embedding_suffix_layer = Embedding(input_dim = len(suffix2ids), output_dim = len(suffix2ids),\
                                    weights=[embedding_matrix[-2]],
                                    input_length=self.max_sequence_length,
                                    trainable=True,mask_zero=True)
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # Cap Feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim = len(cap2ids), output_dim = len(cap2ids),\
                                    weights=[embedding_matrix[-1]],
                                    input_length=self.max_sequence_length,
                                    trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge layer
        merge_layer = Dropout(drop)(concatenate([embedded_word_sequences,embedded_suffix_sequences, embedded_cap_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer,return_sequences=True,recurrent_dropout=drop))(merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class,activation='softmax'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        model = Model(inputs=[sequence_word_input,sequence_suffix_input, sequence_cap_input ], outputs=preds_main)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        # checkpoint
        filepath=output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary),rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM with building_model_feature_wsc_by_bilstm")
        return model, callbacks_list
    
    def building_model_feature_wsc_by_multi_bilstm_crf2(self,voca2id_datasets, embedding_matrix,rapport_save_path,output_directory,number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop=0.3
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = Dropout(drop)(embedding_word_layer(sequence_word_input))
        l_lstm_word = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(embedded_word_sequences)
        preds_word = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word)

        # Suffix Feature
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[-2]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # Merge 1
        embedded_suffix_sequences_merge = Dropout(drop)(concatenate([embedded_suffix_sequences,preds_word]))
        l_lstm_suffix = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(embedded_suffix_sequences_merge)
        preds_suffix = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_suffix)
        # Cap Feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                        weights=[embedding_matrix[-1]],
                                        input_length=self.max_sequence_length,
                                        trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge 2
        embedded_cap_sequences_merge = Dropout(drop)(concatenate([embedded_cap_sequences,preds_word, preds_suffix]))
        # BiLSTM Layer
        l_lstm_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(embedded_cap_sequences_merge)
        
        preds_cap = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_cap))

        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_suffix_input, sequence_cap_input], outputs=crf(preds_cap))
        model.compile(loss=crf_loss,optimizer='rmsprop',metrics=[crf_viterbi_accuracy])

        # Checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True, mode='max')
        #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wsc_by_multi_bilstm_crf2")
        return model, callbacks_list


    def building_model_feature_wsc_by_multi_bilstm_crf3(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                       output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        # Suffix Feature
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[-2]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # Cap Feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                        weights=[embedding_matrix[-1]],
                                        input_length=self.max_sequence_length,
                                        trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge 0
        embedded_sequences_merge00 = Dropout(drop)(concatenate([embedded_word_sequences, embedded_suffix_sequences,embedded_cap_sequences]))

        l_lstm_all = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(embedded_sequences_merge00)
        preds_all= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_all)

        l_lstm_word = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_sequences)
        preds_word = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word)
        # Merge 1
        embedded_sequences_merge01 = Dropout(drop)(concatenate([preds_all, preds_word]))
        l_lstm_suffix = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_suffix_sequences)
        preds_suffix = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_suffix)
        l_lstm_all2 = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_sequences_merge01)

        preds_all2 = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_all2)
        embedded_sequences_merge02 = Dropout(drop)(concatenate([preds_all,preds_all2, preds_suffix]))
        # Merge 2
        l_lstm_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_cap_sequences)
        preds_cap = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_cap)
        l_lstm_all3 = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_sequences_merge02)
        preds_all3 = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_all3)
        embedded_sequences_merge03 = Dropout(drop)(concatenate([preds_all, preds_all2, preds_all3, preds_cap]))

        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_sequences_merge03)
        preds_main= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_main)

        crf = CRF(nb_output_class)
        # Output Layer
        model = Model(inputs=[sequence_word_input, sequence_suffix_input, sequence_cap_input], outputs=crf(preds_main))
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])
        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM with building_model_feature_wsc_by_multi_bilstm_crf3")
        return model, callbacks_list

    def building_model_feature_wsc_by_multi_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                        output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)
        # Suffix Feature
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[-2]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # Cap Feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                        weights=[embedding_matrix[-1]],
                                        input_length=self.max_sequence_length,
                                        trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)

        #Level 1
        # Merge word + suffix
        embedding_word_suffix_sequence = Dropout(drop)(concatenate([embedded_word_sequences,embedded_suffix_sequences]))
        l_lstm_word_suffix = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedding_word_suffix_sequence)
        preds_word_suffix= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_suffix)

        # Merge word + cap
        embedding_word_cap_sequence = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_cap_sequences]))
        l_lstm_word_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedding_word_cap_sequence)
        preds_word_cap = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_cap)

        # Merge suffix + cap
        embedding_suffix_cap_sequence = Dropout(drop)(
            concatenate([embedded_suffix_sequences, embedded_cap_sequences]))
        l_lstm_suffix_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedding_suffix_cap_sequence)
        preds_suffix_cap = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_suffix_cap)

        # Merge Final
        embedded_merge = Dropout(drop)(concatenate([embedded_word_sequences,embedded_suffix_sequences,embedded_cap_sequences,preds_word_suffix,preds_word_cap,preds_suffix_cap]))
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_merge)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_main)

        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_suffix_input, sequence_cap_input], outputs=crf(preds_main))
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # Checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wsc_by_multi_bilstm_crf")
        return model, callbacks_list

    # Three models based on the features: Word, Lemma, Suffix, Cap (wlsc)
    def building_model_feature_wlsc_by_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                 output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[5]
        cap2ids = voca2id_datasets[6]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # suffix feature
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[5]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # cap feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                        weights=[embedding_matrix[6]],
                                        input_length=self.max_sequence_length,
                                        trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_lemma_sequences, embedded_suffix_sequences,embedded_cap_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_suffix_input,sequence_cap_input], outputs=crf(preds_main))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wlsc_by_bilstm_crf")
        return model, callbacks_list
    def building_model_feature_wlsc_by_2bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                 output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[5]
        cap2ids = voca2id_datasets[6]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # suffix feature
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[5]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # cap feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                        weights=[embedding_matrix[6]],
                                        input_length=self.max_sequence_length,
                                        trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_lemma_sequences, embedded_suffix_sequences,embedded_cap_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        # BiLSTM Layer
        l_lstm_main2 = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            preds_main)
        preds_main2 = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main2))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_suffix_input,sequence_cap_input], outputs=crf(preds_main2))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wlsc_by_2bilstm_crf")
        return model, callbacks_list
    def building_model_feature_wlsc_by_bilstm(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                             output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[5]
        cap2ids = voca2id_datasets[6]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # suffix feature
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[5]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # cap feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                        weights=[embedding_matrix[6]],
                                        input_length=self.max_sequence_length,
                                        trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_lemma_sequences,embedded_suffix_sequences,embedded_cap_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='softmax'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_suffix_input,sequence_cap_input], outputs=preds_main)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM with building_model_feature_wlsc_by_bilstm")
        return model, callbacks_list

    def building_model_feature_wlsc_by_multi_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                        output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[5]
        cap2ids = voca2id_datasets[6]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop=0.3
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # suffix feature
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[5]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # cap feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                           weights=[embedding_matrix[6]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)

        # Merge Word, lemma
        embedded_word_lemma_sequences = Dropout(drop)(concatenate([embedded_word_sequences, embedded_lemma_sequences]))
        l_lstm_word_lemma = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_lemma_sequences)
        preds_word_lemma = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_lemma)

        # Merge Word, suffix
        embedded_word_suffix_sequences = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_suffix_sequences]))
        l_lstm_word_suffix = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_suffix_sequences)
        preds_word_suffix = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_suffix)

        # Merge Word, cap
        embedded_word_cap_sequences = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_cap_sequences]))
        l_lstm_word_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_cap_sequences)
        preds_word_cap = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_cap)

        # Merge lemma, suffix
        embedded_lemma_suffix_sequences = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_suffix_sequences]))
        l_lstm_lemma_suffix = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_lemma_suffix_sequences)
        preds_lemma_suffix = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_lemma_suffix)

        # Merge lemma, cap
        embedded_lemma_cap_sequences = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_cap_sequences]))
        l_lstm_lemma_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_lemma_cap_sequences)
        preds_lemma_cap = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_lemma_cap)

        # Merge suffix, cap
        embedded_suffix_cap_sequences = Dropout(drop)(
            concatenate([embedded_suffix_sequences, embedded_cap_sequences]))
        l_lstm_suffix_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_suffix_cap_sequences)
        preds_suffix_cap= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_suffix_cap)

        # Merge all
        embedded_all = Dropout(drop)(concatenate([embedded_word_sequences, embedded_lemma_sequences,embedded_suffix_sequences,embedded_cap_sequences,
                                                  preds_word_lemma, preds_word_suffix, preds_word_cap,preds_lemma_suffix,
                                                  preds_lemma_cap,preds_suffix_cap] ))
        l_lstm_all = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_all)
        preds_all= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_all)


        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_suffix_input,sequence_cap_input],
                      outputs=crf(preds_all))
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # Checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wlsc_by_multi_bilstm_crf")
        return model, callbacks_list

    # Three models based on the features: Word, Lemma, Postag, Suffix, Cap (wlpsc)
    def building_model_feature_wlpsc_by_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                 output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[5]
        cap2ids = voca2id_datasets[6]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)

        # suffix feature
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[5]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # cap feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                        weights=[embedding_matrix[6]],
                                        input_length=self.max_sequence_length,
                                        trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_lemma_sequences, embedded_postag_sequences, embedded_suffix_sequences,embedded_cap_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_postag_input, sequence_suffix_input,sequence_cap_input], outputs=crf(preds_main))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wlpsc_by_bilstm_crf")
        return model, callbacks_list
    def building_model_feature_wlpsc_by_2bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                 output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[5]
        cap2ids = voca2id_datasets[6]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)

        # suffix feature
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[5]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # cap feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                        weights=[embedding_matrix[6]],
                                        input_length=self.max_sequence_length,
                                        trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_lemma_sequences, embedded_postag_sequences, embedded_suffix_sequences,embedded_cap_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        # BiLSTM Layer
        l_lstm_main2 = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            preds_main)
        preds_main2 = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main2))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_postag_input, sequence_suffix_input,sequence_cap_input], outputs=crf(preds_main2))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wlpsc_by_2bilstm_crf")
        return model, callbacks_list
    def building_model_feature_wlpsc_by_bilstm(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                             output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[5]
        cap2ids = voca2id_datasets[6]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)

        # suffix feature
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[5]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # cap feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                        weights=[embedding_matrix[6]],
                                        input_length=self.max_sequence_length,
                                        trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_lemma_sequences,embedded_postag_sequences,embedded_suffix_sequences,embedded_cap_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='softmax'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_postag_input, sequence_suffix_input,sequence_cap_input], outputs=preds_main)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM with building_model_feature_wlpsc_by_bilstm")
        return model, callbacks_list

    def building_model_feature_wlpsc_by_multi_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                        output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[5]
        cap2ids = voca2id_datasets[6]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop=0.3
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)

        # suffix feature
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[5]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # cap feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                           weights=[embedding_matrix[6]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)

        # Merge Word, lemma
        embedded_word_lemma_sequences = Dropout(drop)(concatenate([embedded_word_sequences, embedded_lemma_sequences]))
        l_lstm_word_lemma = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_lemma_sequences)
        preds_word_lemma = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_lemma)

        # Merge Word, Postag
        embedded_word_postag_sequences = Dropout(drop)(concatenate([embedded_word_sequences, embedded_postag_sequences]))
        l_lstm_word_postag = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_postag_sequences)
        preds_word_postag = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_postag)

        # Merge Word, suffix
        embedded_word_suffix_sequences = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_suffix_sequences]))
        l_lstm_word_suffix = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_suffix_sequences)
        preds_word_suffix = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_suffix)

        # Merge Word, cap
        embedded_word_cap_sequences = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_cap_sequences]))
        l_lstm_word_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_cap_sequences)
        preds_word_cap = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_cap)

        # Merge lemma, postag
        embedded_lemma_postag_sequences = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_postag_sequences]))
        l_lstm_lemma_postag = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_lemma_postag_sequences)
        preds_lemma_postag = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_lemma_postag)

        # Merge lemma, suffix
        embedded_lemma_suffix_sequences = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_suffix_sequences]))
        l_lstm_lemma_suffix = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_lemma_suffix_sequences)
        preds_lemma_suffix = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_lemma_suffix)

        # Merge lemma, cap
        embedded_lemma_cap_sequences = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_cap_sequences]))
        l_lstm_lemma_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_lemma_cap_sequences)
        preds_lemma_cap = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_lemma_cap)

        # Merge postag, suffix
        embedded_postag_suffix_sequences = Dropout(drop)(
            concatenate([embedded_postag_sequences, embedded_suffix_sequences]))
        l_lstm_postag_suffix = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_postag_suffix_sequences)
        preds_postag_suffix = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_postag_suffix)

        # Merge postag, cap
        embedded_postag_cap_sequences = Dropout(drop)(
            concatenate([embedded_postag_sequences, embedded_cap_sequences]))
        l_lstm_postag_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_postag_cap_sequences)
        preds_postag_cap= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_postag_cap)

        # Merge suffix, cap
        embedded_suffix_cap_sequences = Dropout(drop)(
            concatenate([embedded_suffix_sequences, embedded_cap_sequences]))
        l_lstm_suffix_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_suffix_cap_sequences)
        preds_suffix_cap= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_suffix_cap)

        # Merge all
        embedded_all = Dropout(drop)(concatenate([embedded_word_sequences, embedded_lemma_sequences,embedded_postag_sequences,embedded_suffix_sequences,embedded_cap_sequences,
                                                  preds_word_lemma, preds_word_postag, preds_word_suffix, preds_word_cap,preds_lemma_postag,preds_lemma_suffix,
                                                  preds_lemma_cap, preds_postag_suffix,preds_postag_cap,preds_suffix_cap] ))
        l_lstm_all = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_all)
        preds_all= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_all)


        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_postag_input, sequence_suffix_input,sequence_cap_input],
                      outputs=crf(preds_all))
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # Checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wlpsc_by_multi_bilstm_crf")
        return model, callbacks_list

    # Three models based on the features: Word, Lemma, Postag, Deprel, Suffix, Cap (wlpdsc)
    def building_model_feature_wlpdsc_by_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                 output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[5]
        cap2ids = voca2id_datasets[6]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # deprel feature
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_deprel_layer = Embedding(input_dim=len(deprel2ids), output_dim=len(deprel2ids),
                                           weights=[embedding_matrix[4]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)
        # suffix feature
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[5]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # cap feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                        weights=[embedding_matrix[6]],
                                        input_length=self.max_sequence_length,
                                        trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_lemma_sequences, embedded_postag_sequences,embedded_deprel_sequences, embedded_suffix_sequences,embedded_cap_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_postag_input,sequence_deprel_input, sequence_suffix_input,sequence_cap_input], outputs=crf(preds_main))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wlpdsc_by_bilstm_crf")
        return model, callbacks_list
    def building_model_feature_wlpdsc_by_2bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                 output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[5]
        cap2ids = voca2id_datasets[6]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # deprel feature
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_deprel_layer = Embedding(input_dim=len(deprel2ids), output_dim=len(deprel2ids),
                                           weights=[embedding_matrix[4]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)
        # suffix feature
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[5]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # cap feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                        weights=[embedding_matrix[6]],
                                        input_length=self.max_sequence_length,
                                        trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_lemma_sequences, embedded_postag_sequences,embedded_deprel_sequences, embedded_suffix_sequences,embedded_cap_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        # BiLSTM Layer
        l_lstm_main2 = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            preds_main)
        preds_main2 = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main2))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_postag_input,sequence_deprel_input, sequence_suffix_input,sequence_cap_input], outputs=crf(preds_main2))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wlpdsc_by_2bilstm_crf")
        return model, callbacks_list
    def building_model_feature_wlpdsc_by_bilstm(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                             output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[5]
        cap2ids = voca2id_datasets[6]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # deprel feature
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_deprel_layer = Embedding(input_dim=len(deprel2ids), output_dim=len(deprel2ids),
                                           weights=[embedding_matrix[4]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)
        # suffix feature
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[5]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # cap feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                        weights=[embedding_matrix[6]],
                                        input_length=self.max_sequence_length,
                                        trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_lemma_sequences,embedded_postag_sequences, embedded_deprel_sequences,embedded_suffix_sequences,embedded_cap_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='softmax'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_postag_input,sequence_deprel_input, sequence_suffix_input,sequence_cap_input], outputs=preds_main)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM with building_model_feature_wlpdsc_by_bilstm")
        return model, callbacks_list

    def building_model_feature_wlpdsc_by_multi_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                        output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[5]
        cap2ids = voca2id_datasets[6]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop=0.3
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # deprel feature
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_deprel_layer = Embedding(input_dim=len(deprel2ids), output_dim=len(deprel2ids),
                                           weights=[embedding_matrix[4]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)
        # suffix feature
        sequence_suffix_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_suffix_layer = Embedding(input_dim=len(suffix2ids), output_dim=len(suffix2ids),
                                           weights=[embedding_matrix[5]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_suffix_sequences = embedding_suffix_layer(sequence_suffix_input)
        # cap feature
        sequence_cap_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_cap_layer = Embedding(input_dim=len(cap2ids), output_dim=len(cap2ids),
                                           weights=[embedding_matrix[6]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_cap_sequences = embedding_cap_layer(sequence_cap_input)

        # Merge Word, lemma
        embedded_word_lemma_sequences = Dropout(drop)(concatenate([embedded_word_sequences, embedded_lemma_sequences]))
        l_lstm_word_lemma = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_lemma_sequences)
        preds_word_lemma = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_lemma)

        # Merge Word, Postag
        embedded_word_postag_sequences = Dropout(drop)(concatenate([embedded_word_sequences, embedded_postag_sequences]))
        l_lstm_word_postag = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_postag_sequences)
        preds_word_postag = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_postag)

        # Merge Word, Deprel
        embedded_word_deprel_sequences = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_deprel_sequences]))
        l_lstm_word_deprel = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_deprel_sequences)
        preds_word_deprel = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_deprel)

        # Merge Word, suffix
        embedded_word_suffix_sequences = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_suffix_sequences]))
        l_lstm_word_suffix = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_suffix_sequences)
        preds_word_suffix = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_suffix)

        # Merge Word, cap
        embedded_word_cap_sequences = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_cap_sequences]))
        l_lstm_word_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_cap_sequences)
        preds_word_cap = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_cap)

        # Merge lemma, postag
        embedded_lemma_postag_sequences = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_postag_sequences]))
        l_lstm_lemma_postag = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_lemma_postag_sequences)
        preds_lemma_postag = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_lemma_postag)

        # Merge lemma, deprel
        embedded_lemma_deprel_sequences = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_deprel_sequences]))
        l_lstm_lemma_deprel = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_lemma_deprel_sequences)
        preds_lemma_deprel = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_lemma_deprel)

        # Merge lemma, suffix
        embedded_lemma_suffix_sequences = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_suffix_sequences]))
        l_lstm_lemma_suffix = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_lemma_suffix_sequences)
        preds_lemma_suffix = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_lemma_suffix)

        # Merge lemma, cap
        embedded_lemma_cap_sequences = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_cap_sequences]))
        l_lstm_lemma_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_lemma_cap_sequences)
        preds_lemma_cap = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_lemma_cap)

        # Merge postag, deprel
        embedded_postag_deprel_sequences = Dropout(drop)(
            concatenate([embedded_postag_sequences, embedded_deprel_sequences]))
        l_lstm_postag_deprel = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_postag_deprel_sequences)
        preds_postag_deprel = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_postag_deprel)

        # Merge postag, suffix
        embedded_postag_suffix_sequences = Dropout(drop)(
            concatenate([embedded_postag_sequences, embedded_suffix_sequences]))
        l_lstm_postag_suffix = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_postag_suffix_sequences)
        preds_postag_suffix = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_postag_suffix)

        # Merge postag, cap
        embedded_postag_cap_sequences = Dropout(drop)(
            concatenate([embedded_postag_sequences, embedded_cap_sequences]))
        l_lstm_postag_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_postag_cap_sequences)
        preds_postag_cap= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_postag_cap)

        # Merge deprel, suffix
        embedded_deprel_suffix_sequences = Dropout(drop)(
            concatenate([embedded_deprel_sequences, embedded_suffix_sequences]))
        l_lstm_deprel_suffix= Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_deprel_suffix_sequences)
        preds_deprel_suffix = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_deprel_suffix)

        # Merge deprel, cap
        embedded_deprel_cap_sequences = Dropout(drop)(
            concatenate([embedded_deprel_sequences, embedded_cap_sequences]))
        l_lstm_deprel_cap= Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_deprel_cap_sequences)
        preds_deprel_cap= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_deprel_cap)

        # Merge suffix, cap
        embedded_suffix_cap_sequences = Dropout(drop)(
            concatenate([embedded_suffix_sequences, embedded_cap_sequences]))
        l_lstm_suffix_cap = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_suffix_cap_sequences)
        preds_suffix_cap= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_suffix_cap)

        # Merge all
        embedded_all = Dropout(drop)(concatenate([embedded_word_sequences, embedded_lemma_sequences,embedded_postag_sequences,embedded_deprel_sequences,embedded_suffix_sequences,embedded_cap_sequences,
                                                  preds_word_lemma, preds_word_postag,preds_word_deprel, preds_word_suffix, preds_word_cap,preds_lemma_postag,preds_lemma_deprel,preds_lemma_suffix,
                                                  preds_lemma_cap,preds_postag_deprel, preds_postag_suffix,preds_postag_cap,preds_deprel_suffix,preds_deprel_cap,preds_suffix_cap] ))
        l_lstm_all = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_all)
        preds_all= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_all)


        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_postag_input, sequence_deprel_input, sequence_suffix_input,sequence_cap_input],
                      outputs=crf(preds_all))
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # Checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wlpdsc_by_multi_bilstm_crf")
        return model, callbacks_list

    # Three models based on the features: Word, Lemma, Postag, Deprel (wlpd)
    def building_model_feature_wlpd_by_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                 output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # deprel feature
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_deprel_layer = Embedding(input_dim=len(deprel2ids), output_dim=len(deprel2ids),
                                           weights=[embedding_matrix[4]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_lemma_sequences, embedded_postag_sequences,embedded_deprel_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_postag_input,sequence_deprel_input], outputs=crf(preds_main))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wlpd_by_bilstm_crf")
        return model, callbacks_list
    def building_model_feature_wlpd_by_2bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                 output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # deprel feature
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_deprel_layer = Embedding(input_dim=len(deprel2ids), output_dim=len(deprel2ids),
                                           weights=[embedding_matrix[4]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_lemma_sequences, embedded_postag_sequences,embedded_deprel_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        # BiLSTM Layer
        l_lstm_main2 = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            preds_main)
        preds_main2 = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main2))
        
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_postag_input,sequence_deprel_input], outputs=crf(preds_main2))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wlpd_by_2bilstm_crf")
        return model, callbacks_list
    def building_model_feature_wlpd_by_bilstm(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                             output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # deprel feature
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_deprel_layer = Embedding(input_dim=len(deprel2ids), output_dim=len(deprel2ids),
                                           weights=[embedding_matrix[4]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_lemma_sequences,embedded_postag_sequences, embedded_deprel_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='softmax'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_postag_input,sequence_deprel_input], outputs=preds_main)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM with building_model_feature_wlpd_by_bilstm")
        return model, callbacks_list

    def building_model_feature_wlpd_by_multi_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                        output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        word2ids = voca2id_datasets[1]
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop=0.3
        # Input and Embedding Layer
        # Word feature
        embedding_word_layer = Embedding(len(word2ids) + 2, self.embedding_dim,
                                         weights=[embedding_matrix[1]],
                                         input_length=self.max_sequence_length,
                                         trainable=True,mask_zero=True)
        sequence_word_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_word_sequences = embedding_word_layer(sequence_word_input)

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # deprel feature
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_deprel_layer = Embedding(input_dim=len(deprel2ids), output_dim=len(deprel2ids),
                                           weights=[embedding_matrix[4]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)

        # Merge Word, lemma
        embedded_word_lemma_sequences = Dropout(drop)(concatenate([embedded_word_sequences, embedded_lemma_sequences]))
        l_lstm_word_lemma = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_lemma_sequences)
        preds_word_lemma = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_lemma)

        # Merge Word, Postag
        embedded_word_postag_sequences = Dropout(drop)(concatenate([embedded_word_sequences, embedded_postag_sequences]))
        l_lstm_word_postag = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_postag_sequences)
        preds_word_postag = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_postag)

        # Merge Word, Deprel
        embedded_word_deprel_sequences = Dropout(drop)(
            concatenate([embedded_word_sequences, embedded_deprel_sequences]))
        l_lstm_word_deprel = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_word_deprel_sequences)
        preds_word_deprel = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_word_deprel)

        # Merge lemma, postag
        embedded_lemma_postag_sequences = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_postag_sequences]))
        l_lstm_lemma_postag = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_lemma_postag_sequences)
        preds_lemma_postag = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_lemma_postag)

        # Merge lemma, deprel
        embedded_lemma_deprel_sequences = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_deprel_sequences]))
        l_lstm_lemma_deprel = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_lemma_deprel_sequences)
        preds_lemma_deprel = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_lemma_deprel)

        # Merge postag, deprel
        embedded_postag_deprel_sequences = Dropout(drop)(
            concatenate([embedded_postag_sequences, embedded_deprel_sequences]))
        l_lstm_postag_deprel = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_postag_deprel_sequences)
        preds_postag_deprel = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_postag_deprel)

        # Merge all
        embedded_all = Dropout(drop)(concatenate([embedded_word_sequences, embedded_lemma_sequences,embedded_postag_sequences,embedded_postag_sequences,
                                                  preds_word_lemma, preds_word_postag,preds_word_deprel,preds_lemma_postag,preds_lemma_deprel,preds_postag_deprel]))
        l_lstm_all = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_all)
        preds_all= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_all)


        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_word_input, sequence_lemma_input, sequence_postag_input, sequence_deprel_input],
                      outputs=crf(preds_all))
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # Checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_wlpd_by_multi_bilstm_crf")
        return model, callbacks_list
    ###
    # Three models based on the features: Lemma (l)
    def building_model_feature_l_by_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                  output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        lemma2ids = voca2id_datasets[2]
        drop = 0.5
        # Input and Embedding Layer
        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True, mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(Dropout(drop)(embedded_lemma_sequences))
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_lemma_input],
                      outputs=crf(preds_main))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_l_by_bilstm_crf")
        return model, callbacks_list
    def building_model_feature_l_by_2bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                  output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        lemma2ids = voca2id_datasets[2]
        drop = 0.5
        # Input and Embedding Layer
        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True, mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(Dropout(drop)(embedded_lemma_sequences))
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
         # BiLSTM Layer
        l_lstm_main2 = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(Dropout(drop)(preds_main))
        preds_main2 = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main2))
        
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_lemma_input],
                      outputs=crf(preds_main2))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_l_by_2bilstm_crf")
        return model, callbacks_list
    
    def building_model_feature_l_by_bilstm(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                              output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        lemma2ids = voca2id_datasets[2]
        drop = 0.5
        # Input and Embedding Layer
        # Word feature

        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True, mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            Dropout(drop)(embedded_lemma_sequences))
        preds_main = TimeDistributed(Dense(nb_output_class, activation='softmax'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        model = Model(inputs=[sequence_lemma_input],
                      outputs=preds_main)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM with building_model_feature_l_by_bilstm")
        return model, callbacks_list
    # Three models based on the features: Lemma, Postag (lp)
    def building_model_feature_lp_by_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                 output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        drop = 0.5
        # Input and Embedding Layer
        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_postag_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_lemma_input, sequence_postag_input], outputs=crf(preds_main))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_lp_by_bilstm_crf")
        return model, callbacks_list

    def building_model_feature_lp_by_2bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                 output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        drop = 0.5
        # Input and Embedding Layer
        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_postag_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        l_lstm_main2 = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            preds_main)
        preds_main2 = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main2))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_lemma_input, sequence_postag_input], outputs=crf(preds_main2))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_lp_by_2bilstm_crf")
        return model, callbacks_list

    def building_model_feature_lp_by_bilstm(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                             output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        drop = 0.5
        # Input and Embedding Layer
        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_lemma_sequences,embedded_postag_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='softmax'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        model = Model(inputs=[sequence_lemma_input, sequence_postag_input], outputs=preds_main)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM with building_model_feature_lp_by_bilstm")
        return model, callbacks_list

    def building_model_feature_lp_by_multi_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                        output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        drop=0.3
        # Input and Embedding Layer
        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        
        # Merge lemma, postag
        embedded_lemma_postag_sequences = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_postag_sequences]))
        l_lstm_lemma_postag = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_lemma_postag_sequences)
        preds_lemma_postag = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_lemma_postag)

        # Merge all
        embedded_all = Dropout(drop)(concatenate([embedded_lemma_sequences,embedded_postag_sequences,
                                                  preds_lemma_postag]))
        l_lstm_all = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_all)
        preds_all= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_all)


        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_lemma_input, sequence_postag_input],
                      outputs=crf(preds_all))
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # Checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_lp_by_multi_bilstm_crf")
        return model, callbacks_list
    ###
    # Three models based on the features: Lemma, Postag, Deprel (lpd)
    def building_model_feature_lpd_by_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                 output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # deprel feature
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_deprel_layer = Embedding(input_dim=len(deprel2ids), output_dim=len(deprel2ids),
                                           weights=[embedding_matrix[4]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_postag_sequences,embedded_deprel_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_lemma_input, sequence_postag_input,sequence_deprel_input], outputs=crf(preds_main))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_lpd_by_bilstm_crf")
        return model, callbacks_list
    def building_model_feature_lpd_by_2bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                 output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # deprel feature
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_deprel_layer = Embedding(input_dim=len(deprel2ids), output_dim=len(deprel2ids),
                                           weights=[embedding_matrix[4]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_postag_sequences,embedded_deprel_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main))
         # BiLSTM Layer
        l_lstm_main2 = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            preds_main)
        preds_main2 = TimeDistributed(Dense(nb_output_class, activation='relu'))(Dropout(drop)(l_lstm_main2))
        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_lemma_input, sequence_postag_input,sequence_deprel_input], outputs=crf(preds_main2))
        # model.compile(loss=crf.loss_function,optimizer='rmsprop',metrics=[crf.accuracy])
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_lpd_by_2bilstm_crf")
        return model, callbacks_list
    def building_model_feature_lpd_by_bilstm(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                             output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop = 0.5
        # Input and Embedding Layer
        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # deprel feature
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_deprel_layer = Embedding(input_dim=len(deprel2ids), output_dim=len(deprel2ids),
                                           weights=[embedding_matrix[4]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)
        # Merge layer
        merge_layer = Dropout(drop)(
            concatenate([embedded_lemma_sequences,embedded_postag_sequences, embedded_deprel_sequences]))
        # BiLSTM Layer
        l_lstm_main = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            merge_layer)
        preds_main = TimeDistributed(Dense(nb_output_class, activation='softmax'))(Dropout(drop)(l_lstm_main))
        # Output Layer
        model = Model(inputs=[sequence_lemma_input, sequence_postag_input,sequence_deprel_input], outputs=preds_main)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        # checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM with building_model_feature_lpd_by_bilstm")
        return model, callbacks_list

    def building_model_feature_lpd_by_multi_bilstm_crf(self, voca2id_datasets, embedding_matrix, rapport_save_path,
                                                        output_directory, number_layer=64):
        nb_output_class = len(voca2id_datasets[-1])
        lemma2ids = voca2id_datasets[2]
        postag2ids = voca2id_datasets[3]
        deprel2ids = voca2id_datasets[4]
        suffix2ids = voca2id_datasets[-3]
        cap2ids = voca2id_datasets[-2]
        drop=0.3
        # Input and Embedding Layer
        # Lemma Feature
        embedding_lemma_layer = Embedding(input_dim=len(lemma2ids) + 2, output_dim=self.embedding_dim,
                                          weights=[embedding_matrix[2]],
                                          input_length=self.max_sequence_length,
                                          trainable=True,mask_zero=True)
        sequence_lemma_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_lemma_sequences = embedding_lemma_layer(sequence_lemma_input)

        # Postag Feature
        sequence_postag_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_postag_layer = Embedding(input_dim=len(postag2ids), output_dim=len(postag2ids),
                                           weights=[embedding_matrix[3]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_postag_sequences = embedding_postag_layer(sequence_postag_input)
        # deprel feature
        sequence_deprel_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_deprel_layer = Embedding(input_dim=len(deprel2ids), output_dim=len(deprel2ids),
                                           weights=[embedding_matrix[4]],
                                           input_length=self.max_sequence_length,
                                           trainable=True,mask_zero=True)
        embedded_deprel_sequences = embedding_deprel_layer(sequence_deprel_input)

        # Merge lemma, postag
        embedded_lemma_postag_sequences = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_postag_sequences]))
        l_lstm_lemma_postag = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_lemma_postag_sequences)
        preds_lemma_postag = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_lemma_postag)

        # Merge lemma, deprel
        embedded_lemma_deprel_sequences = Dropout(drop)(
            concatenate([embedded_lemma_sequences, embedded_deprel_sequences]))
        l_lstm_lemma_deprel = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_lemma_deprel_sequences)
        preds_lemma_deprel = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_lemma_deprel)

        # Merge postag, deprel
        embedded_postag_deprel_sequences = Dropout(drop)(
            concatenate([embedded_postag_sequences, embedded_deprel_sequences]))
        l_lstm_postag_deprel = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_postag_deprel_sequences)
        preds_postag_deprel = TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_postag_deprel)

        # Merge all
        embedded_all = Dropout(drop)(concatenate([embedded_lemma_sequences,embedded_postag_sequences,embedded_postag_sequences,
                                                   preds_lemma_postag,preds_lemma_deprel,preds_postag_deprel]))
        l_lstm_all = Bidirectional(LSTM(units=number_layer, return_sequences=True, recurrent_dropout=drop))(
            embedded_all)
        preds_all= TimeDistributed(Dense(nb_output_class, activation='relu'))(l_lstm_all)


        # Output Layer
        crf = CRF(nb_output_class)
        model = Model(inputs=[sequence_lemma_input, sequence_postag_input, sequence_deprel_input],
                      outputs=crf(preds_all))
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])

        # Checkpoint
        filepath = output_directory + "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]
        sumary = model.summary()
        self.write_rapport_file(str(sumary), rapport_save_path)
        print("Model Fitting - Bidirectional-LSTM-CRF with building_model_feature_lpd_by_multi_bilstm_crf")
        return model, callbacks_list
    ###

    def fitting_data_model(self,model,datasets_train, labels_train,datasets_dev, labels_dev, voca2id_datasets, embedding_matrix,model_save_path,callbacks_list,number_epoch=0):
        
        nb_output_class=len(voca2id_datasets[-1])
        print("nb_output_class",nb_output_class)
        print("datasets_train",len(datasets_train))
        n_train_samples = datasets_train[0].shape[0]
        n_dev_samples = datasets_dev[0].shape[0]
        train_generator =self.generator(all_X=datasets_train, all_Y=labels_train, n_classes=nb_output_class ,batch_size=self.batch_size)
        dev_generator = self.generator(all_X=datasets_dev, all_Y=labels_dev, n_classes=nb_output_class ,batch_size=self.batch_size)
        
        history = model.fit_generator(train_generator,
                     steps_per_epoch=n_train_samples//self.batch_size,
                     validation_data=dev_generator,
                     validation_steps=n_dev_samples//self.batch_size,
                     epochs=number_epoch, callbacks=callbacks_list,
                     verbose=1,
                     workers=30, use_multiprocessing=True)
        model.save(model_save_path)
        return history
    

    def test_model(self,datasets_test,labels_test,voca2id_datasets,model_save_path,rapport_save_path,load_with="local"):
        nb_output_class = len(voca2id_datasets[-1])
        model = self.load_keras_model(model_save_path,load_with)
        y_tag_test = to_categorical(labels_test, num_classes=nb_output_class)
        #datasets_test = datasets_test[1:]
        test_results = model.evaluate(datasets_test, y_tag_test, verbose=1)
        result_test = 'TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1])
        self.write_rapport_file(result_test, rapport_save_path)
        print(result_test)
        print("Testing Finish")
    
    def pred2label(self,pred,id2tag,predict=True):
        out = []
        for pred_i in pred:
            out_i = []
            for p in pred_i:
                p_i = np.argmax(p, axis=-1)
                if p_i != 0:
                    out_i.append(id2tag.get(p_i))#.replace("PAD","0"))
            out.append(out_i)
        return out

    def evaluation(self, datasets_test, labels_test, voca2id_datasets, id2voca_dataset, model_save_path, rapport_save_path,load_with="local"):
        nb_output_class = len(voca2id_datasets[-1])
        model = self.load_keras_model(model_save_path, load_with)
        y_tag_test = to_categorical(labels_test, num_classes=nb_output_class)
        # datasets_test = datasets_test[1:]
        test_results = model.evaluate(datasets_test, y_tag_test, verbose=1)
        result_test = 'TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1])
        self.write_rapport_file(result_test, rapport_save_path)
        print(result_test)
        print("Testing Finish")

        #model = self.load_keras_model(model_save_path,load_with)
        id2word = id2voca_dataset[1]
        id2tag = id2voca_dataset[-1]
        tag2id = voca2id_datasets[-1]

        y_predict=[]
        y_verify=[]
        for i in range(0,len(datasets_test[0])):
            sentence_predict = []
            for item_set in datasets_test:
                sentence_predict.append(np.array([item_set[i]]))
            s_predict = model.predict(sentence_predict)
            y_p = np.argmax(s_predict, axis=-1)
            y_t = labels_test[i]
            temp_result_pred=[]
            temp_result_verify=[]
            for w, t, p in zip(datasets_test[0][i], y_t, y_p[0]):
                if w != 0:
                    temp_result_pred.append(id2tag[p])
                    temp_result_verify.append(id2tag[t])
            y_predict.append(temp_result_pred)
            y_verify.append(temp_result_verify)



        print(y_predict[1])
        print("\n==========")
        print(y_verify[1])
        print("pred_labels shape:", len(y_predict))
        print("test_labels shape:", len(y_verify))

        print("%d == %d\n"%(len(y_predict), len(y_verify)))
        count_same=0
        for i in range(0,len(y_predict)):
            if np.array_equal(y_predict[i], y_verify[i]):
                count_same +=1
        accuracy = count_same/len(y_predict)

        count_lex_same=0
        all_lex=0
        for i in range(0,len(y_verify)):
            for j in range(0, len(y_verify[i])):
                all_lex += 1
                if np.array_equal(y_predict[i][j], y_verify[i][j]):
                    count_lex_same +=1
        
        accuracy_words = count_lex_same/all_lex
        result=""
        result += "\nAccuracy Sentence: {:.2%}\n".format(accuracy)
        result += "Accuracy Word: {:.2%}".format(accuracy_words)

        print(result)
        self.write_rapport_file(result, rapport_save_path)

        f1_ = "\nF1-score: {:.1%}".format(f1_score(y_verify, y_predict))
        report = "\n"+classification_report(y_verify, y_predict)
        print(f1_)
        #print(report)
        self.write_rapport_file(f1_, rapport_save_path)
        print("Evaluation Finish")

        i =0
        sentence_predict = []
        # datasets_test = datasets_test[1:]
        for item_set in datasets_test:
            sentence_predict.append(np.array([item_set[i]]))
        s_predict = model.predict(sentence_predict)
        p = np.argmax(s_predict, axis=-1)
        verify = labels_test[i]
        text = ""
        text = text + "{:15}||{:5}||{}".format("Word", "True", "Pred")
        print(text)
        line = 30 * "="
        print(line)
        text = text + "\n" + line + "\n"
        for w, t, pred in zip(datasets_test[0][i], verify, p[0]):
            if w != 0:
                result = "{:15}: {:5} {}".format(id2word[w], id2tag[t], id2tag[pred])
                print(result)
                text = text + result + "\n"
        with open(rapport_save_path, "a") as file_write:
            file_write.write(text+"\n")

            file_write.write(report)
        print("Example Finish")

    def example_test(self,datasets_test,labels_test,id2voca_dataset,model_save_path, example_file_save,load_with="local"):
        id2word = id2voca_dataset[1]
        id2tag = id2voca_dataset[-1]
        model = self.load_keras_model(model_save_path,load_with)
        i = 0
        sentence_predict=[]
        #datasets_test = datasets_test[1:]
        for item_set in datasets_test:
            sentence_predict.append(np.array([item_set[i]]))
        s_predict = model.predict(sentence_predict)
        p = np.argmax(s_predict, axis=-1)
        verify = labels_test[i]
        text=""
        text = text + "{:15}||{:5}||{}".format("Word", "True", "Pred")
        print(text)
        line=30 * "="
        print(line)
        text = text +"\n"+line+"\n"
        for w, t, pred in zip(datasets_test[0][i], verify, p[0]):
            if w != 0:
                result ="{:15}: {:5} {}".format(id2word[w], id2tag[t], id2tag[pred])
                print(result)
                text = text + result + "\n"
        with open(example_file_save,"w") as file_write:
            file_write.write(text)
    # only in the running with one features on words 
    def sentence_example(self,sentence,word2id,id2word,id2tag,model_save_path,example_file_save):
        model = self.load_keras_model(model_save_path,load_with="")
        sentence = sentence.lower().split(" ")
        sentence = [np.asarray([word2id.get(w,1) for w in sentence])]
        x_sentence_test = pad_sequences(sentence,maxlen=self.max_sequence_length)
        p = model.predict(np.array([x_sentence_test[0]]))
        p = np.argmax(p, axis=-1)
        text=""
        text = "{:15}||{}".format("Word", "Prediction")
        print(text)
        line = "\n" +30 * "="+ "\n"
        print(line)
        text =text + line
        for w, pred in zip(x_sentence_test[0], p[0]):
            if w != 0:
                result = "{:15}: {:5}".format(id2word[w-1], id2tag[pred]) 
                print(result + '\n')
                text = text + result + "\n"
        with open(example_file_save,"w") as file_write:
            file_write.write(text)

    def create_custom_objects(self):
        instanceHolder = {"instance": None}
        class ClassWrapper(CRF):
            def __init__(self, *args, **kwargs):
                instanceHolder["instance"] = self
                super(ClassWrapper, self).__init__(*args, **kwargs)
        def loss(*args):
            method = getattr(instanceHolder["instance"], "loss_function")
            return method(*args)
        def accuracy(*args):
            method = getattr(instanceHolder["instance"], "accuracy")
            return method(*args)
        return {"ClassWrapper": ClassWrapper ,"CRF": ClassWrapper, "loss": loss, "accuracy":accuracy}
    def generator(self, all_X,all_Y, n_classes, batch_size=32):
        num_samples = len(all_X[0])
        while True:
            for offset in range(0, num_samples, batch_size):
                all_ds=[]
                for i in range(len(all_X)):
                    all_ds.append(all_X[i][offset:offset + batch_size])
                y = all_Y[offset:offset + batch_size]
                y = to_categorical(y, num_classes=n_classes)
                
                yield all_ds, y
                
    def load_keras_model(self,path,load_with="local"):
        if load_with=="non_crf": # non crf at local and server
            return load_model(path)
        else: #  crf at local machine
            #return load_model(path, custom_objects=self.create_custom_objects())
             # crf at server machine
            return load_model(path,custom_objects={'CRF': CRF,'crf_loss': crf_loss,'crf_viterbi_accuracy': crf_viterbi_accuracy})
    

    def write_rapport_file(self, content, rapport_save_path):
        with open(rapport_save_path,'a') as file_report:
            file_report.write("\n======================================\n")
            file_report.write(content)
            file_report.write("\n======================================\n")
