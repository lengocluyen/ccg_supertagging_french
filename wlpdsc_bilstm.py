import os
import sys
import pickle
from model import ModelTraining
from data_handle import DataHandle
from sklearn.model_selection import train_test_split
import numpy as np
from time import gmtime, strftime
import time
from visualization import Visualisation
root = os.getcwd()
"""
if "lengocluyen" in root:
    print("Running in local machine")
else:
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
"""
if "lengocluyen" in root:
    print("Running in local machine")
    # parameters
    MAX_SEQUENCE_LENGTH = 5
    EMBEDDING_DIM = 200
    TEST_SPLIT = 0.2
    DEV_SPLIT = 0.2
    BATCH_SIZE = 32
    NUMBER_LAYER = 2
    EPOCH = 1
else:
    # parameters
    MAX_SEQUENCE_LENGTH = 120
    EMBEDDING_DIM = 200
    TEST_SPLIT = 0.2
    DEV_SPLIT = 0.2
    BATCH_SIZE = 32
    NUMBER_LAYER = 128
    EPOCH = 17
if "lengocluyen" in root:
    print("Running in local machine")
    bin_word2vec_path = "/home/yannis/ccgtagging/english/glove/glove.6B.200d.txt"
    dataset_path = "./ccgresult_en_t/"
else:
    bin_word2vec_path =  "/home/yannis/ccgtagging/english/glove/glove.6B.200d.txt"
    dataset_path = "/home/yannis/ccgtagging/english/ccgresult_en/"

# Setting save path default
dataset_in_text_save = root + "/common/dataset.pkl"
dataset_split_save_path=root + "/common/dataset_handled.pkl"
embedding_word_matrix_path = root + "/common/embedding_word_matrix.pkl"
embedding_char_matrix_path = root + "/common/embedding_char_matrix.pkl"
embedding_lemma_matrix_path = root + "/common/embedding_lemma_matrix.pkl"
embedding_postag_matrix_path = root + "/common/embedding_postag_matrix.pkl"
embedding_deprel_matrix_path = root + "/common/embedding_deprel_matrix.pkl"
embedding_suffix_matrix_path = root + "/common/embedding_suffix_matrix.pkl"
embedding_cap_matrix_path = root + "/common/embedding_cap_matrix.pkl"

model_save_path="model.h5"
rapport_save_path="repport_configuration.txt"
example_file_save = "example.txt"
history_train_path = "trainHistoryDict.pkl"
image_folder=""

# Data Preparation for Model
# index in data set: 0: char, 1 word, 2 lemma, 3 postag, 4 deprel, 5 suffix, 6 cap, 7, targe
def get_data_preparation_full(data_handle):
    datasets, datasets_train, datasets_dev, datasets_test,\
    labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = data_handle.data_preparation_padding(dataset_in_text_save,max_sequence_length=MAX_SEQUENCE_LENGTH,dataset_split_save_path=dataset_split_save_path)
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("XY","Set","Original","Train","Dev","Test"))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X0","Char" ,str(np.array(datasets[0].shape)), str(np.array(datasets_train[0].shape)), str(np.array(datasets_dev[0].shape)),str(np.array(datasets_test[0].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X1","Words" ,str(np.array(datasets[1].shape)), str(np.array(datasets_train[1].shape)), str(np.array(datasets_dev[1].shape)),str(np.array(datasets_test[1].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Lemma",str(np.array(datasets[2].shape)), str(np.array(datasets_train[2].shape)), str(np.array(datasets_dev[2].shape)),str(np.array(datasets_test[2].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X3","Xpostag",str(np.array(datasets[3].shape)), str(np.array(datasets_train[3].shape)), str(np.array(datasets_dev[3].shape)),str(np.array(datasets_test[3].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X4","Deprel",str(np.array(datasets[4].shape)), str(np.array(datasets_train[4].shape)), str(np.array(datasets_dev[4].shape)),str(np.array(datasets_test[4].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X5","Suffix",str(np.array(datasets[5].shape)), str(np.array(datasets_train[5].shape)), str(np.array(datasets_dev[5].shape)),str(np.array(datasets_test[5].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X6","Cap",str(np.array(datasets[6].shape)), str(np.array(datasets_train[6].shape)), str(np.array(datasets_dev[6].shape)),str(np.array(datasets_test[6].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("Y ","Tag",str(np.array(labels.shape)), str(np.array(labels_train.shape)), str(np.array(labels_dev.shape)),str(np.array(labels_test.shape))))
    return datasets, datasets_train, datasets_dev, datasets_test,\
        labels, labels_train, labels_dev, labels_test, \
        voca2id_datasets,id2voca_dataset
# get features: [word] by index: [1: word] in builde_set variable
# named: features_wsc
def get_data_preparation_word(data_handle):
    datasets, datasets_train, datasets_dev, datasets_test, \
    labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = data_handle.data_preparation_padding(dataset_in_text_save,build_set=[1],test_size=TEST_SPLIT,dev_size=DEV_SPLIT, max_sequence_length=MAX_SEQUENCE_LENGTH,dataset_split_save_path=dataset_split_save_path)
    print("Datset Feature Length: ", len(datasets))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("XY","Set","Original","Train","Dev","Test"))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X1","Words" ,str(np.array(datasets[0].shape)), str(np.array(datasets_train[0].shape)), str(np.array(datasets_dev[0].shape)),str(np.array(datasets_test[0].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("Y ","Tag",str(np.array(labels.shape)), str(np.array(labels_train.shape)), str(np.array(labels_dev.shape)),str(np.array(labels_test.shape))))
    return datasets, datasets_train, datasets_dev, datasets_test,\
        labels, labels_train, labels_dev, labels_test, \
        voca2id_datasets,id2voca_dataset
# get features: [word, suffix] by index: [1: word, 5: suffix] in builde_set variable
# named: features_wsc
def get_data_preparation_word_suffix(data_handle):
    datasets, datasets_train, datasets_dev, datasets_test, \
    labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = data_handle.data_preparation_padding(dataset_in_text_save,build_set=[1,5],test_size=TEST_SPLIT,dev_size=DEV_SPLIT, max_sequence_length=MAX_SEQUENCE_LENGTH,dataset_split_save_path=dataset_split_save_path)
    print("Datset Feature Length: ", len(datasets))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("XY","Set","Original","Train","Dev","Test"))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X1","Words" ,str(np.array(datasets[0].shape)), str(np.array(datasets_train[0].shape)), str(np.array(datasets_dev[0].shape)),str(np.array(datasets_test[0].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X5","Suffix",str(np.array(datasets[1].shape)), str(np.array(datasets_train[1].shape)), str(np.array(datasets_dev[1].shape)),str(np.array(datasets_test[1].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("Y ","Tag",str(np.array(labels.shape)), str(np.array(labels_train.shape)), str(np.array(labels_dev.shape)),str(np.array(labels_test.shape))))
    return datasets, datasets_train, datasets_dev, datasets_test,\
        labels, labels_train, labels_dev, labels_test, \
        voca2id_datasets,id2voca_dataset


# get features: [word, suffix, cap] by index: [1: word, 5: suffix, 6: cap] in builde_set variable
# named: features_wsc
def get_data_preparation_word_suffix_cap(data_handle):
    datasets, datasets_train, datasets_dev, datasets_test, \
    labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = data_handle.data_preparation_padding(dataset_in_text_save,build_set=[1,5,6],test_size=TEST_SPLIT,dev_size=DEV_SPLIT, max_sequence_length=MAX_SEQUENCE_LENGTH,dataset_split_save_path=dataset_split_save_path)
    print("Datset Feature Length: ", len(datasets))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("XY","Set","Original","Train","Dev","Test"))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X1","Words" ,str(np.array(datasets[0].shape)), str(np.array(datasets_train[0].shape)), str(np.array(datasets_dev[0].shape)),str(np.array(datasets_test[0].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X5","Suffix",str(np.array(datasets[1].shape)), str(np.array(datasets_train[1].shape)), str(np.array(datasets_dev[1].shape)),str(np.array(datasets_test[1].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X6","Cap",str(np.array(datasets[2].shape)), str(np.array(datasets_train[2].shape)), str(np.array(datasets_dev[2].shape)),str(np.array(datasets_test[2].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("Y ","Tag",str(np.array(labels.shape)), str(np.array(labels_train.shape)), str(np.array(labels_dev.shape)),str(np.array(labels_test.shape))))
    return datasets, datasets_train, datasets_dev, datasets_test,\
        labels, labels_train, labels_dev, labels_test, \
        voca2id_datasets,id2voca_dataset

# get features: [char, word, suffix, cap] by index: [0: char, 1: word, 5:suffix, 6:cap] in builde_set variable
# named: features_cwsc
def get_data_preparation_char_word_suffix_cap(data_handle):
    datasets, datasets_train, datasets_dev, datasets_test, \
    labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = data_handle.data_preparation_padding(dataset_in_text_save,build_set=[0,1,5,6],test_size=TEST_SPLIT,dev_size=DEV_SPLIT, max_sequence_length=MAX_SEQUENCE_LENGTH,dataset_split_save_path=dataset_split_save_path)
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("XY","Set","Original","Train","Dev","Test"))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X0","Char" ,str(np.array(datasets[0].shape)), str(np.array(datasets_train[0].shape)), str(np.array(datasets_dev[0].shape)),str(np.array(datasets_test[0].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X1","Word",str(np.array(datasets[1].shape)), str(np.array(datasets_train[1].shape)), str(np.array(datasets_dev[1].shape)),str(np.array(datasets_test[1].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X5","Suffix",str(np.array(datasets[2].shape)), str(np.array(datasets_train[2].shape)), str(np.array(datasets_dev[2].shape)),str(np.array(datasets_test[2].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X6","Cap",str(np.array(datasets[3].shape)), str(np.array(datasets_train[3].shape)), str(np.array(datasets_dev[3].shape)),str(np.array(datasets_test[3].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("Y ","Tag",str(np.array(labels.shape)), str(np.array(labels_train.shape)), str(np.array(labels_dev.shape)),str(np.array(labels_test.shape))))
    return datasets, datasets_train, datasets_dev, datasets_test,\
        labels, labels_train, labels_dev, labels_test, \
        voca2id_datasets,id2voca_dataset
# get features: [char, word, lemma,suffix, cap] by index: [0:char, 1:word, 2:lemma,5:suffix, 6:cap] in builde_set variable
# named: features_cwlsc
def get_data_preparation_char_word_lemma_suffix_cap(data_handle):
    datasets, datasets_train, datasets_dev, datasets_test, \
    labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = data_handle.data_preparation_padding(dataset_in_text_save,build_set=[0,1,2,5,6],test_size=TEST_SPLIT,dev_size=DEV_SPLIT, max_sequence_length=MAX_SEQUENCE_LENGTH,dataset_split_save_path=dataset_split_save_path)
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("XY","Set","Original","Train","Dev","Test"))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X0","Char" ,str(np.array(datasets[0].shape)), str(np.array(datasets_train[0].shape)), str(np.array(datasets_dev[0].shape)),str(np.array(datasets_test[0].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X1","Word",str(np.array(datasets[1].shape)), str(np.array(datasets_train[1].shape)), str(np.array(datasets_dev[1].shape)),str(np.array(datasets_test[1].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Lemma",str(np.array(datasets[2].shape)), str(np.array(datasets_train[2].shape)), str(np.array(datasets_dev[2].shape)),str(np.array(datasets_test[2].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X5","Suffix",str(np.array(datasets[3].shape)), str(np.array(datasets_train[3].shape)), str(np.array(datasets_dev[3].shape)),str(np.array(datasets_test[3].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X6","Cap",str(np.array(datasets[4].shape)), str(np.array(datasets_train[4].shape)), str(np.array(datasets_dev[4].shape)),str(np.array(datasets_test[4].shape))))
    
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("Y ","Tag",str(np.array(labels.shape)), str(np.array(labels_train.shape)), str(np.array(labels_dev.shape)),str(np.array(labels_test.shape))))
    return datasets, datasets_train, datasets_dev, datasets_test,\
        labels, labels_train, labels_dev, labels_test, \
        voca2id_datasets,id2voca_dataset
# get features: [char, word, lemma, postag,suffix, cap] by index: [0:char, 1:word, 2:lemma, 3:postag, 5:suffix, 6:cap] in builde_set variable
# named: features_cwlpsc
def get_data_preparation_char_word_lemma_postag_suffix_cap(data_handle):
    datasets, datasets_train, datasets_dev, datasets_test, \
    labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = data_handle.data_preparation_padding(dataset_in_text_save,build_set=[0,1,2,3,5,6],test_size=TEST_SPLIT,dev_size=DEV_SPLIT, max_sequence_length=MAX_SEQUENCE_LENGTH,dataset_split_save_path=dataset_split_save_path)
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("XY","Set","Original","Train","Dev","Test"))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X0","Char" ,str(np.array(datasets[0].shape)), str(np.array(datasets_train[0].shape)), str(np.array(datasets_dev[0].shape)),str(np.array(datasets_test[0].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X1","Word",str(np.array(datasets[1].shape)), str(np.array(datasets_train[1].shape)), str(np.array(datasets_dev[1].shape)),str(np.array(datasets_test[1].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Lemma",str(np.array(datasets[2].shape)), str(np.array(datasets_train[2].shape)), str(np.array(datasets_dev[2].shape)),str(np.array(datasets_test[2].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Lemma",str(np.array(datasets[3].shape)), str(np.array(datasets_train[3].shape)), str(np.array(datasets_dev[3].shape)),str(np.array(datasets_test[3].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X5","Suffix",str(np.array(datasets[5].shape)), str(np.array(datasets_train[5].shape)), str(np.array(datasets_dev[5].shape)),str(np.array(datasets_test[5].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X6","Cap",str(np.array(datasets[6].shape)), str(np.array(datasets_train[6].shape)), str(np.array(datasets_dev[6].shape)),str(np.array(datasets_test[6].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("Y ","Tag",str(np.array(labels.shape)), str(np.array(labels_train.shape)), str(np.array(labels_dev.shape)),str(np.array(labels_test.shape))))
    return datasets, datasets_train, datasets_dev, datasets_test,\
        labels, labels_train, labels_dev, labels_test, \
        voca2id_datasets,id2voca_dataset
# get features: [char, word, lemma, postag,deprel,suffix, cap] by index: [0:char, 1:word, 2:lemma, 3:postag, 4:deprel, 5:suffix, 6:cap] in builde_set variable
#named: features_cwlpdsc
def get_data_preparation_char_word_lemma_postag_deprel_suffix_cap(data_handle):
    return get_data_preparation_full(data_handle)
# get features: [lemma] by index: [2:lemma] in builde_set variable
# named: features_lp
def get_data_preparation_lemma(data_handle):
    datasets, datasets_train, datasets_dev, datasets_test, \
    labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = data_handle.data_preparation_padding(dataset_in_text_save,build_set=[2],test_size=TEST_SPLIT,dev_size=DEV_SPLIT, max_sequence_length=MAX_SEQUENCE_LENGTH,dataset_split_save_path=dataset_split_save_path)
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("XY","Set","Original","Train","Dev","Test"))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X1","Lemma",str(np.array(datasets[0].shape)), str(np.array(datasets_train[0].shape)), str(np.array(datasets_dev[0].shape)),str(np.array(datasets_test[0].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("Y ","Tag",str(np.array(labels.shape)), str(np.array(labels_train.shape)), str(np.array(labels_dev.shape)),str(np.array(labels_test.shape))))
    return datasets, datasets_train, datasets_dev, datasets_test,\
        labels, labels_train, labels_dev, labels_test, \
        voca2id_datasets,id2voca_dataset

# get features: [lemma, postag] by index: [2:lemma, 3:postag] in builde_set variable
# named: features_lp
def get_data_preparation_lemma_postag(data_handle):
    datasets, datasets_train, datasets_dev, datasets_test, \
    labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = data_handle.data_preparation_padding(dataset_in_text_save,build_set=[2,3],test_size=TEST_SPLIT,dev_size=DEV_SPLIT, max_sequence_length=MAX_SEQUENCE_LENGTH,dataset_split_save_path=dataset_split_save_path)
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("XY","Set","Original","Train","Dev","Test"))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X1","Lemma",str(np.array(datasets[0].shape)), str(np.array(datasets_train[0].shape)), str(np.array(datasets_dev[0].shape)),str(np.array(datasets_test[0].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Postag",str(np.array(datasets[1].shape)), str(np.array(datasets_train[1].shape)), str(np.array(datasets_dev[1].shape)),str(np.array(datasets_test[1].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("Y ","Tag",str(np.array(labels.shape)), str(np.array(labels_train.shape)), str(np.array(labels_dev.shape)),str(np.array(labels_test.shape))))
    return datasets, datasets_train, datasets_dev, datasets_test,\
        labels, labels_train, labels_dev, labels_test, \
        voca2id_datasets,id2voca_dataset

# get features: [lemma, postag, deprel] by index: [2:lemma, 3:postag, 4:deprel] in builde_set variable
# named: features_lpd
def get_data_preparation_lemma_postag_deprel(data_handle):
    datasets, datasets_train, datasets_dev, datasets_test, \
    labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = data_handle.data_preparation_padding(dataset_in_text_save,build_set=[2,3,4],test_size=TEST_SPLIT,dev_size=DEV_SPLIT, max_sequence_length=MAX_SEQUENCE_LENGTH,dataset_split_save_path=dataset_split_save_path)
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("XY","Set","Original","Train","Dev","Test"))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X1","Lemma",str(np.array(datasets[0].shape)), str(np.array(datasets_train[0].shape)), str(np.array(datasets_dev[0].shape)),str(np.array(datasets_test[0].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Postag",str(np.array(datasets[1].shape)), str(np.array(datasets_train[1].shape)), str(np.array(datasets_dev[1].shape)),str(np.array(datasets_test[1].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X3","Deprel",str(np.array(datasets[2].shape)), str(np.array(datasets_train[2].shape)), str(np.array(datasets_dev[2].shape)),str(np.array(datasets_test[2].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("Y ","Tag",str(np.array(labels.shape)), str(np.array(labels_train.shape)), str(np.array(labels_dev.shape)),str(np.array(labels_test.shape))))
    return datasets, datasets_train, datasets_dev, datasets_test,\
        labels, labels_train, labels_dev, labels_test, \
        voca2id_datasets,id2voca_dataset
# get features: [word, lemma, postag, deprel] by index: [1:word, 2:lemma, 3:postag, 4:deprel] in builde_set variable
# named: features_cwlpsc
def get_data_preparation_word_lemma_postag_deprel(data_handle):
    datasets, datasets_train, datasets_dev, datasets_test, \
    labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = data_handle.data_preparation_padding(dataset_in_text_save,build_set=[1,2,3,4],test_size=TEST_SPLIT,dev_size=DEV_SPLIT, max_sequence_length=MAX_SEQUENCE_LENGTH,dataset_split_save_path=dataset_split_save_path)
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("XY","Set","Original","Train","Dev","Test"))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X1","Word",str(np.array(datasets[0].shape)), str(np.array(datasets_train[0].shape)), str(np.array(datasets_dev[0].shape)),str(np.array(datasets_test[0].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Lemma",str(np.array(datasets[1].shape)), str(np.array(datasets_train[1].shape)), str(np.array(datasets_dev[1].shape)),str(np.array(datasets_test[1].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Postag",str(np.array(datasets[2].shape)), str(np.array(datasets_train[2].shape)), str(np.array(datasets_dev[2].shape)),str(np.array(datasets_test[2].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X5","Deprel",str(np.array(datasets[3].shape)), str(np.array(datasets_train[3].shape)), str(np.array(datasets_dev[3].shape)),str(np.array(datasets_test[3].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("Y ","Tag",str(np.array(labels.shape)), str(np.array(labels_train.shape)), str(np.array(labels_dev.shape)),str(np.array(labels_test.shape))))
    return datasets, datasets_train, datasets_dev, datasets_test,\
        labels, labels_train, labels_dev, labels_test, \
        voca2id_datasets,id2voca_dataset
#
# get features: [word, lemma, postag, suffix, cap] by index: [1:word, 2:lemma, 3: postag,5:suffix, 6:cap] in builde_set variable
# named: features_cwlpsc
def get_data_preparation_word_lemma_postag_suffix_cap(data_handle):
    datasets, datasets_train, datasets_dev, datasets_test, \
    labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = data_handle.data_preparation_padding(dataset_in_text_save,build_set=[1,2,3,5,6],test_size=TEST_SPLIT,dev_size=DEV_SPLIT, max_sequence_length=MAX_SEQUENCE_LENGTH,dataset_split_save_path=dataset_split_save_path)
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("XY","Set","Original","Train","Dev","Test"))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X1","Word",str(np.array(datasets[0].shape)), str(np.array(datasets_train[0].shape)), str(np.array(datasets_dev[0].shape)),str(np.array(datasets_test[0].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Lemma",str(np.array(datasets[1].shape)), str(np.array(datasets_train[1].shape)), str(np.array(datasets_dev[1].shape)),str(np.array(datasets_test[1].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Postag",str(np.array(datasets[2].shape)), str(np.array(datasets_train[2].shape)), str(np.array(datasets_dev[2].shape)),str(np.array(datasets_test[2].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Suffix",str(np.array(datasets[3].shape)), str(np.array(datasets_train[3].shape)), str(np.array(datasets_dev[3].shape)),str(np.array(datasets_test[3].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X5","Cap",str(np.array(datasets[4].shape)), str(np.array(datasets_train[4].shape)), str(np.array(datasets_dev[4].shape)),str(np.array(datasets_test[4].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("Y ","Tag",str(np.array(labels.shape)), str(np.array(labels_train.shape)), str(np.array(labels_dev.shape)),str(np.array(labels_test.shape))))
    return datasets, datasets_train, datasets_dev, datasets_test,\
        labels, labels_train, labels_dev, labels_test, \
        voca2id_datasets,id2voca_dataset


# get features: [word, lemma, suffix, cap] by index: [1:word, 2:lemma, 5:suffix, 6:cap] in builde_set variable
# named: features_cwlpsc
def get_data_preparation_word_lemma_suffix_cap(data_handle):
    datasets, datasets_train, datasets_dev, datasets_test, \
    labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = data_handle.data_preparation_padding(dataset_in_text_save,build_set=[1,2,5,6],test_size=TEST_SPLIT,dev_size=DEV_SPLIT, max_sequence_length=MAX_SEQUENCE_LENGTH,dataset_split_save_path=dataset_split_save_path)
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("XY","Set","Original","Train","Dev","Test"))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X1","Word",str(np.array(datasets[0].shape)), str(np.array(datasets_train[0].shape)), str(np.array(datasets_dev[0].shape)),str(np.array(datasets_test[0].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Lemma",str(np.array(datasets[1].shape)), str(np.array(datasets_train[1].shape)), str(np.array(datasets_dev[1].shape)),str(np.array(datasets_test[1].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Suffix",str(np.array(datasets[2].shape)), str(np.array(datasets_train[2].shape)), str(np.array(datasets_dev[2].shape)),str(np.array(datasets_test[2].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X5","Cap",str(np.array(datasets[3].shape)), str(np.array(datasets_train[3].shape)), str(np.array(datasets_dev[3].shape)),str(np.array(datasets_test[3].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("Y ","Tag",str(np.array(labels.shape)), str(np.array(labels_train.shape)), str(np.array(labels_dev.shape)),str(np.array(labels_test.shape))))
    return datasets, datasets_train, datasets_dev, datasets_test,\
        labels, labels_train, labels_dev, labels_test, \
        voca2id_datasets,id2voca_dataset
#
# get features: [word, lemma, postag,deprel, suffix, cap] by index: [1:word, 2:lemma, 3: postag, 4:deprel, 5:suffix, 6:cap] in builde_set variable
# named: features_cwlpsc
def get_data_preparation_word_lemma_postag_deprel_suffix_cap(data_handle):
    datasets, datasets_train, datasets_dev, datasets_test, \
    labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = data_handle.data_preparation_padding(dataset_in_text_save,build_set=[1,2,3,4,5,6],test_size=TEST_SPLIT,dev_size=DEV_SPLIT, max_sequence_length=MAX_SEQUENCE_LENGTH,dataset_split_save_path=dataset_split_save_path)
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("XY","Set","Original","Train","Dev","Test"))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X1","Word",str(np.array(datasets[0].shape)), str(np.array(datasets_train[0].shape)), str(np.array(datasets_dev[0].shape)),str(np.array(datasets_test[0].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Lemma",str(np.array(datasets[1].shape)), str(np.array(datasets_train[1].shape)), str(np.array(datasets_dev[1].shape)),str(np.array(datasets_test[1].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Postag",str(np.array(datasets[2].shape)), str(np.array(datasets_train[2].shape)), str(np.array(datasets_dev[2].shape)),str(np.array(datasets_test[2].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X2","Deprel",str(np.array(datasets[3].shape)), str(np.array(datasets_train[3].shape)), str(np.array(datasets_dev[3].shape)),str(np.array(datasets_test[3].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X5","Suffix",str(np.array(datasets[4].shape)), str(np.array(datasets_train[4].shape)), str(np.array(datasets_dev[4].shape)),str(np.array(datasets_test[4].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("X5", "Cap", str(np.array(datasets[5].shape)),str(np.array(datasets_train[5].shape)),str(np.array(datasets_dev[5].shape)),str(np.array(datasets_test[5].shape))))
    print("{:2}: {:12} {:20} {:20} {:20} {}".format("Y ","Tag",str(np.array(labels.shape)), str(np.array(labels_train.shape)), str(np.array(labels_dev.shape)),str(np.array(labels_test.shape))))
    return datasets, datasets_train, datasets_dev, datasets_test,\
        labels, labels_train, labels_dev, labels_test, \
        voca2id_datasets,id2voca_dataset


def get_embedding_matrix(voca2id_datasets, data_handle):
    embedding_char_matrix = data_handle.building_embdding_matrix(voca2id_datasets[0],embedding_char_matrix_path)
    embedding_word_matrix = data_handle.building_embdding_matrix_by_word2vec(voca2id_datasets[1],embedding_word_matrix_path,bin_word2vec_path,embedding_dim=EMBEDDING_DIM)
    embedding_lemma_matrix = data_handle.building_embdding_matrix_by_word2vec(voca2id_datasets[2],embedding_lemma_matrix_path,bin_word2vec_path,embedding_dim=EMBEDDING_DIM)
    
    embedding_postag_matrix = data_handle.building_embdding_matrix(voca2id_datasets[3],embedding_postag_matrix_path)
    embedding_deprel_matrix = data_handle.building_embdding_matrix(voca2id_datasets[4],embedding_deprel_matrix_path)
    embedding_suffix_matrix = data_handle.building_embdding_matrix(voca2id_datasets[5],embedding_suffix_matrix_path)
    embedding_cap_matrix = data_handle.building_embdding_matrix(voca2id_datasets[6],embedding_cap_matrix_path)

    embedding_matrix = [embedding_char_matrix,embedding_word_matrix,embedding_lemma_matrix,embedding_postag_matrix,embedding_deprel_matrix,embedding_suffix_matrix,embedding_cap_matrix]

    return embedding_matrix

def save_computation_time_to_rapport(start, finish):
    r = "%s seconds" % (start - finish)
    with open (rapport_save_path, 'a') as write_file:
        write_file.write(root + r)
    print(r)

# Three models based on the features: Word(w)
def run_feature_w_by_bilstm(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_w_by_bilstm(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder, load_with="non_crf")
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    #model_trainning.test_model(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,model_save_path=model_save_path, rapport_save_path=rapport_save_path,load_with="non_crf")
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="non_crf")
    #model_trainning.example_test(datasets_test, labels_test, id2voca_dataset, model_save_path, example_file_save,load_with="non_crf")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_w_by_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_w_by_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    #model_trainning.test_model(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,model_save_path=model_save_path, rapport_save_path=rapport_save_path, load_with="local")
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    #model_trainning.example_test(datasets_test, labels_test, id2voca_dataset, model_save_path, example_file_save,load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")
def run_feature_w_by_2bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_w_by_2bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    #model_trainning.test_model(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,model_save_path=model_save_path, rapport_save_path=rapport_save_path, load_with="local")
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    #model_trainning.example_test(datasets_test, labels_test, id2voca_dataset, model_save_path, example_file_save,load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")
# Three models based on the features: Word, Suffix(ws)
def run_feature_ws_by_bilstm(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_suffix(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_ws_by_bilstm(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder, load_with="non_crf")
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    #model_trainning.test_model(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,model_save_path=model_save_path, rapport_save_path=rapport_save_path,load_with="non_crf")
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="non_crf")
    #model_trainning.example_test(datasets_test, labels_test, id2voca_dataset, model_save_path, example_file_save,load_with="non_crf")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_ws_by_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_suffix(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_ws_by_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    #model_trainning.test_model(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,model_save_path=model_save_path, rapport_save_path=rapport_save_path, load_with="local")
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    #model_trainning.example_test(datasets_test, labels_test, id2voca_dataset, model_save_path, example_file_save,load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")
def run_feature_ws_by_2bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_suffix(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_ws_by_2bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    #model_trainning.test_model(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,model_save_path=model_save_path, rapport_save_path=rapport_save_path, load_with="local")
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    #model_trainning.example_test(datasets_test, labels_test, id2voca_dataset, model_save_path, example_file_save,load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")
def run_feature_ws_by_multi_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_suffix(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_ws_by_multi_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    #visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    #model_trainning.test_model(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,model_save_path=model_save_path, rapport_save_path=rapport_save_path, load_with="local")
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    #model_trainning.example_test(datasets_test, labels_test, id2voca_dataset, model_save_path, example_file_save,load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)
    print("Finish")

# Three models based on the features: Word, Suffix, Cap (wsc)
def run_feature_wsc_by_bilstm(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root +"/"+ output_directory+"/"
    model_save_path=root_directory+ "model.h5"
    rapport_save_path=root_directory+ "repport_configuration.txt"
    example_file_save = root_directory+ "example.txt"
    history_train_path = root_directory+ "train_history_dict.pkl"
    image_folder = root_directory
    # Init 
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test,labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = get_data_preparation_word_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets,data_handle)
    
    # Build Model
    model_execute,callbacks_list = model_trainning.building_model_feature_wsc_by_bilstm(voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix,rapport_save_path=rapport_save_path,output_directory=root_directory,number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute,datasets_train, labels_train, datasets_dev,labels_dev,voca2id_datasets, embedding_matrix,model_save_path,callbacks_list=callbacks_list,number_epoch=EPOCH)
   
    # Save Model Information
    visualisation.accuracy_history(history,folder=image_folder,load_with="non_crf")
    #visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test,labels_test=labels_test,voca2id_datasets=voca2id_datasets,id2voca_dataset=id2voca_dataset,model_save_path=model_save_path,rapport_save_path=rapport_save_path,load_with="non_crf")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time,finish_time)
    
    print("Finish")

def run_feature_wsc_by_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root +"/"+ output_directory+"/"
    model_save_path=root_directory+ "model.h5"
    rapport_save_path=root_directory+ "repport_configuration.txt"
    example_file_save = root_directory+ "example.txt"
    history_train_path = root_directory+ "train_history_dict.pkl"
    image_folder = root_directory
    # Init 
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test,labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = get_data_preparation_word_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets,data_handle)
    
    # Build Model
    model_execute,callbacks_list = model_trainning.building_model_feature_wsc_by_bilstm_crf(voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix,rapport_save_path=rapport_save_path,output_directory=root_directory,number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute,datasets_train, labels_train, datasets_dev,labels_dev,voca2id_datasets, embedding_matrix,model_save_path,callbacks_list=callbacks_list,number_epoch=EPOCH)
   
    # Save Model Information
    visualisation.accuracy_history(history,folder=image_folder)
    #visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test,labels_test=labels_test,voca2id_datasets=voca2id_datasets,id2voca_dataset=id2voca_dataset,model_save_path=model_save_path,rapport_save_path=rapport_save_path,load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time,finish_time)
    
    print("Finish")
def run_feature_wsc_by_2bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root +"/"+ output_directory+"/"
    model_save_path=root_directory+ "model.h5"
    rapport_save_path=root_directory+ "repport_configuration.txt"
    example_file_save = root_directory+ "example.txt"
    history_train_path = root_directory+ "train_history_dict.pkl"
    image_folder = root_directory
    # Init 
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test,labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = get_data_preparation_word_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets,data_handle)
    
    # Build Model
    model_execute,callbacks_list = model_trainning.building_model_feature_wsc_by_2bilstm_crf(voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix,rapport_save_path=rapport_save_path,output_directory=root_directory,number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute,datasets_train, labels_train, datasets_dev,labels_dev,voca2id_datasets, embedding_matrix,model_save_path,callbacks_list=callbacks_list,number_epoch=EPOCH)
   
    # Save Model Information
    visualisation.accuracy_history(history,folder=image_folder)
    #visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test,labels_test=labels_test,voca2id_datasets=voca2id_datasets,id2voca_dataset=id2voca_dataset,model_save_path=model_save_path,rapport_save_path=rapport_save_path,load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time,finish_time)
    
    print("Finish")
def run_feature_wsc_by_multi_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root +"/"+ output_directory+"/"
    model_save_path=root_directory+ "model.h5"
    rapport_save_path=root_directory+ "repport_configuration.txt"
    example_file_save = root_directory+ "example.txt"
    history_train_path = root_directory+ "train_history_dict.pkl"
    image_folder = root_directory
    # Init 
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test,labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets,id2voca_dataset = get_data_preparation_word_lemma_postag_deprel(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets,data_handle)
    
    # Build Model
    model_execute,callbacks_list = model_trainning.building_model_feature_wlpd_by_multi_bilstm_crf(voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix,rapport_save_path=rapport_save_path,output_directory=root_directory,number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute,datasets_train, labels_train, datasets_dev,labels_dev,voca2id_datasets, embedding_matrix,model_save_path,callbacks_list=callbacks_list,number_epoch=EPOCH)
   
    # Save Model Information
    visualisation.accuracy_history(history,folder=image_folder)
    #visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test,labels_test=labels_test,voca2id_datasets=voca2id_datasets,id2voca_dataset=id2voca_dataset,model_save_path=model_save_path,rapport_save_path=rapport_save_path,load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time,finish_time)
    
    print("Finish")

# Three models based on the features: Word, Lemma, Suffix, Cap (wlsc)
def run_feature_wlsc_by_bilstm(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlsc_by_bilstm(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder, load_with="non_crf")
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="non_crf")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_wlsc_by_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlsc_by_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")
def run_feature_wlsc_by_2bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlsc_by_2bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")
def run_feature_wlsc_by_multi_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlsc_by_multi_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    #visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)
    print("Finish")
# Three models based on the features: Word, Lemma, Postag, Suffix, Cap (wlpsc)
def run_feature_wlpsc_by_bilstm(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_postag_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlpsc_by_bilstm(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder, load_with="non_crf")
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="non_crf")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_wlpsc_by_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_postag_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlpsc_by_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")
def run_feature_wlpsc_by_2bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_postag_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlpsc_by_2bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")
def run_feature_wlpsc_by_multi_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_postag_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlpsc_by_multi_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    #visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)
    print("Finish")
# Three models based on the features: Word, Lemma, Postag, Deprel, Suffix, Cap (wlpdsc)
def run_feature_wlpdsc_by_bilstm(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_postag_deprel_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlpdsc_by_bilstm(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder, load_with="non_crf")
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="non_crf")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_wlpdsc_by_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_postag_deprel_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlpdsc_by_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")
def run_feature_wlpdsc_by_2bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_postag_deprel_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlpdsc_by_2bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_wlpdsc_by_multi_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_postag_deprel_suffix_cap(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlpdsc_by_multi_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    #visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)
    print("Finish")
# Three models based on the features: Word, Lemma, Postag, Deprel (wlpd)
def run_feature_wlpd_by_bilstm(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_postag_deprel(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlpd_by_bilstm(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder, load_with="non_crf")
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="non_crf")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_wlpd_by_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_postag_deprel(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlpd_by_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

def run_feature_wlpd_by_2bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_postag_deprel(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlpd_by_2bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)
    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_wlpd_by_multi_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_word_lemma_postag_deprel(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_wlpd_by_multi_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    #visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)
    print("Finish")
# Three models based on the features: Lemma, Postag, Deprel (lpd)
def run_feature_lpd_by_bilstm(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_lemma_postag_deprel(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_lpd_by_bilstm(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder, load_with="non_crf")
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="non_crf")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_lpd_by_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_lemma_postag_deprel(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_lpd_by_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_lpd_by_2bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_lemma_postag_deprel(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_lpd_by_2bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_lpd_by_multi_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_lemma_postag_deprel(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_lpd_by_multi_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    #visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)
    print("Finish")

# Three models based on the features: Lemma, Postag (lp)
def run_feature_lp_by_bilstm(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_lemma_postag(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_lp_by_bilstm(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder, load_with="non_crf")
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="non_crf")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_lp_by_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_lemma_postag(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_lp_by_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_lp_by_2bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_lemma_postag(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_lp_by_2bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_lp_by_multi_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_lemma_postag(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_lp_by_multi_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    #visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)
    print("Finish")

# Three models based on the features: Lemma (l)
def run_feature_l_by_bilstm(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_lemma(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_l_by_bilstm(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder, load_with="non_crf")
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="non_crf")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_l_by_bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_lemma(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_l_by_bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")

def run_feature_l_by_2bilstm_crf(output_directory):
    start_time = time.time()
    data_handle = DataHandle(dataset_path)
    # load data
    if not os.path.exists(dataset_in_text_save):
        data_handle.save_dataset_by_pickle(dataset_in_text_save)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    root_directory = root + "/" + output_directory + "/"
    model_save_path = root_directory + "model.h5"
    rapport_save_path = root_directory + "repport_configuration.txt"
    example_file_save = root_directory + "example.txt"
    history_train_path = root_directory + "train_history_dict.pkl"
    image_folder = root_directory
    # Init
    model_trainning = ModelTraining(max_sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBEDDING_DIM,
                                    test_split=TEST_SPLIT, dev_split=DEV_SPLIT, batch_size=BATCH_SIZE)
    visualisation = Visualisation()
    # Load Dataset
    datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, \
    voca2id_datasets, id2voca_dataset = get_data_preparation_lemma(data_handle)
    embedding_matrix = get_embedding_matrix(voca2id_datasets, data_handle)

    # Build Model
    model_execute, callbacks_list = model_trainning.building_model_feature_l_by_2bilstm_crf(
        voca2id_datasets=voca2id_datasets, embedding_matrix=embedding_matrix, rapport_save_path=rapport_save_path,
        output_directory=root_directory, number_layer=NUMBER_LAYER)
    # Fitting Model Data
    history = model_trainning.fitting_data_model(model_execute, datasets_train, labels_train, datasets_dev, labels_dev,
                                                 voca2id_datasets, embedding_matrix, model_save_path,
                                                 callbacks_list=callbacks_list, number_epoch=EPOCH)

    # Save Model Information
    visualisation.accuracy_history(history, folder=image_folder)
    # visualisation.save_model_to_graphic(model_execute, folder=image_folder)
    with open(history_train_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # Test and Evaluation Model
    model_trainning.evaluation(datasets_test=datasets_test, labels_test=labels_test, voca2id_datasets=voca2id_datasets,
                               id2voca_dataset=id2voca_dataset, model_save_path=model_save_path,
                               rapport_save_path=rapport_save_path, load_with="local")
    finish_time = time.time()
    save_computation_time_to_rapport(start_time, finish_time)

    print("Finish")


if __name__ == "__main__":
    # Three models based on the features: Word, Lemma, Postag, Deprel (wlpd)
    """
    print("Start running BiLSTM model with feature set: Word, Lemma, Postag, Deprel")
    output = "test_server_features_wlpd_by_bilstm"
    run_feature_wlpd_by_bilstm(output)
    print("Finish running BiLSTM model with feature set: Word, Lemma, Postag, Deprel")

    print("Start running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel")
    output = "test_server_features_wlpd_by_bilstm_crf"
    run_feature_wlpd_by_bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel")

    print("Start running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel")
    output = "test_server_features_wlpd_by_2bilstm_crf"
    run_feature_wlpd_by_2bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel")

    print("Start running Multi-BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel")
    output = "test_server_features_wlpd_by_multi_bilstm_crf"
    run_feature_wlpd_by_multi_bilstm_crf(output)
    print("Finish running Multi-BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel")

    # Three models based on the features: Lemma, Postag, Deprel (lpd)
    print("Start running BiLSTM model with feature set: Lemma, Postag, Deprel")
    output = "test_server_features_lpd_by_bilstm"
    run_feature_lpd_by_bilstm(output)
    print("Finish running BiLSTM model with feature set: Lemma, Postag, Deprel")

    print("Start running BiLSTM-CRF model with feature set: Lemma, Postag, Deprel")
    output = "test_server_features_lpd_by_bilstm_crf"
    run_feature_lpd_by_bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Lemma, Postag, Deprel")

    print("Start running BiLSTM-CRF model with feature set: Lemma, Postag, Deprel")
    output = "test_server_features_lpd_by_2bilstm_crf"
    run_feature_lpd_by_2bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Lemma, Postag, Deprel")

    print("Start running Multi-BiLSTM-CRF model with feature set: Lemma, Postag, Deprel")
    output = "test_server_features_lpd_by_multi_bilstm_crf"
    run_feature_lpd_by_multi_bilstm_crf(output)
    print("Finish running Multi-BiLSTM-CRF model with feature set: Lemma, Postag, Deprel")

    # Three models based on the features: Lemma, Postag (lp)
    print("Start running BiLSTM model with feature set: Lemma, Postag")
    output = "test_server_features_lp_by_bilstm"
    run_feature_lp_by_bilstm(output)
    print("Finish running BiLSTM model with feature set: Lemma, Postag")

    print("Start running BiLSTM-CRF model with feature set: Lemma, Postag")
    output = "test_server_features_lp_by_bilstm_crf"
    run_feature_lp_by_bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Lemma, Postag")

    print("Start running BiLSTM-CRF model with feature set: Lemma, Postag")
    output = "test_server_features_lp_by_2bilstm_crf"
    run_feature_lp_by_2bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Lemma, Postag")

    print("Start running Multi-BiLSTM-CRF model with feature set: Lemma, Postag")
    output = "test_server_features_lp_by_multi_bilstm_crf"
    run_feature_lp_by_multi_bilstm_crf(output)
    print("Finish running Multi-BiLSTM-CRF model with feature set: Lemma, Postag")

    # Three models based on the features: Lemma (l)
    print("Start running BiLSTM model with feature set: Lemma")
    output = "test_server_features_l_by_bilstm"
    run_feature_l_by_bilstm(output)
    print("Finish running BiLSTM model with feature set: Lemma")

    print("Start running BiLSTM-CRF model with feature set: Lemma")
    output = "test_server_features_l_by_bilstm_crf"
    run_feature_l_by_bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Lemma")

    print("Start running BiLSTM-CRF model with feature set: Lemma")
    output = "test_server_features_l_by_2bilstm_crf"
    run_feature_l_by_2bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Lemma")


    
    # Three models based on the features: Word (w)
    print("Start running BiLSTM model with feature set: Word")
    output = "test_server_features_w_by_bilstm"
    run_feature_w_by_bilstm(output)
    print("Finish running BiLSTM model with feature set: Word")
    
    print("Start running BiLSTM-CRF model with feature set: Word")
    output = "test_server_features_w_by_bilstm_crf"
    run_feature_w_by_bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word")

    print("Start running BiLSTM-CRF model with feature set: Word")
    output = "test_server_features_w_by_2bilstm_crf"
    run_feature_w_by_2bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word")

    # Three models based on the features: Word, Suffix (ws)
    print("Start running BiLSTM model with feature set: Word, Suffix")
    output = "test_server_features_ws_by_bilstm"
    run_feature_ws_by_bilstm(output)
    print("Finish running BiLSTM model with feature set: Word, Suffix")

    print("Start running BiLSTM-CRF model with feature set: Word, Suffix")
    output = "test_server_features_ws_by_bilstm_crf"
    run_feature_ws_by_bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word, Suffix")

    print("Start running BiLSTM-CRF model with feature set: Word, Suffix")
    output = "test_server_features_ws_by_2bilstm_crf"
    run_feature_ws_by_2bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word, Suffix")

    print("Start running Multi-BiLSTM-CRF model with feature set: Word, Suffix")
    output = "test_server_features_ws_by_multi_bilstm_crf"
    run_feature_ws_by_multi_bilstm_crf(output)
    print("Finish running Multi-BiLSTM-CRF model with feature set: Word, Suffix")
  
    # Three models based on the features: Word, Suffix, Cap (wsc)
    print("Start running BiLSTM model with feature set: Word, Suffix, Cap")
    output = "test_server_features_wsc_by_bilstm"
    run_feature_wsc_by_bilstm(output)
    print("Finish running BiLSTM model with feature set: Word, Suffix, Cap")

    print("Start running BiLSTM-CRF model with feature set: Word, Suffix, Cap")
    output = "test_server_features_wsc_by_bilstm_crf"
    run_feature_wsc_by_bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word, Suffix, Cap")

    print("Start running BiLSTM-CRF model with feature set: Word, Suffix, Cap")
    output = "test_server_features_wsc_by_2bilstm_crf"
    run_feature_wsc_by_2bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word, Suffix, Cap")

    print("Start running Multi-BiLSTM-CRF model with feature set: Char, Word, Suffix, Cap")
    output = "test_server_features_wsc_by_multi_bilstm_crf"
    run_feature_wsc_by_multi_bilstm_crf(output)
    print("Finish running Multi-BiLSTM-CRF model with feature set: Char, Word, Suffix, Cap")

    # Three models based on the features: Word, Lemma, Deprel, Suffix, Cap (wlsc)
    print("Start running BiLSTM model with feature set: Word, Suffix, Cap")
    output = "test_server_features_wlsc_by_bilstm"
    run_feature_wlsc_by_bilstm(output)
    print("Finish running BiLSTM model with feature set: Word, Lemma,  Suffix, Cap")

    print("Start running BiLSTM-CRF model with feature set: Word, Lemma, Suffix, Cap")
    output = "test_server_features_wlsc_by_bilstm_crf"
    run_feature_wlsc_by_bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word, Lemma, Suffix, Cap")

    print("Start running BiLSTM-CRF model with feature set: Word, Lemma, Suffix, Cap")
    output = "test_server_features_wlsc_by_2bilstm_crf"
    run_feature_wlsc_by_2bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word, Lemma, Suffix, Cap")

    print("Start running Multi-BiLSTM-CRF model with feature set: Word, Lemma, Suffix, Cap")
    output = "test_server_features_wlsc_by_multi_bilstm_crf"
    run_feature_wlsc_by_multi_bilstm_crf(output)
    print("Finish running Multi-BiLSTM-CRF model with feature set: Word, Lemma, Suffix, Cap")

    # Three models based on the features: Word, Lemma, Postag, Suffix, Cap (wlpsc)
    print("Start running BiLSTM model with feature set: Word, Lemma, Postag, Suffix, Cap")
    output = "test_server_features_wlpsc_by_bilstm"
    run_feature_wlpsc_by_bilstm(output)
    print("Finish running BiLSTM model with feature set: Word, Lemma, Postag,, Suffix, Cap")

    print("Start running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Suffix, Cap")
    output = "test_server_features_wlpsc_by_bilstm_crf"
    run_feature_wlpsc_by_bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Suffix, Cap")

    print("Start running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Suffix, Cap")
    output = "test_server_features_wlpsc_by_2bilstm_crf"
    run_feature_wlpsc_by_2bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Suffix, Cap")

    print("Start running Multi-BiLSTM-CRF model with feature set: Word, Lemma, Postag, Suffix, Cap")
    output = "test_server_features_wlpsc_by_multi_bilstm_crf"
    run_feature_wlpsc_by_multi_bilstm_crf(output)
    print("Finish running Multi-BiLSTM-CRF model with feature set: Word, Lemma, Postag, Suffix, Cap")
    """
    # Three models based on the features: Word, Lemma, Postag, Deprel, Suffix, Cap (wlpdsc)
    print("Start running BiLSTM model with feature set: Word, Lemma, Postag, Deprel, Suffix, Cap")
    output = "test_server_features_wlpdsc_by_bilstm"
    run_feature_wlpdsc_by_bilstm(output)
    print("Finish running BiLSTM model with feature set: Word, Lemma, Postag, Deprel, Suffix, Cap")
    """"
    print("Start running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel, Suffix, Cap")
    output = "test_server_features_wlpdsc_by_bilstm_crf"
    run_feature_wlpdsc_by_bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel, Suffix, Cap")

    print("Start running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel, Suffix, Cap")
    output = "test_server_features_wlpdsc_by_2bilstm_crf"
    run_feature_wlpdsc_by_2bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel, Suffix, Cap")

    print("Start running Multi-BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel, Suffix, Cap")
    output = "test_server_features_wlpdsc_by_multi_bilstm_crf"
    run_feature_wlpdsc_by_multi_bilstm_crf(output)
    print("Finish running Multi-BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel, Suffix, Cap")

    # Three models based on the features: Word, Lemma, Postag, Deprel (wlpd)
    print("Start running BiLSTM model with feature set: Word, Lemma, Postag, Deprel")
    output = "test_server_features_wlpd_by_bilstm"
    run_feature_wlpd_by_bilstm(output)
    print("Finish running BiLSTM model with feature set: Word, Lemma, Postag, Deprel")

    print("Start running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel")
    output = "test_server_features_wlpd_by_bilstm_crf"
    run_feature_wlpd_by_bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel")

    print("Start running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel")
    output = "test_server_features_wlpd_by_2bilstm_crf"
    run_feature_wlpd_by_2bilstm_crf(output)
    print("Finish running BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel")

    print("Start running Multi-BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel")
    output = "test_server_features_wlpd_by_multi_bilstm_crf"
    run_feature_wlpd_by_multi_bilstm_crf(output)
    print("Finish running Multi-BiLSTM-CRF model with feature set: Word, Lemma, Postag, Deprel")
    """





