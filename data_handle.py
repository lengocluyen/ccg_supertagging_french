import os, re, codecs
import numpy as np
import pickle
from six import string_types

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

import numpy as np


class DataHandle():
    def __init__(self, directory_data=None):
        self.directory_data = directory_data
        self.file_data = os.listdir(directory_data)
        #print("Total Number of Files:", len(self.file_data))

    def load_glove_model(self,gloveFile):
        print ("Loading Glove Model")
        f = open(gloveFile,'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print ("Done.",len(model)," words loaded!")
        return model

    def read_corpus_data(self):
        corpus = []
        for file in os.listdir(self.directory_data):
            file_path = os.path.join(self.directory_data, file)
            with open(file_path, "r") as file_data:
                blank_line = 0
                sentence=[]
                for line in file_data:
                    line = line.rstrip('\n')
                    if len(line) > 0:
                        # 0:id, 1:words, 2:lemma, 3:postag, 4:deprel, 5:ccgtag, 6:deprel_id,
                        line = line.split("\t")
                        sentence.append([line[0].replace("\n", ""), line[1].replace("\n", ""), line[2].replace("\n", ""),
                                        line[3].replace("\n", ""), \
                                        line[4].replace("\n", ""), line[5].replace("\n", ""), line[6].replace("\n", "")])
                    else:
                        if blank_line == 0:
                            blank_line = 1
                        else:
                            blank_line = 0
                            corpus.append(sentence)
                            sentence=[]
                            #print("The End of the sentence.")
        return corpus
    #corpus = [
    #           [sentence01:[
    #                       word1: id, word, lemma, upostag, xpostag, head, deprel, deps, ccgtag
    #                       word2: id, word, lemma, upostag, xpostag, head, deprel, deps, ccgtag
    #                       .....]
    #           ]
    #           [sentence02:[
    #                       word1: id, word, lemma, upostag, xpostag, head, deprel, deps, ccgtag
    #                       word2: id, word, lemma, upostag, xpostag, head, deprel, deps, ccgtag
    #                       .....]
    #           ]
    #           ......
    #         ]

    def dataset_words(self):
        corpus = self.read_corpus_data()
        chars, words,lemma,postag,deprel, suffix,cap, tags = [],[],[],[],[],[],[],[]
        x_sentence_char=[] #[[sentence[word[char]]]]
        x_sentence_word=[]
        x_sentence_lemma=[]
        x_sentence_postag=[]
        x_sentence_deprel=[]
        x_sentence_suffix=[]
        x_sentence_cap=[]
        y_sentence_tag = []
        for item_sentence in corpus:
            tem_sentence_char=[]
            temp_sentence_word = []
            temp_sentence_lemma=[]
            temp_sentence_postag=[]
            temp_sentence_deprel=[]
            temp_sentence_suffix=[]
            temp_sentence_cap=[]
            temp_sentence_tag=[]
            for item_element in item_sentence:
                word = item_element[1].lower()
                words.append(word)
                temp_word_char=[]
                for item in word:
                    temp_word_char.append(item)
                    chars.append(item)
                first_char="Cap"
                if  word[0].isupper():
                    first_char="Cap"
                else:
                    first_char="Non"
                last_chars="__"
                if len(word)>2:
                    last_chars = str(word[-2:]).lower()
                else:
                    last_chars = str(word[-1:]).lower()
                lemma.append(item_element[2].lower())
                postag.append(item_element[3].lower())
                deprel.append(item_element[4].lower())
                suffix.append(last_chars.lower())
                cap.append(first_char)
                tags.append(item_element[5])
                tem_sentence_char.append(temp_word_char)
                temp_sentence_word.append(item_element[1].lower())
                temp_sentence_lemma.append(item_element[2].lower())
                temp_sentence_postag.append(item_element[3].lower())
                temp_sentence_deprel.append(item_element[4].lower())
                temp_sentence_suffix.append(last_chars)
                temp_sentence_cap.append(first_char)
                temp_sentence_tag.append(item_element[5])
            x_sentence_char.append(tem_sentence_char)
            x_sentence_word.append(temp_sentence_word)
            x_sentence_lemma.append(temp_sentence_lemma)
            x_sentence_postag.append(temp_sentence_postag)
            x_sentence_deprel.append(temp_sentence_deprel)
            x_sentence_suffix.append(temp_sentence_suffix)
            x_sentence_cap.append(temp_sentence_cap)
            y_sentence_tag.append(temp_sentence_tag)
        return set(chars),set(words),set(lemma),set(postag),set(deprel),set(suffix), set(cap), set(tags),x_sentence_char, x_sentence_word,x_sentence_lemma,x_sentence_postag,x_sentence_deprel,x_sentence_suffix,x_sentence_cap, y_sentence_tag

    def dataset_encoding(self):
        chars, words,lemma,postag,deprel, suffix, cap, tags,xs_char, xs_word,xs_lemma,xs_postag,xs_deprel,xs_suffix, xs_cap, ys_tag = self.dataset_words()
        char2id, id2char={},{}
        word2id, id2word ={},{}
        lemma2id, id2lemma ={},{}
        postag2id, id2postag={},{}
        deprel2id, id2deprel={},{}
        suffix2id, id2suffix = {},{}
        cap2id, id2cap={},{}
        tag2id, id2tag = {},{}

        char2id["PAD"]=0
        char2id["UNK"]=1
        for i, char in enumerate(chars):
            char2id[char] = i+2
            id2char[i+2] = char

        word2id["PAD"] = 0
        id2word[0]="PAD"
        word2id["UNK"] = 1
        id2word[1]="UNK"
        for i, word in enumerate(words):
            word2id[word] = i+2
            id2word[i+2]=word
        lemma2id["PAD"] = 0
        id2lemma[0]="PAD"
        lemma2id["UNK"] = 1
        id2lemma[1]="UNK"
        for i,la in enumerate(lemma):
            lemma2id[la] = i+2
            id2lemma[i+2]=la
        postag2id["PAD"] = 0
        id2postag[0]="PAD"
        for i,uptag in enumerate(postag):
            postag2id[uptag] = i+1
            id2postag[i+1]=uptag
        deprel2id["PAD"] = 0
        id2deprel[0]="PAD"
        for i,dp in enumerate(deprel):
            deprel2id[dp] = i+1
            id2deprel[i+1]=dp
        suffix2id["PAD"] = 0
        id2suffix[0]="PAD"
        suffix2id["UNK"] = 1
        id2suffix[1]="UNK"
        for i, suffix in enumerate(suffix):
            suffix2id[suffix] = i+2
            id2suffix[i+2]=suffix
        cap2id["PAD"]= 0
        id2cap[0]="PAD"
        for i, cap in enumerate(cap):
            cap2id[cap] = i+1
            id2cap[i+1]=cap

        tag2id["PAD"]= 0
        id2tag[0]="PAD"
        for i, tag in enumerate(tags):
            tag2id[tag]=i+1
            id2tag[i+1]=tag
        xs_char2ids, xs_word2ids,xs_lemma2ids,xs_postag2ids,xs_deprel2ids,xs_suffix2ids,xs_cap2ids, ys_tag2ids=[],[],[],[],[],[],[],[]
        for s_word_i in xs_char:
            temp_sentence_char=[]
            for word in s_word_i:
                tem_word_sentence_char=[]
                for char in word:
                    tem_word_sentence_char.append(char2id[char])
                temp_sentence_char.append(tem_word_sentence_char)
            xs_char2ids.append(temp_sentence_char)
        for s_word_i in xs_word:
            temp_sentence_word = []
            for word in s_word_i:
                temp_sentence_word.append(word2id[word])
            xs_word2ids.append(temp_sentence_word)
        for s_lemma_i in xs_lemma:
            temp_sentence_lemma = []
            for la in s_lemma_i:
                temp_sentence_lemma.append(lemma2id[la])
            xs_lemma2ids.append(temp_sentence_lemma)
        for s_postag_i in xs_postag:
            temp_sentence_postag = []
            for upt in s_postag_i:
                temp_sentence_postag.append(postag2id[upt])
            xs_postag2ids.append(temp_sentence_postag)
        for s_deprel_i in xs_deprel:
            temp_sentence_deprel = []
            for dp in s_deprel_i:
                temp_sentence_deprel.append(deprel2id[dp])
            xs_deprel2ids.append(temp_sentence_deprel)
        for s_suffix_i in xs_suffix:
            temp_sentence_suffix = []
            for dp in s_suffix_i:
                temp_sentence_suffix.append(suffix2id[dp])
            xs_suffix2ids.append(temp_sentence_suffix)
        for s_cap_i in xs_cap:
            temp_sentence_cap = []
            for dp in s_cap_i:
                temp_sentence_cap.append(cap2id[dp])
            xs_cap2ids.append(temp_sentence_cap)

        for s_tags_i in ys_tag:
            temp_tag = []
            for tag in s_tags_i:
                temp_tag.append(tag2id[tag])
            ys_tag2ids.append(temp_tag)
        
        return np.asarray(xs_char2ids), np.asarray(xs_word2ids),np.asarray(xs_lemma2ids),\
        np.asarray(xs_postag2ids),np.asarray(xs_deprel2ids) ,np.asarray(xs_suffix2ids), np.asarray(xs_cap2ids),\
         np.asarray(ys_tag2ids),char2id, id2char, word2id,id2word,lemma2id,id2lemma,postag2id,id2postag,\
         deprel2id,id2deprel,suffix2id,id2suffix,cap2id, id2cap,tag2id,id2tag
    
    def save_dataset_by_pickle(self,output_path=None):
        xs_char2ids, xs_word2ids,xs_lemma2ids,xs_postag2ids,xs_deprel2ids,xs_suffix2ids,xs_cap2ids, yx_tag2ids,char2id, id2char, word2id, id2word, lemma2id,id2lemma,postag2id,id2postag,deprel2id,id2deprel,suffix2id, id2suffix,cap2id, id2cap, tag2id, id2tag = self.dataset_encoding()
        pickle_files_saved = [xs_char2ids, xs_word2ids,xs_lemma2ids,xs_postag2ids,xs_deprel2ids,xs_suffix2ids,xs_cap2ids, yx_tag2ids, char2id, id2char, word2id, id2word, lemma2id,id2lemma,postag2id,id2postag,deprel2id,id2deprel,suffix2id,id2suffix, cap2id, id2cap, tag2id, id2tag]
        
        with open(output_path,"wb") as file:
            pickle.dump(pickle_files_saved,file)
        print("Saved as pickle file")
    
    def data_preparation(self,pickle_file):
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)    

    def data_preparation_padding(self,pickle_file, build_set=None,test_size=0.2, dev_size=0.2, max_sequence_length=100, dataset_split_save_path=None):
        if not os.path.exists(dataset_split_save_path):
            #fix max word len = 30
            max_word_len=30
            xs_char2ids_original, xs_word2ids_orginal,xs_lemma2ids_orginal,xs_postag2ids_orginal,xs_deprel2ids_orginal,xs_suffix2ids_original,xs_cap2ids_original, ys_tag2ids_orginal, char2id, id2char, word2id, id2word, lemma2id,id2lemma,postag2id,id2postag,deprel2id,id2deprel,suffix2id, id2suffix,cap2id, id2cap, tag2id, id2tag = self.data_preparation(pickle_file)
            new_xs_char2ids=[]
            for sentence in xs_char2ids_original:
                tem_sentence = pad_sequences(sentence,maxlen=max_word_len)
                new_xs_char2ids.append(tem_sentence)
            xs_char2ids = pad_sequences(new_xs_char2ids,maxlen=max_sequence_length) 
            xs_word2ids = pad_sequences(xs_word2ids_orginal,maxlen=max_sequence_length)
            xs_lemma2ids = pad_sequences(xs_lemma2ids_orginal,maxlen=max_sequence_length)
            xs_postag2ids = pad_sequences(xs_postag2ids_orginal,maxlen=max_sequence_length)
            xs_deprel2ids = pad_sequences(xs_deprel2ids_orginal,maxlen=max_sequence_length)
            xs_suffix2ids = pad_sequences(xs_suffix2ids_original,maxlen=max_sequence_length)
            xs_cap2ids = pad_sequences(xs_cap2ids_original,maxlen=max_sequence_length)
            ys_tag2ids = pad_sequences(ys_tag2ids_orginal,maxlen=max_sequence_length)
            
            # shuffle the data
            #x_sentence, y_tag = shuffle(x_sentence,y_tag)
            xs_char2ids_train, xs_char2ids_test,xs_word2ids_train,xs_word2ids_test,\
             xs_lemma2ids_train,xs_lemma2ids_test,\
            xs_postag2ids_train,xs_postag2ids_test,\
            xs_deprel2ids_train,xs_deprel2ids_test,xs_suffix_train, xs_suffix_test,\
            xs_cap_train, xs_cap_test,\
             ys_tag2ids_train,ys_tag2ids_test = train_test_split(xs_char2ids, xs_word2ids,xs_lemma2ids,xs_postag2ids,xs_deprel2ids,xs_suffix2ids,xs_cap2ids, ys_tag2ids, test_size=test_size,random_state=42)
            
            xs_char2ids_train, xs_char2ids_dev, xs_word2ids_train,xs_word2ids_dev,\
            xs_lemma2ids_train,xs_lemma2ids_dev,xs_postag2ids_train,xs_postag2ids_dev,\
            xs_deprel2ids_train,xs_deprel2ids_dev, xs_suffix_train, xs_suffix_dev, \
            xs_cap_train, xs_cap_dev,\
            ys_tag2ids_train,ys_tag2ids_dev = train_test_split(xs_char2ids_train,xs_word2ids_train,xs_lemma2ids_train,xs_postag2ids_train,xs_deprel2ids_train,xs_suffix_train,xs_cap_train, ys_tag2ids_train, test_size=dev_size,random_state=1)


            full_datasets = [xs_char2ids,xs_word2ids,xs_lemma2ids,xs_postag2ids,xs_deprel2ids,xs_suffix2ids, xs_cap2ids]
            full_datasets_train = [xs_char2ids_train,xs_word2ids_train,xs_lemma2ids_train,xs_postag2ids_train,xs_deprel2ids_train, xs_suffix_train,xs_cap_train]
            full_datasets_dev = [xs_char2ids_dev,xs_word2ids_dev,xs_lemma2ids_dev,xs_postag2ids_dev,xs_deprel2ids_dev,xs_suffix_dev,xs_cap_dev]
            full_datasets_test = [xs_char2ids_test,xs_word2ids_test,xs_lemma2ids_test,xs_postag2ids_test,xs_deprel2ids_test,xs_suffix_test,xs_cap_test]
            
            
            labels = ys_tag2ids
            labels_train = ys_tag2ids_train
            labels_dev = ys_tag2ids_dev
            labels_test = ys_tag2ids_test
            voca2id_datasets =[char2id,word2id,lemma2id,postag2id,deprel2id,suffix2id,cap2id,tag2id]
            id2voca_dataset=[id2char,id2word,id2lemma,id2postag,id2deprel,id2suffix,id2cap,id2tag]
            
            with open(dataset_split_save_path,"wb") as file:
                save_matrix = [full_datasets, full_datasets_train, full_datasets_dev, full_datasets_test, labels, labels_train, labels_dev, labels_test, voca2id_datasets,id2voca_dataset]
                pickle.dump(save_matrix,file)
        datasets=[]
        datasets_train=[]
        datasets_dev=[]
        datasets_test=[]
        try:
            with open(dataset_split_save_path, 'rb') as f:
                load_matrix = pickle.load(f)
                full_datasets, full_datasets_train, full_datasets_dev, full_datasets_test, labels, labels_train, labels_dev, labels_test, voca2id_datasets,id2voca_dataset = load_matrix      
                if build_set is not None and len(build_set)>0:
                    for item in build_set:
                        datasets.append(full_datasets[item])
                        datasets_train.append(full_datasets_train[item])
                        datasets_dev.append(full_datasets_dev[item])
                        datasets_test.append(full_datasets_test[item])
                else:
                    datasets = full_datasets
                    datasets_train = full_datasets_train
                    datasets_dev = full_datasets_dev
                    datasets_test = full_datasets_test            
                return datasets, datasets_train, datasets_dev, datasets_test, labels, labels_train, labels_dev, labels_test, voca2id_datasets,id2voca_dataset
        except:
            return None
    

    
    def data_train_and_test(self,datasets, labels, test_size):

        return train_test_split(datasets,labels, test_size=test_size,random_state=42)
    
    def data_train_and_dev_in_train_set(self,x_sentence_train, y_tag_train, dev_size):
        return train_test_split(x_sentence_train, y_tag_train, test_size=dev_size,random_state=1)

    def building_embdding_matrix(self,item2id,embedding_item_matrix_path):
        embedding_matrix = np.random.random((len(item2id)+1,len(item2id)+1))
        if not os.path.exists(embedding_item_matrix_path):
            vocalist = []
            for word,i in item2id.items():
                vocalist.append(i)
            embedding_matrix=to_categorical(np.array(vocalist))
            with open(embedding_item_matrix_path,"wb") as file:
                save_matrix = [embedding_matrix]
                pickle.dump(save_matrix,file)
        else:
            with open(embedding_item_matrix_path, 'rb') as f:
                load_matrix = pickle.load(f)
                embedding_matrix = load_matrix[0]          
        return embedding_matrix

    def building_embdding_matrix_by_word2vec(self,item2id,embedding_item_matrix_path,binpath=None,embedding_dim=0):
        embedding_matrix = np.random.random((len(item2id)+2,embedding_dim))
        if not os.path.exists(embedding_item_matrix_path):
            word2vec = self.load_glove_model(binpath)
            notfound=0
            for word,i in item2id.items():
                try:
                    embedding_vector = word2vec[word]
                    embedding_matrix[i] = embedding_vector
                except:
                    notfound+=1
            with open(embedding_item_matrix_path,"wb") as file:
                save_matrix = [embedding_matrix]
                pickle.dump(save_matrix,file)
        else:
            with open(embedding_item_matrix_path, 'rb') as f:
                load_matrix = pickle.load(f)
                embedding_matrix = load_matrix[0]          
        return embedding_matrix




