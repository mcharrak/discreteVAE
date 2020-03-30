#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:26:48 2018

@author: Amine
"""

DATA_PATH="Dataset/"

import numpy as np
import copy
from gensim.models import KeyedVectors #import gensim to create only In-Vocb-Word-Embedding
from gensim.models import FastText #import gensim tool to create OOV capable model (with n-grams)


"""
    possible latents_classes values:
    {
    'vaild tuple': array([   0,    1,    2, ..., 1097, 1098, 1099]), (POS 0)
    'object_sing/pl': array([0, 1]),                                 (POS 1)
    'sentence_type': array([0, 1]),                                  (POS 2)
    'gender': array([0, 1]),                                         (POS 3)
    'subject_sing/pl': array([0, 1]),                                (POS 4)
    '1st_2nd_3rd_person': array([0, 1, 2]),                          (POS 5)
    'pos/neg_verb': array([0, 1]),                                   (POS 6)
    'verb_tense': array([0, 1, 2]),                                  (POS 7)
    'verb_style/aspect': array([0, 1]),                              (POS 8)
    }
"""
"""
Meaning of latent vector positions (9 positions):
latent_vec =[ 0. tuple_id | 1. sing./pl.obj | 2. quest./stat. | 3. male/fem |
              4. sing./pl.subj | 5. #person | 6. pos./neg.verb | 7. tense | 
              8. aspect ]

where aspect: simple or progressive
"""


class DataManager(object):
    def __init__(self, args = None):
        # assign input to class-variables
        if args is None:
            train_split          = 0.133
            self.n_used_tuples   = 300
            self.repr_type       = "one-hot"
        else:
            train_split          = args.train_split
            self.n_used_tuples   = args.n_used_tuples
            self.repr_type       = args.repr_type

             
        self.dataset_zip = np.load(DATA_PATH+"dSentences.npz",encoding='latin1')

        self.sentences_array     = self.dataset_zip["sentences_array"]
        self.latents_classes     = self.dataset_zip['latents_classes']
        self.latents_classes_with_idx = np.hstack((self.latents_classes,np.arange(self.latents_classes.shape[0]).reshape(-1,1)))
        self.metadata_dict       = self.dataset_zip['metadata'][()]
        self.latents_sizes       = self.metadata_dict['latent_sizes']

        
        self.n_generative_factors = self.latents_classes.shape[1]
        ### Same order as in dataset :)
        self.list_of_gf_strings = ["verb_obj_tuple","obj_sing_pl","gender",
                                   "subj_sing_pl","sent_type","nr_person",
                                   "pos_neg_verb","verb_tense","verb_style"]

        self.n_tuples            = self.latents_classes.max() + 1        

        self.n_syntaxes          = np.product(self.latents_sizes[1:])
        self.n_train_syntaxes    = int(self.n_syntaxes * train_split)
        self.n_test_syntaxes     = (self.n_syntaxes - self.n_train_syntaxes)

        # calculate actual train_split due to rounnding little bit different then input train_split values
        self.train_split = self.n_train_syntaxes / self.n_syntaxes
        
        #now let us load dataset, get sentence length, embedding matrix and dictionaries
        self.sent_len, self.original_sentences, self.dataset, self.id_to_token, self.token_to_id, self.embedding_matrix = self._load()

        #now let us create our train and test set latents classes
        self.train_latents_classes_with_idx, self.test_latents_classes_with_idx = self._create_train_and_test_latents_classes()        
        #now let us fetch their indices regarding the raw dataset (as above with self.dataset or self.latents_classes which share some indices correspondence)
        print("Finished creating train/test latents classes arrays.")

        self.train_indices, self.test_indices = self._get_indices_from_latents_classes()
        #calculate actual number of train_samples, test_samples and total_samples (by only using used_tuples (e.g. 300) out of all tuples (1100)
        self.train_samples = len(self.train_indices)
        self.test_samples = len(self.test_indices)
        self.n_samples = self.train_samples + self.test_samples

        print("Finished creating train/test dataset indices. Now creating X_train.")        

        #finally get our train and test set with embedded representation (either one-hot,fasttext or glove embedding)
        self.X_train = self.dataset[self.train_indices]
        print("Finished creating X_train. Now creating X_test.")
        self.X_test = self.dataset[self.test_indices]
        print("Finished creating X_test. Process completed.")
        # determine input dim of a single complete sentence
        self.input_dim = self.X_train.shape[1]
        #calculate actual number of train_samples, test_samples and i
        #total_samples (by only using used_tuples (e.g. 300) out of all tuples (1100)
        self.train_samples = self.X_train.shape[0]
        self.test_samples = self.X_test.shape[0]
        self.n_samples = self.train_samples + self.test_samples       


    def _load(self):
        # Load sentences as array and convert them from byte strings to regular utf-8 strings and finally add 
        # padding in order to do comparisons between original sentences and reconstruction sentences
        list_of_strings = [sentence.decode('utf-8') for sentence in self.sentences_array]
        #conversion of list of strings to list of lists containing tokens
        list_of_lists = [string.split() for string in list_of_strings]
        #determine length of longest sentence
        max_len_sentence = max([len(sublist) for sublist in list_of_lists])
        #first copy of list_of_lists in case we might need the original list
        padded_list_of_lists = copy.deepcopy(list_of_lists)
        #define string which indicates padding

        if self.repr_type == "one-hot":
            #define string which indicates padding
            pad = "."
            #pad all sublists with the paddding string in order to make them same length
            #note that this happens in place!
            #no perform the extension of pad-string IN-PLACE!
            for sublist in padded_list_of_lists:
                sublist.extend( (max_len_sentence - len(sublist)) \
                            * [pad])
            original_sentences = padded_list_of_lists
            print("Loading dataset!")
            dataset = np.load(DATA_PATH+"one-hot_dataset.npy")
            # now load the dictionaries in order to reconstruct sentences

            print("Loading id - token dictionaries!")
            id_to_token = np.load(DATA_PATH + "one-hot_id_to_token_dict.npy").tolist()
            token_to_id = np.load(DATA_PATH + "one-hot_token_to_id_dict.npy").tolist()
            # just pro-forma adding a value for embedding_matrix, it is not going to be used
            embedding_matrix = np.ones(shape = (1,1)) #shape will be: (1,1) # so we can extract the shape of it :)
        
        ### FastText Embeddings of Dimension 300
        if self.repr_type == "word2vec300d":
            #define string which indicates padding
            pad = "."
            #pad all sublists with the paddding string in order to make them same length
            #note that this happens in place!
            #no perform the extension of pad-string IN-PLACE!
            for sublist in padded_list_of_lists:
                sublist.extend( (max_len_sentence - len(sublist)) * [pad])
            original_sentences = padded_list_of_lists
            print("Loading dataset!")
            dataset = np.load(DATA_PATH+"glove300d_dataset_dot_padded.npy")
            # now load the gensim model in order to find most similar word and reconstruct sentences
            print("Loading Embedding-Model Matrix...")
            embedding_matrix = np.load(DATA_PATH + "glove300d_embedding_matrix.npy")
            id_to_token = np.load(DATA_PATH + "glove300d_id_to_token_dict.npy").tolist()
            token_to_id = np.load(DATA_PATH + "glove300d_token_to_id_dict.npy").tolist()

        ### Glove Embeddings of Dimension 50
        if self.repr_type == "word2vec50d":
            #define string which indicates padding
            pad = "."
            #pad all sublists with the paddding string in order to make them same length
            #note that this happens in place!
            #no perform the extension of pad-string IN-PLACE!
            for sublist in padded_list_of_lists:
                sublist.extend( (max_len_sentence - len(sublist)) * [pad])
            original_sentences = padded_list_of_lists
            print("Loading dataset!")
            dataset = np.load(DATA_PATH+"glove50d_dataset_dot_padded.npy")
            # now load the gensim model in order to find most similar word and reconstruct sentences
            print("Loading Embedding-Model Matrix...")
            embedding_matrix = np.load(DATA_PATH + "glove50d_embedding_matrix.npy")
            id_to_token = np.load(DATA_PATH + "glove50d_id_to_token_dict.npy").tolist()
            token_to_id = np.load(DATA_PATH + "glove50d_token_to_id_dict.npy").tolist()


        return max_len_sentence, original_sentences, dataset, id_to_token, token_to_id, embedding_matrix

        
        
    def _create_train_and_test_latents_classes(self):
        all_tuple_indices = np.arange(self.n_tuples)
        selected_tuple_indices = np.random.choice(all_tuple_indices, replace = False, size = self.n_used_tuples)
        
        tuple_selected_latents_classes_with_idx = self.latents_classes_with_idx[np.isin(self.latents_classes_with_idx[:,0],selected_tuple_indices)]
        # [:,0] because first column contains tuple-ids
        # self.n_generative_factors+1 because last column represents idx of original latent vector -> will be used when fetching X_train/X_test with indices
        reshaped_tuple_selected_latents_classes_with_idx = tuple_selected_latents_classes_with_idx.reshape((self.n_used_tuples, -1, self.n_generative_factors + 1))      
        train_set = []
        test_set = []
        for entry in reshaped_tuple_selected_latents_classes_with_idx:
            all_syntax_indices = np.arange(self.n_syntaxes)
            random_syntax_indices_train_set = np.random.choice(all_syntax_indices,size=self.n_train_syntaxes,replace=False) #replace=False: for only-unique values
            random_syntax_indices_test_set = np.delete(all_syntax_indices,random_syntax_indices_train_set)
            train_set.append(entry[random_syntax_indices_train_set,:])
            test_set.append(entry[random_syntax_indices_test_set,:])
        train_latents_classes_with_idx = np.array(train_set).reshape(-1, self.n_generative_factors + 1)
        test_latents_classes_with_idx = np.array(test_set).reshape(-1, self.n_generative_factors + 1)
        return train_latents_classes_with_idx, test_latents_classes_with_idx

    
    def _get_indices_from_latents_classes(self):
        train_indices = self.train_latents_classes_with_idx[:,-1]
        test_indices = self.test_latents_classes_with_idx[:,-1]        
    
        return train_indices, test_indices

    def get_samples_by_indices(self, indices):
        samples = self.dataset[indices]
        return samples



    @property
    def train_size(self):
        return self.train_samples

    @property
    def test_size(self):
        return self.test_samples

    @property
    def sample_size(self):
        return self.n_samples

    ### This function returns the sentence as a list, each list contains all words(strings) of the sentence
    def ground_truth(self, indices):
        ground_truth_sentences = []
        for index in indices:
            sentence = self.original_sentences[index]
            ground_truth_sentences.append(sentence)
        return ground_truth_sentences


    # indices are the batch indices (i.e. defines sampels we will be using from one_hot_dataset)
    def get_sentences(self, indices):
        sentences = []
        for index in indices:
          sentence = self.dataset[index]
          sentences.append(sentence)
        sentences = np.asarray(sentences)
        return sentences

    # this function get_sentence takes as input the generative factors, 
    # calculates the index in the raw-dataset and return the sentence as np.array
    def get_sentence(self, tuple_id = 0, object_sing_pl = 0, sentence_type = 0,
                           gender = 0, pronoun_sing_pl = 0, pronoun_1st_2n_3rd = 0,
                           pos_neg_verb = 0, tense_time = 0, tense_style = 0):

        latents = [tuple_id, object_sing_pl, sentence_type, gender, pronoun_sing_pl,
                   pronoun_1st_2n_3rd, pos_neg_verb, tense_time, tense_style]
        #given those latents, our job is to determine the idx in order to fetch
        #the correct sentence (row) in the dataset containing all sentences
        index = (self.latents_classes == latents).all( axis = 1).nonzero()[0][0]
        #source : https://stackoverflow.com/questions/25823608/find-matching-rows-in-2-dimensional-numpy-array
        return self.get_sentences([index])[0] #[0] because it returns an array as it usually is used to create an numpy array


    # this function return randomly number(amount) of n_sentences
    def get_random_sentences(self, n_sentences):
        indices = [np.random.randint(self.n_samples,dtype=np.uint32) for i in range(n_sentences)]
        return self.get_sentences(indices)

### ------------------------------------------------------------------------------------------------------------------------------------

    # this function returns L sentences given a fixed factor k (0-8) [9values] and fixed form value
    # L is the number of samples we use to generate a single vote
    def get_L_sentences(self, L,factor_index, form_value):
        ### latents_classes looks like this: [tuple_id (1100),object_sing./pl. (2),sent_type (2),gender (2),subj._sing./pl. (2),#person_1st/2nd/3rd (3),pos./neg. (2),verb_tense (3),verb_style (2)]
        ### or their max value entries(per se the form values you can choose including the last number)  [1099,    1,    1,    1,    1,    2,    1,    2,    1]
        column = self.latents_classes[:,factor_index]
        all_indices = np.squeeze(np.argwhere(column == form_value))
        np.random.shuffle(all_indices)
        L_indices = all_indices[:L]
        L_sentences = self.get_sentences(L_indices)
        return L_sentences
