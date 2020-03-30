#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 21:40:20 2018

@author: Amine
"""

import tensorflow as tf
import os
import numpy as np
from data_manager import DataManager
import argparse
import utils
import copy


### force this Evaluation to happen on CPU with CUDA_VISIBLE_DEVICES = "" / or "-1"
os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

PATH_ALL_CHECKPOINTS = "/usr/itetnas03/data-tik-01/aminem/logs_and_checkpoints/checkpoints/"

def create_session():
    # clean up the graph in case it is not empty
    tf.reset_default_graph()
    sess = tf.Session()
    return sess

### Load meta graph and restore weights
def setup_graph(session, experiment_name):
    # define full path folder name 
    folder_name = PATH_ALL_CHECKPOINTS + experiment_name 
    # the first thing to do when restoring a TensorFlow model is to load the 
    # graph structure from the ".meta" file into the current graph.
    # extract .meta file name in current folder PATH_ALL_CHECKPOINTS + experiemtn_name
    meta_file_name = [name for name in os.listdir(folder_name) if name.endswith(".meta")][0] #[0] because there is only a single meta file in each folder
    print(meta_file_name)
    # recreate the network using tf.train.import() function 
    # therfore let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(meta_graph_or_file=folder_name + "/" + meta_file_name)
    saver.restore(session, tf.train.latest_checkpoint(folder_name))
    # now lets grab the graph which is the current default_graph
    graph = tf.get_default_graph()
    return graph, saver

def get_graph_tensor(graph, tensor_name):
    runable_tensor = graph.get_tensor_by_name(tensor_name)
    return runable_tensor

def get_graph_tensors(graph, output_tensor_names):
    runable_tensor_list = []
    for tensor_name in output_tensor_names:
        runable_tensor_list.append(get_graph_tensor(graph,tensor_name))
    return runable_tensor_list

def create_feed_dict(feed_names, feed_values):
    # e.g. In [5]: feed_tensor_list = ["A", "B", "C"]
    #      In [6]: feed_values = [ 1,   2,   3 ]
    #      In [7]: feed_dict = dict(zip(feed_names, feed_values))
    feed_dict = dict(zip(feed_names, feed_values))
    return feed_dict


### Feed values should contain xs and phase(as set to False)
def transform(graph, sess, feed_values):
    feed_tensor_names = ["x:0", "phase:0"]
    output_tensor_names = ["vae/reparametrize/hard_concatenate:0"]
    runable_tensor_list = get_graph_tensors(graph = graph, output_tensor_names = output_tensor_names )
    # x -> concatenated complete sample and single latents; each ideally describing another generative factor
    feed_dict = create_feed_dict(feed_names = feed_tensor_names, feed_values = feed_values)
    complete_samples = sess.run(runable_tensor_list, feed_dict = feed_dict)
    ### [0] because complete_samples is a list of arrays (in this case only a single entry in the list)
    return complete_samples[0]


def generate(graph, sess, feed_values):
    # we feed the decoder from the soft_concatenate tensor node but the actual values are from the hard_concanate sampling node
    feed_tensor_names = ["vae/reparametrize/soft_concatenate:0", "embedding_matrix:0"]
    runable_tensor_list = get_graph_tensors(graph = graph, output_tensor_names = ["accuracy/prediction:0"])
    feed_dict = create_feed_dict(feed_names = feed_tensor_names, feed_values = feed_values)
    predictions = sess.run(runable_tensor_list, feed_dict = feed_dict)
    if type(predictions) == list:
        predictions = predictions[0]
    else:
        pass
    # finally we have to squeeze predictions in order to avoid nested dimensions
    predictions = np.squeeze(predictions)
    return predictions 

def pick_complete_sample_by_index(complete_samples, indices):
    ### Returned argument must be a array consisting of representations of z as arrays
    complete_sample = complete_samples[indices,None]
    return complete_sample


def traverse_single_dimension(complete_sample,traversal_array,idx): 
    ### Determine # of traversals for given traversal_array (count of rows or columns)
    n_traversals = traversal_array.shape[0]

    vstacked_complete_sample = np.vstack([complete_sample]*n_traversals)
    #generates a list of splits per latent dimension    
    vstacked_split = np.split(vstacked_complete_sample,args.split_indices,axis=1)
    ### Insert traversal_array on index position
    vstacked_split[idx] = traversal_array
    ### From list back to array
    traversed_complete_sample = np.concatenate(vstacked_split,axis=1)
    ### generate() fucntion expects input of dimension (samples x z_dim) which is actually the same shape as single sample (1,sum_of_one-hot_dimensions)
    return traversed_complete_sample


#the function below will perform latent traversal and print the latent traversals effect (as sentences i.e. visually)
def traverse_all_dim(graph, sess, id_to_token_dict, embedding_matrix, complete_sample, args):

    ### Fetch traversal_arrays
    traversal_arrays = args.traversal_arrays
    ### First create the regular reconstruction of the network
    untraversed_output = generate(graph, sess, feed_values= [complete_sample, embedding_matrix])
    print("Original reconstruction of untraversed sample representation:")
    #we need to insert a list of z_representations into the decode_predictions function
    if type(untraversed_output) == list:
        pass
    else:
        untraversed_output = [untraversed_output]
    utils.decode_predictions(untraversed_output,id_to_token_dict)
    ### Now perform the latent traversals
    for idx in range(len(args.latents_dimensions)):
        ### Generate copy of complete_sample to not modify it
        copy_complete_sample = copy.deepcopy(complete_sample)
        ### Fetch correct traversal array by index :)
        traversal_array = traversal_arrays[idx]
        traversed_sample = traverse_single_dimension(copy_complete_sample,traversal_array,idx)
        traversed_outputs = generate(graph, sess, feed_values = [traversed_sample, embedding_matrix])
        print("Traversals in latent dimension{} of size {}:".format(idx+1,args.latents_dimensions[idx]))
        utils.decode_predictions(traversed_outputs,id_to_token_dict)



def main(args):
    
    # create manager
    print("Loading data manager")
    manager = DataManager(args)
    print("Manager loading completed")

    #first lets create a session
    sess = create_session()
    
    #next let us load the graph and its variables
    print("Loading and constructing graph and model ...")
    graph, saver = setup_graph(session = sess, experiment_name = args.experiment_name)
    print("Model loading and construction finished!")
    
    #get necessary network inputs and other structures from manager
    X                 = manager.dataset
    #lets pick up the indices
    train_indices     = manager.train_indices
    test_indices      = manager.test_indices
    embedding_matrix  = manager.embedding_matrix
    id_to_token       = manager.id_to_token

    #perform transformation: input x -> hidden-representation z
    print("Transforming input to hidden")
    my_timer = utils.Timer() 
    complete_samples = transform(graph, sess, feed_values = [X, False])
    past_time = my_timer.get_past_time()
    print("Transformation duration: %s" % past_time)

    print("This is what a single entry of complete_samples looks like:",complete_samples[0])

    np.random.seed()

    train_idx = int(np.random.choice(a = train_indices, size =  1, replace = False))
    test_idx  = int(np.random.choice( a = test_indices,  size  = 1, replace = False))

    picked_train_sample = pick_complete_sample_by_index(complete_samples, indices = train_idx)
    picked_test_sample  = pick_complete_sample_by_index(complete_samples, indices = test_idx)
  

    # LATENT TRAVERSAL WITH TRAINED DATA SAMPLE
    print("\nCase: Training data traversal:")
    traverse_all_dim(graph, sess, id_to_token, embedding_matrix,picked_train_sample,args)
    
    # LATENT TRAVERSAL WITH UNKNOWN DATA SAMPLE
    print("\nCase: Testing data traversal:")
    #NOW LETS DO THE SAME BUT LATENT TRAVERSEL WITH TESTING DATA, WHICH MEANS WE HAVE NOT TRAINED ON IT, LETS SEE IF IT CAN GENERALIZE
    traverse_all_dim(graph, sess, id_to_token, embedding_matrix, picked_test_sample,args)
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Joint-VAE latent traversal evaluation')

 
    parser.add_argument('--experiment_name', default='word2vec300d_gamma_19.0_lr_1e-05_n_tuples_2_n_epoch_500/', type=str, help='name of the experiment we want to evaluate')
    parser.add_argument('--trav_steps', default=100, type=int, help='number of traversal steps in z_i_interval used for sentence interpolation/traversal')     
    parser.add_argument('--train_split', default=0.133, type=float, help='what fraction of dataset for training purpose')
    parser.add_argument('--n_used_tuples', default=3, type=int, help='how many (verb,object)-tuples should training and testing set contain (max. is 1100)')
    parser.add_argument('--n_z_dims', default=8, type=int, help='Denotes the number of one-hot vectors in the latent code. E.g. 9 one-hot vectors for 9 generative factors')


    ### Now lets create the args Namespace (which is in effect a dicitionary)
    args = parser.parse_args()
    if "one-hot" in args.experiment_name:
        args.repr_type = 'one-hot'
    elif "word2vec50d" in args.experiment_name:
        args.repr_type = 'word2vec50d'
    elif "word2vec300d" in args.experiment_name:
        args.repr_type = 'word2vec300d'

    ### Define dimensions of generative factors latent variables
    args.latents_dimensions = np.array([1200]+ [5] * args.n_z_dims)
    ### Define split indices (w/o last index which is the sum of all latens together)
    args.split_indices = np.cumsum(args.latents_dimensions)[:-1]
    ### Create traversals arrays in a list
    traversal_arrays = []
    for latent_dimension in args.latents_dimensions:
        traversal_arrays.append(np.eye(latent_dimension))
    args.traversal_arrays = traversal_arrays

    main(args)
