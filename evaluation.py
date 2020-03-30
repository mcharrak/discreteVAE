import tensorflow as tf
import os
import numpy as np
from data_manager import DataManager
import argparse
import utils


### force this Evaluation to happen on CPU with CUDA_VISIBLE_DEVICES = "" / or "-1"
os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

PATH_ALL_CHECKPOINTS = "/usr/itetnas03/data-tik-01/aminem/logs_and_checkpoints/checkpoints/"


def create_session():
    # clean up the graph in case it is not empty
    tf.reset_default_graph()
    sess = tf.Session()
    return sess

# load meta graph and restore weights
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


# feed values should contain xs and phase(as set to False)
def transform(graph, sess, feed_values):
    feed_tensor_names = ["x:0", "phase:0"]
    runable_tensor_list = get_graph_tensors(graph = graph, output_tensor_names = ["vae/reparametrize/z:0", "vae/z_mean:0", "vae/z_sigma:0"] )
    # x -> z, z_mean, z_log_sigma
    feed_dict = create_feed_dict(feed_names = feed_tensor_names, feed_values = feed_values)
#    print(feed_dict)
    z, z_mean, z_sigma = sess.run(runable_tensor_list, feed_dict = feed_dict)
    return z, z_mean, z_sigma

# z has dimensions (samples x z_dim)
def pick_single_z_by_index(z, indices):
    #the returned argument must be a array consisting of representations of z as arrays
    single_z = z[indices,None]
    return single_z

# z_mean is of shape ( samples x z_dim )
# single_z_mean is of shape (1 x z_dim )
def pick_single_z_mean_by_index(z_mean, indices):
    #the returned array is the mean vector of the z representation at index position of z
    single_z_mean = z_mean[indices, None]
    return single_z_mean

def generate(graph, sess, feed_values):
    feed_tensor_names = ["vae/reparametrize/z:0", "embedding_matrix:0"]
    runable_tensor_list = get_graph_tensors(graph = graph, output_tensor_names = ["accuracy/prediction:0"])
    feed_dict = create_feed_dict(feed_names = feed_tensor_names, feed_values = feed_values)
#    print(feed_dict)
    predictions = sess.run(runable_tensor_list, feed_dict = feed_dict)
    if type(predictions) == list:
        predictions = predictions[0]
    else:
        pass
    # finally we have to squeeze predictions in order to avoid nested dimensions
    predictions = np.squeeze(predictions)
    return predictions    


# traversal range is from z_mean (OF THE TEST SAMPLE Z_TEST) +- 3*z.std(axis=0) (in effect std of z along each of its 20 dimensions)
def calculate_traversal_range(single_z_mean, z_std_dimensionwise, trav_steps, z, AE_setup, perc_cutoff = 0.1):

    n_cutoffs_per_side = int(trav_steps * perc_cutoff)

    if AE_setup:
        z_mins = z.min(axis = 0)
        z_maxs = z.max(axis = 0)
        range_interavals = list(zip(z_mins,z_maxs)) 
        rounded_range_intervals = []
        for range_interval in  range_interavals:
            min_rounded = round(range_interval[0])
            max_rounded = round(range_interval[1])        
            rounded_range_intervals.append((min_rounded,max_rounded))
        list_of_traversal_arrays = []
        for entry in rounded_range_intervals:
            traversal_array = np.linspace(start = entry[0], stop = entry[1], num = trav_steps)
            # now let us remove perc_cutoff( in %) of of all values from the left border and from right border),
            # as they lead to very instable reconstructions
            # we won't perform the cutoff at the border because we try to cover the whole space and therfore need those entries
#            traversal_array = traversal_array[n_cutoffs_per_side: -n_cutoffs_per_side]
            # finally append the traversal array to the list of traversal_arrays
            list_of_traversal_arrays.append(traversal_array)


    else:
        # first caluclate the max deviation from basis(mean) by using 3*std() per dimension 
        three_z_std_dimensionwise = 3*z_std_dimensionwise # these values are now our max values
#        print("z_std:", three_z_std_dimensionwise)
        list_of_traversal_arrays = []
#        print("Sinlge_z_mean has the following appearance {} and the shape is {} and the single element z_mean[0] looks like this {} and has the following shape {}".format(single_z_mean, single_z_mean.shape, single_z_mean[0], single_z_mean[0].shape))
        for index, entry in np.ndenumerate( three_z_std_dimensionwise.squeeze() ): #alternatively for i in range(len(three_z_std_dimensionwise)) and accessing values with three_z_s_dimensionwise[i]....
            idx = index[0]
#            print("index from np.ndenumerate should be a tuple", index)
#            print("idx, by fetching 0th element", idx)
            min_value = (-1.0) * entry
            max_value = (+1.0) * entry
#            print("minValue", min_value)
#            print("maxValue", max_value)
            #adding the mean value of the range of sigma*eps values which are possible (remember z=mean+eps*1*sigma, so here we use 3*sigma as a buffer)
            # because single_z_mean has shape (1,20) we need to acces first row and than the specific entries i.e. columns)
            traversal_array = single_z_mean[0][idx] + np.linspace(start = min_value, stop = max_value, num = trav_steps)
            # now let us remove perc_cutoff( in %) of of all values from the left border and from right border),
            # as they lead to very instable reconstructions
            traversal_array = traversal_array[n_cutoffs_per_side: -n_cutoffs_per_side]
            # finally append the traversal array to the list of traversal_arrays
            list_of_traversal_arrays.append(traversal_array)


    return list_of_traversal_arrays

def traverse_single_dim(single_z_mean, latent_dim_idx, list_of_traversal_arrays, z): 
    #single_z_mean is a numpy array
    traversal_values = list_of_traversal_arrays[latent_dim_idx]
    traversed_sentences_in_z = []
    for traversal_value in traversal_values:
        traversed_sentence_in_z = single_z_mean.copy()
        #now lets change the value of the latent_dim_idx
        traversed_sentence_in_z[:,latent_dim_idx] = traversal_value
        traversed_sentences_in_z.append(traversed_sentence_in_z)
    traversed_sentences_in_z = np.asarray(traversed_sentences_in_z).squeeze() #squeze beacause else we have cumbersome dimension of (samples x 1 x z_dim) which we cannot feed into generate() fucntion which expects input of dimension (samples x z_dim) which is actually the same shape as z.shape
    return traversed_sentences_in_z


#the function below will perform latent traversal and print the latent traversals effect (as sentences i.e. visually)
def traverse_all_dim(graph, sess, id_to_token_dict, embedding_matrix, single_z_mean, list_of_traversal_arrays, z):
 
    num_dim = len(list_of_traversal_arrays)
    original_output = generate(graph, sess, feed_values= [single_z_mean, embedding_matrix])
    print("Original reconstruction of underlying sample z representation:")
    #we need to insert a list of z_representations into the decode_predictions function
    if type(original_output) == list:
        pass
    else:
        original_output = [original_output]

    utils.decode_predictions(original_output,id_to_token_dict)
    for latent_dim_idx in range(num_dim):
        traversed_sentences_in_z = traverse_single_dim(single_z_mean, latent_dim_idx, list_of_traversal_arrays, z)
        traversed_outputs = generate(graph, sess, feed_values = [traversed_sentences_in_z, embedding_matrix])
        print("Traversals in z_dim{}:".format(latent_dim_idx))
        utils.decode_predictions(traversed_outputs,id_to_token_dict)



def main(args):
    ### Given the args.experiment_name we have to set the right input for data-manager
    if "one-hot" in args.experiment_name:
        args.repr_type = 'one-hot'
    elif "word2vec50d" in args.experiment_name:
        args.repr_type = 'word2vec50d'
    elif "word2vec300d" in args.experiment_name:
        args.repr_type = 'word2vec300d'
    else:
        print("wrong experiment_name!")

    ### Create manager
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
    z, z_mean, z_sigma = transform(graph, sess, feed_values = [X, False])
    past_time = my_timer.get_past_time()
    print("Transformation duration: %s" % past_time)

    # calculate z_std_dimensionwise for the latent traversals
    z_std_dimensionwise = z.std(axis = 0, keepdims = True)  
    # calculate average z_sigma values to get an idea of collapsed/non-collapsed latents
    z_sigma_average = z_sigma.mean(axis = 0, keepdims = True)

    if "_0.0_" in args.experiment_name:
        AE_setup = 1
        print("Doing traversals evaluation based on z-ranges (min/max) because given experiment is an AE setup!") 
    else:
        AE_setup = 0
        print("Doing traversals evaluation based on multiples of z-std because given experiment in an VAE setup!")
         #calculate z_std_dimensionwise for the latent traversals
        # statistics information
        print("Overview of what the z_std-values in each dimension look like:", z_std_dimensionwise)
        print("In order to get an idea which dimensionare collapsed and which are being used \nbelow an overview of what averaged z_sigma-values in each dimension look like:", z_sigma_average)


    train_idx = int(np.random.choice(a = train_indices, size =  1, replace = False))
    test_idx  = int(np.random.choice( a = test_indices,  size  = 1, replace = False))

    picked_z_train_mean = pick_single_z_mean_by_index(z_mean, indices = train_idx)
    picked_z_test_mean  = pick_single_z_mean_by_index(z_mean, indices = test_idx)
  



    #get traversal values by calculating min &  max values and rounding to closest integer value
    list_of_traversal_arrays = calculate_traversal_range(picked_z_train_mean, z_std_dimensionwise, args.trav_steps, z, AE_setup)
    print("\nCase: training data traversal:")
    print("Values we will traverse for in each latent dimension:", list_of_traversal_arrays)
    # LATENT TRAVERSAL WITH TRAINED DATA SAMPLE
    # printing reconstruction of underlying original sample-z (not-traversed) and 
    # printing all possible traversals of that sample-z afterwards with the following function traverse_all_dim
    traverse_all_dim(graph, sess, id_to_token, embedding_matrix,  picked_z_train_mean, list_of_traversal_arrays, z)
    


    #get traversal values by calculating min &  max values and rounding to closest integer value
    list_of_traversal_arrays = calculate_traversal_range(picked_z_test_mean, z_std_dimensionwise, args.trav_steps, z, AE_setup)
    print("\nCase: testing data traversal:")
    print("Values we will traverse for in each latent dimension:", list_of_traversal_arrays)
    # LATENT TRAVERSAL WITH UNKNOWN DATA SAMPLE
    #NOW LETS DO THE SAME BUT LATENT TRAVERSEL WITH TESTING DATA, WHICH MEANS WE HAVE NOT TRAINED ON IT, LETS SEE IF IT CAN GENERALIZE
    traverse_all_dim(graph, sess, id_to_token, embedding_matrix, picked_z_test_mean, list_of_traversal_arrays, z)

#   tensor_names we will need ["x:0", "phase :0", "vae/reparametrize/z:0", "vae/z_log_sigma_sq:0", "vae/z_mean:0"] 
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VAE latent traversal evaluation')

 
    parser.add_argument('--experiment_name', default='one-hot_beta_0.5_lr_3.5e-05', type=str, help='name of the experiment we want to evaluate')
    parser.add_argument('--trav_steps', default=100, type=int, help='number of traversal steps in z_i_interval used for sentence interpolation/traversal')     
    parser.add_argument('--train_split', default=0.133, type=float, help='what fraction of dataset for training purpose')
    parser.add_argument('--n_used_tuples', default=300, type=int, help='how many (verb,object)-tuples should training and testing set contain (max. is 1100)')
  
    # Now lets create the args Namespace (which is in effect a dicitionary)
    args = parser.parse_args()
    if "one-hot" in args.experiment_name:
        args.repr_type = 'one-hot'
    elif "word2vec50d" in args.experiment_name:
        args.repr_type = 'word2vec50d'
    elif "word2vec300d" in args.experiment_name:
        args.repr_type = 'word2vec300d'
    else:
        print("wrong experiment_name!")

    main(args)













