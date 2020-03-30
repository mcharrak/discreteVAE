##############################################################################
#this block is GPU server specific adjustment for tensorflow
import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #Do not assign whole gpu memory, just use it on the go
config.allow_soft_placement = True #If a operation is not define it the default device, let it execute in another.
#config.log_device_placement=True
##############################################################################

from beta_vae import VAE
from data_manager import DataManager
import utils #in order to calculate disentanglement score
import argparse
import numpy as np
from utils import str2bool
  
def main(args, config):
    
    #setting configurations for either gpu (on cluster) or cpu (on local machine)        
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU'].replace("\n",",")
    else:
        pass
    seed = args.seed

    # set random seed for numpy ops
    np.random.seed(seed)
    # set random seed for tensorflow graph
    # must set random seed before creating session.
    # setting seed comes from this GitHubGist https://gist.github.com/tnq177/ce34bcf6b20243b0b5b23c78833e7945
    tf.reset_default_graph()
    tf.set_random_seed(seed)
 
    print("Creating DataManager and loading training and testing dataset")
    manager = DataManager(args)

    sess = tf.Session(config=config)
    print("Creating model...")
    model = VAE(args, manager)
    print("Model created!")
    
    print("running vars:", model.running_vars)
    
    sess.run(tf.global_variables_initializer())
    # initialize/reset all running variables
    sess.run(model.running_vars_initializer)

    #create saver for checkpoints saving
    saver = utils.create_checkpoint_saver(sess, args, model)
    print("Will Start Training Now")
    model.train(sess, manager, saver, args)
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Vanilla-Beta-VAE')

    #define what dataset representation to use
    parser.add_argument('--repr_type', default='word2vec50d', type=str, help='use of one-hot or  word2vec embedding; options are <one-hot> and <word2vec>')
    parser.add_argument('--enc_type', default='mlp', type=str, help='use of mlp,cnn or rnn as encoder architecture')
    parser.add_argument('--dec_type', default='mlp', type=str, help='use of mlp,cnn or rnn as decoder architecture')
    parser.add_argument('--train_split', default=0.75, type=float, help='what fraction of dataset for training purpose')
    parser.add_argument('--n_used_tuples', default=1100, type=int, help='how many (verb,object)-tuples should training and testing set contain (max. is 1100)')

    parser.add_argument('--seed', default=50, type=int, help='random seed')
    parser.add_argument('--name', default='main', type=str, help='name of the experiment')
    parser.add_argument('--gpu', default=True, type=str2bool, help='enable gpu mode set to True or False or also 1 or 0')
    parser.add_argument('--epoch_size', default=100, type=int, help='epoch size')
    parser.add_argument('--batch_size', default=50, type=int, help='batch size')
    parser.add_argument('--keep_n_ckpts', default = 1, type=int, help='number of checkpoints we want to keep, if 1, then we only store the latest checkpoint')
    parser.add_argument('--prun_thold', default= 0.25, type=float, help='pruning threshold for sigma values, only latents with smaller sigma are allowed to vote')

    parser.add_argument('--z_dim', default=13, type=int, help='dimension of the representation z')
    parser.add_argument('--activation_fn', default="elu", type=str, help='choose activation function (elu or relu)')


    parser.add_argument('-enc_units_per_layer', action='store', dest='enc_units_per_layer_list',
                        type=int, nargs='*', default=[512, 256],
                        help="Length of the list: # of intermediate layers. Each list-entry: # of units. Example Usage: -enc_units_per_layer 256 1024")
    parser.add_argument('-dec_units_per_layer', action='store', dest='dec_units_per_layer_list',
                        type=int, nargs='*', default=[256, 512],
                        help="Length of the list: # of intermediate layers. Each list-entry: # of units. Example Usage: -dec_units_per_layer 256 1024")


    parser.add_argument('-filters_per_layer', action='store', dest='filters_per_layer_list',
                        type=int, nargs='*', default=[256, 512, 1024],
                        help="Number of filters in each layer when using convolutional encoder/decoder. Each list-entry: # of filters. Example Usage: -filters_per_layer 512 256 128")

    parser.add_argument('-kernel_width_per_layer', action='store', dest='kernel_width_per_layer_list',
                        type=int, nargs='*', default=[4, 3, 2],
                        help="This is the kernel width per layer which does 1D convolution. Where the original order is used for the encoder, meaning wider kernel at the first layer and reduced kernel width at final layer. Each list-entry: denotes kernel width. Example Usage: -kernel_width_per_layer 4 3 2")


    parser.add_argument('--beta', default=0.25, type=float, help='beta parameter for KL-term in original beta-VAE')    
    parser.add_argument('--lr', default=5.0e-5, type=float, help='learning rate')

    parser.add_argument('--checkpoints_dir', default="logs_and_checkpoints/checkpoints/" , type=str, help='checkpoints directory')
    parser.add_argument('--logs_dir', default="logs_and_checkpoints/log/" , type=str, help='log files directory')    
    

    args = parser.parse_args()

    main(args, config)  
