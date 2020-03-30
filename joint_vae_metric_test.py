#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:14:30 2018

@author: Amine
"""

import tensorflow as tf
import numpy as np
import utils
import shutil
import os
import copy
layers = tf.contrib.layers
import io
import matplotlib
### Set matplotlib not to use Xwindows backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPS = 1e-20

class VAE(object):
    def __init__(self,args,manager):
        """
        Class which defines model and forward pass.
        Parameters
        ----------
        latent_spec : dict
            Specifies latent distribution. For example:
            {'cont': 10, 'disc': [10, 4, 3]} encodes 10 normal variables and
            3 gumbel softmax variables of dimension 10, 4 and 3. A latent spec
            can include both 'cont' and 'disc' or only 'cont' or only 'disc'.
        temperature : float
            Temperature for gumbel softmax distribution.
        """
        ### define data parameters, extract these from manager input 
        self.input_dim = manager.input_dim
        self.sent_len = manager.sent_len
        self.emb_dim = int(self.input_dim / self.sent_len)   
        self.emb_matrix_shape = manager.embedding_matrix.shape
        self.train_split = args.train_split
        
        ### Define data parameters, extract these from args input 
        if args.activation_fn == "elu":
            self.activation_fn = tf.nn.elu
        if args.activation_fn == "relu":
            self.activation_fn = tf.nn.relu
        self.filters_per_layer_list = args.filters_per_layer_list
        self.kernel_width_per_layer_list = args.kernel_width_per_layer_list

        self.lr = args.lr
        self.disc_gamma = args.disc_gamma
        self.cont_capacity = tuple(args.cont_capacity)
        self.disc_capacity = tuple(args.disc_capacity)
        self.batch_size = args.batch_size
        self.epoch_size = args.epoch_size
        self.latent_spec = args.latent_spec
        self.code_dim_list =  args.z_dim_disc_list
        self.split_indices = np.cumsum(self.code_dim_list)[:-1]

        self.is_continuous = 'cont' in self.latent_spec
        self.is_discrete = 'disc' in self.latent_spec
        self.temp = args.temp
        self.enc_units_per_layer = args.enc_units_per_layer_list
        self.dec_units_per_layer = args.dec_units_per_layer_list
        self.repr_type = args.repr_type
        self.enc_type = args.enc_type
        self.dec_type = args.dec_type
        self.n_used_tuples = args.n_used_tuples

        ### Calculate dimensions of latent distribution
        self.latent_cont_dim = 0
        self.latent_disc_dim = 0
        self.num_disc_latents = 0
        if self.is_continuous:
            self.latent_cont_dim = self.latent_spec['cont']
        if self.is_discrete:
            self.latent_disc_dim += sum([dim for dim in self.latent_spec['disc']])
            self.num_disc_latents = len(self.latent_spec['disc'])
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim

        ### Give an experiment name
        self.experiment_name = (self.repr_type + '_gamma_' + str(self.disc_gamma) + 
                                '_lr_' + str(self.lr) + '_n_tuples_' + str(args.n_used_tuples) + "_enc_type_" + args.enc_type + "_dec_type_" + args.dec_type +
                                '_split_' + str(self.train_split) + '_n_epoch_' + str(self.epoch_size) + '_disc_dims_' + str(len(self.latent_spec["disc"])))

        ### Create autoencoder network
        self._create_network()
    
        ### Define loss function and corresponding optimizer
        self._create_loss_optimizer()


    def sample_normal(self,z_mean,z_logvar):
        """
        Samples from a normal distribution using the reparameterization trick.
        Parameters
        ----------
        z_mean : Mean of the normal distribution. Shape (B, D) where D is dimension
               of distribution.
        z_logvar : Diagonal log variance of the normal distribution. Shape (B, D)
        """
        with tf.variable_scope("reparametrize_cont"):
            eps_shape = tf.shape(z_mean)
            eps = tf.random_normal(shape=eps_shape,mean=0,stddev=1)
            z_std = tf.exp(0.5 * z_logvar)
            # z = z_mu + z_sigma( a.k.a z_std) * epsilon
            #z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps))
            z = z_mean + (z_std*eps)
            z_mean = z_mean
            return z, z_mean


    def sample_gumbel_softmax(self,alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.
        Parameters
        ----------
        alpha : Parameters of the gumbel-softmax distribution. Shape (B, D)
        Where D denotes # of classes/categories a specific discrete latent variable can take on
        """
        with tf.variable_scope("reparametrize_disc"):
            ### 1. Sample from gumbel distribution Gumbel(0,1)
            unif_shape = tf.shape(alpha)
            unif = tf.random_uniform(shape=unif_shape,minval=0,maxval=1)
            gumbel = -tf.log(-tf.log(unif + EPS) + EPS)
            
            # Reparameterize to create gumbel softmax sample
            log_alpha = tf.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temp
            gumbel_softmax_samples = tf.nn.softmax(logit,name="soft_gumble")
            
            ### In reconstruction mode, pick most likely sample
            max_values = tf.reduce_max(alpha,axis=1,keepdims=True)
            one_hot_samples = tf.cast(tf.equal(alpha,max_values),dtype=alpha.dtype,name="hard_gumble")       
            return gumbel_softmax_samples, one_hot_samples
    

    def reparametrize(self, latent_dist):
        """
        Samples from latent distribution using the reparametrization trick.
        Parameters
        ----------
        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both, containing the parameters
            of the latent distributions as torch.autograd.Variable instances.
        """
        with tf.variable_scope("reparametrize"):
            soft_samples = []
            ### Fetch parameters for reconstruction mode
            hard_samples = []
            if self.is_continuous:
                z_mean, z_logvar = latent_dist['cont']
                self.z,  self.z_mean= self.sample_normal(z_mean, z_logvar)
                soft_samples.append(self.z)
                hard_samples.append(self.z_mean)
                
                
            if self.is_discrete:
                for alpha in latent_dist['disc']:
                    soft_disc_sample, hard_disc_sample = self.sample_gumbel_softmax(alpha)
                    soft_samples.append(soft_disc_sample)
                    hard_samples.append(hard_disc_sample)
                
            ### Concatenate continuous and discrete samples into one large sample
            ### List of Tensors -> Single Tensors
            soft_complete_samples = tf.concat(soft_samples,axis=1, name="soft_concatenate")
            hard_complete_samples = tf.concat(hard_samples,axis=1, name="hard_concatenate")

        return soft_complete_samples, hard_complete_samples


    ### Returns the latent distributions like z_mean,z_logvar and discrete latents
    ### as a dictinoary with two keys
    def _create_mlp_encoder(self,x,reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):
            layer = x
            layer_nr = 1
            for units in self.enc_units_per_layer:
                 layer = layers.fully_connected(inputs = layer, num_outputs = units, 
                                                activation_fn = self.activation_fn, 
                                                scope = 'fc'+str(layer_nr))
                 layer_nr = layer_nr + 1         
            self.hidden   = layer

            latent_dist = {}
            self.alphas = []
            if self.is_continuous:              
                latent_dist["cont"] = []              
                self.z_mean     = layers.fully_connected(inputs=self.hidden,
                                                       num_outputs=self.latent_cont_dim,
                                                       activation_fn = None,
                                                       scope='z_mean')
                latent_dist["cont"].append(self.z_mean)              
                self.z_logvar  = layers.fully_connected(inputs=self.hidden,
                                                       num_outputs=self.latent_cont_dim,
                                                       activation_fn = None,
                                                       scope='z_logvar')              
                latent_dist["cont"].append(self.z_logvar)
                self.alphas.extend(latent_dist["cont"])
                
            if self.is_discrete:
                latent_dist["disc"] = []
                for (gf_idx,disc_dim) in enumerate(self.latent_spec["disc"]):
                    alpha   =     layers.fully_connected(inputs=self.hidden,
                                                         num_outputs=disc_dim,
                                                         activation_fn = None,
                                                         scope='gf_nr_'+str(gf_idx+1))
                    latent_dist["disc"].append(tf.nn.softmax(alpha))
                self.alphas.extend(latent_dist["disc"])
                
        return latent_dist



    def _create_cnn_encoder(self, x,reuse=False):
      with tf.variable_scope("enc", reuse=reuse):
        print("shape of x before reshaping",x.get_shape().as_list())

        x = tf.reshape(tensor = x, shape = [-1, self.emb_dim, self.sent_len, 1], name = "x_reshaped") #(Batch, 1924, 7, 1)
        print("shape of x after reshaping",x.get_shape().as_list())

        self.kernel_size_1 = [self.emb_dim, self.kernel_width_per_layer_list[0]] # [1924,4]
        print("Kernel size 1:", self.kernel_size_1)

        conv1 = tf.layers.conv2d(inputs = x, filters = self.filters_per_layer_list[0],
                               strides=1, padding = 'valid', kernel_size = self.kernel_size_1, activation=tf.nn.elu)

        self.conv1_shape = conv1.get_shape().as_list()
        print("Shape of conv1:", self.conv1_shape)

        self.kernel_size_2 = [ self.conv1_shape[1], self.kernel_width_per_layer_list[1]] # [1,3]
        print("Kernel size 2:", self.kernel_size_2)

        conv2 = tf.layers.conv2d(inputs = conv1, filters = self.filters_per_layer_list[1],
                               strides=1, padding = 'valid', kernel_size= self.kernel_size_2, activation=tf.nn.elu)

        self.conv2_shape = conv2.get_shape().as_list()
        print("Shape of conv2:", self.conv2_shape)

        self.kernel_size_3 = [ self.conv2_shape[1], self.kernel_width_per_layer_list[2]] # [1,2]
        print("Kernel size 3:", self.kernel_size_3)
        conv3       = tf.layers.conv2d(inputs = conv2, filters = self.filters_per_layer_list[2],
                                       strides=1, padding = 'valid', kernel_size= self.kernel_size_3, activation=tf.nn.elu)


        conv3       = tf.contrib.layers.flatten(conv3)
        self.hidden   = conv3

        latent_dist = {}
        self.alphas = []
        if self.is_continuous:
            latent_dist["cont"] = []
            self.z_mean     = layers.fully_connected(inputs=self.hidden,
                                                       num_outputs=self.latent_cont_dim,
                                                       activation_fn = None,
                                                       scope='z_mean')
            latent_dist["cont"].append(self.z_mean)
            self.z_logvar  = layers.fully_connected(inputs=self.hidden,
                                                       num_outputs=self.latent_cont_dim,
                                                       activation_fn = None,
                                                       scope='z_logvar')
            latent_dist["cont"].append(self.z_logvar)
            self.alphas.extend(latent_dist["cont"])

        if self.is_discrete:
            latent_dist["disc"] = []
            for (gf_idx,disc_dim) in enumerate(self.latent_spec["disc"]):
                alpha   =     layers.fully_connected(inputs=self.hidden,
                                                         num_outputs=disc_dim,
                                                         activation_fn = None,
                                                         scope='gf_nr_'+str(gf_idx+1))
                latent_dist["disc"].append(tf.nn.softmax(alpha))
            self.alphas.extend(latent_dist["disc"])

        return latent_dist




    ### Define MLP decoder        
    def _create_mlp_decoder(self,complete_sample,reuse=False):
        """
        Decodes sample from latent distribution into a sentence.
        Parameters
        ----------
        latent_sample : 
            Sample from latent distribution. Shape (B, L) 
            Where L is dimension of complete latent distribution.
        """        
        with tf.variable_scope("decoder", reuse=reuse):
            layer = complete_sample
            layer_nr = 1
            for units in self.dec_units_per_layer:
                 layer = layers.fully_connected(inputs = layer, num_outputs = units, 
                                                activation_fn = self.activation_fn, 
                                                scope = 'fc'+str(layer_nr))
                 layer_nr = layer_nr + 1                
            logit = layers.fully_connected(inputs = layer, num_outputs = self.input_dim, 
                                           activation_fn = None, scope='logit')
        return logit

### was reuse_tf.AUTO_REUSE
    def _create_cnn_decoder(self,complete_sample,reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):

            dim_features = self.filters_per_layer_list[0] ### this is equivalent to 512
            input_layer = complete_sample
            ### Blow up dimension from 20 (hidden_dim) to 128 (dimension of last encoder convolutional layer output)
            deconv0 = layers.fully_connected(inputs = input_layer, num_outputs = dim_features, activation_fn = None, scope="deconv_prep")
            deconv0 = tf.reshape(tensor = deconv0, shape = [-1,1,1,dim_features])
            print("Deconv0 shape:", deconv0.get_shape().as_list())

            deconv1 = tf.layers.conv2d_transpose(inputs = deconv0, filters = self.filters_per_layer_list[1],
                               strides=1, padding = 'valid', kernel_size= self.kernel_size_3, activation=tf.nn.elu)
            print("Deconv1 shape:", deconv1.get_shape().as_list())

            deconv2 = tf.layers.conv2d_transpose(inputs = deconv1, filters = self.filters_per_layer_list[2],
                               strides=1, padding = 'valid', kernel_size= self.kernel_size_2, activation=tf.nn.elu)
            print("Deconv2 shape:", deconv2.get_shape().as_list())

            deconv3 = tf.layers.conv2d_transpose(inputs = deconv2, filters = 1,
                               strides=1, padding = 'valid', kernel_size= self.kernel_size_1, activation=tf.nn.elu)
            print("Deconv3 shape:", deconv3.get_shape().as_list())

            logit = tf.reshape(tensor = deconv3, shape = [-1,self.emb_dim*self.sent_len])

            print("Logit shape:", logit.get_shape().as_list())
        return logit


   
    def _create_network(self):
        ### tf Graph input(in effect placeholders)
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name= "x")
        self.x_split = tf.transpose(tf.convert_to_tensor(tf.split(value=self.x, 
                                                                  num_or_size_splits = self.sent_len, 
                                                                  axis = 1)), [1,0,2])
        ### Embedding Matrix for Word2Vec Embedding
        self.embedding_matrix = tf.placeholder(tf.float32,
                                               shape = self.emb_matrix_shape, name = "embedding_matrix")
        self.normed_embedding_matrix = tf.nn.l2_normalize(self.embedding_matrix, 
                                                          axis = 1)
    
        with tf.variable_scope("vae"):
          ### select the right encoder as set in the argparser main file
          if self.enc_type == 'mlp':  
              ### 1. Get hidden latent distributions
              self.latent_dist = self._create_mlp_encoder(self.x)
          if self.enc_type == 'cnn':
              self.latent_dist = self._create_cnn_encoder(self.x)

          ### 2. Draw sample z from discrete and continuous distributions
          self.soft_complete_samples, self.hard_complete_samples = self.reparametrize(self.latent_dist)

          ### 3. Decode samples
          if self.dec_type == 'mlp':
              self.logit = self._create_mlp_decoder(self.soft_complete_samples)
             #self.hard_logit = self._create_mlp_decoder(self.hard_complete_samples, reuse=True)
          if self.dec_type == 'cnn':
              self.logit = self._create_cnn_decoder(self.soft_complete_samples)
             #self.hard_logit = self._create_cnn_decoder(self.hard_complete_samples, reuse=True)

          self.logit_split = tf.transpose(
                               tf.convert_to_tensor(tf.split(value = self.logit, 
                                                             num_or_size_splits = self.sent_len, 
                                                             axis = 1), name="logit_split" ), [1,0,2]) 
         # self.hard_logit_split = tf.transpose(
         #                      tf.convert_to_tensor(tf.split(value = self.hard_logit, 
         #                                                    num_or_size_splits = self.sent_len, 
         #                                                    axis = 1), name="hard_logit_split" ), [1,0,2])

    def _create_loss_optimizer(self):
        
        ### Create list with all running_vars count/total variables which we will extend in the following 
        self.running_vars = []
        ### Handy tf.metrics variables collected in list and runable when needed
        self.tf_metrics_tensor_dict = {}
        ### Existing summary_op collector
        summary_op_list = []
        
        ### Single disc latent loss running metrics
        self.single_disc_latent_losses = []
                
        ### KL-Loss-Term-Placeholders to prevent if-cases
        self.cont_latent_loss = 0.0
        self.disc_latent_loss = 0.0

        ### Reconstruction loss
        with tf.name_scope("reconstr_loss"):
            if self.repr_type == 'one-hot':
                reconstr_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits= self.logit_split ,
                        labels= self.x_split ) #collapses axis=2
                # this is the error-value PER SENTENCE #collapses axis=1                
                reconstr_loss = tf.reduce_sum(reconstr_loss,axis=1)    
            if "word2vec" in self.repr_type:
                #this is SE (SQUARRED ERRORS) on axis =2
                reconstr_loss = tf.square(tf.subtract( self.logit_split, 
                                                       self.x_split))
                ### Now lets get rid of the vector dimension 300
                ### This is SSE per word (SUM OF SQUARRED ERRORS) #collapses axis=2
                reconstr_loss = tf.reduce_sum( reconstr_loss, axis = 2) 
                ### Now lets get rid of the sentence length 7 by adding up all SSE values 
                ### -> To get Sum of SSE (SSSE)
                ### This is the error-value per sentence (sum over all 7 SSE values in this sentence) #collapses axis=1
                reconstr_loss = tf.reduce_sum( reconstr_loss, axis=1)

            reconstr_loss = tf.reduce_mean(reconstr_loss, 
                                           name = "reconstr_loss") #this is equivalent to tf.reduce_sum(reconstr_loss) / batch_size -> WHICH IS THE AVERAGE error-value PER SENTENCE
            self.reconstr_loss, self.reconstr_loss_update = tf.metrics.mean(
                                                reconstr_loss,name="metric")
            self.running_vars.extend(tf.get_collection(
                                                tf.GraphKeys.LOCAL_VARIABLES, 
                                                scope = "reconstr_loss/metric"))
            reconstr_loss_summary_op      = tf.summary.scalar('reconstr_loss',        
                                                              self.reconstr_loss)
            summary_op_list.append(reconstr_loss_summary_op)
            self.tf_metrics_tensor_dict.update(reconstr_loss = self.reconstr_loss)

        if self.is_discrete:
            ### Create list for tracking single disc latent loss updates
            self.list_single_disc_latent_loss_update = []
            
            ### Calculates the KL divergence between a categorical distribution and a uniform categorical distribution.    
            with tf.name_scope("disc_latent_loss"):
                self.disc_dims = tf.convert_to_tensor(value=self.latent_spec["disc"],dtype=tf.float32)
                self.log_dims  = tf.log(self.disc_dims)
                ### Calculate negative entropy of each row
                self.neg_entropies = [tf.reduce_sum(alpha*tf.log(alpha+EPS), axis=1) for alpha in self.alphas]
                ### Take mean of negative entropy across batch
                self.mean_neg_entropies = [tf.reduce_mean(neg_entropy,axis=0) for neg_entropy in self.neg_entropies]
                ### Disc KL losses for each alpha with uniform categorical variable
                self.disc_latent_losses = self.log_dims + self.mean_neg_entropies
                ### Add dimension such that we can split object into list of tensors
                expanded_losses = tf.expand_dims(self.disc_latent_losses,axis=1)
                ### Now split into list of tensors
                split_losses = tf.split(value=expanded_losses, num_or_size_splits=self.num_disc_latents,axis=0)
                ### Logging of single disc latent losses
                for index,single_loss in enumerate(split_losses):  #####HOW TO LOOP OVER THIS OBJECT, WHAT TYPE OF OBJECT IS THIS ????
                    loss_name = str(index+1) + '_nr_dims_'+ str(self.latent_spec['disc'][index]) 
                    single_disc_latent_loss, single_disc_latent_loss_update = tf.metrics.mean(single_loss, name=loss_name)
                    ### Append each single disc latent loss update value into list of runnable tensor_updates
                    self.list_single_disc_latent_loss_update.append(single_disc_latent_loss_update)          
                    ### Apend each single disc latent loss value into list of runnable tensors
                    self.single_disc_latent_losses.append(single_disc_latent_loss)
                    self.running_vars.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "disc_latent_loss/"+loss_name))
                    single_disc_latent_loss_summary_op = tf.summary.scalar('single_disc_latent_loss_' + loss_name , single_disc_latent_loss)
                    summary_op_list.append(single_disc_latent_loss_summary_op)
                
                self.disc_total_kld = tf.reduce_sum(self.disc_latent_losses)
                self.disc_latent_loss, self.disc_latent_loss_update = tf.metrics.mean(self.disc_total_kld, name="metric")
                self.running_vars.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "disc_latent_loss/metric"))
                disc_latent_loss_summary_op   = tf.summary.scalar('disc_latent_loss', 
                                                                  self.disc_latent_loss)
                summary_op_list.append(disc_latent_loss_summary_op)
                self.tf_metrics_tensor_dict.update(disc_latent_loss = self.disc_latent_loss)
        
        
        if self.is_continuous:
            ### Normal Latent Loss/ Total_kld (alias total KL divergence)
            with tf.name_scope("cont_latent_loss"):
                klds = -0.5 * ( 1 + self.z_logvar - tf.square(self.z_mean) - tf.exp(self.z_logvar) ) # this is a 2D-array of size (Batch x z_dim)
                self.cont_dimensionwise_kld = tf.reduce_mean(klds, axis=0, name = "dimensionwise_kld") # this is a 1D-array of size (1 x z_dim)
                self.cont_mean_kld = tf.reduce_mean(tf.reduce_mean(klds, axis=1), name = "mean_kld") # this is a 1D-scalar of size (1 x 1) and the relationship to total_kld is total_kld = z_dim*total_kld
                cont_total_kld = tf.reduce_mean(tf.reduce_sum(klds, axis=1), name = "cont_total_kld") # this is a scalar of size (1 x 1) and MUST BE USED FOR TOTAL LOSS OF VAE (i.e. multiplied by beta-value)
                ### total_kld == latent_loss
                self.cont_latent_loss, self.cont_latent_loss_update = tf.metrics.mean(cont_total_kld, name = "metric")
                self.running_vars.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "cont_latent_loss/metric"))
                cont_latent_loss_summary_op   = tf.summary.scalar('disc_latent_loss',
                                                                  self.cont_latent_loss)
                summary_op_list.append(cont_latent_loss_summary_op)
                self.tf_metrics_tensor_dict.update(cont_latent_loss = self.cont_latent_loss)

                
        ### Total Loss as final joint-VAE loss
        with tf.name_scope("total_loss"):
            pre_total_loss = (reconstr_loss + (self.disc_gamma * self.disc_total_kld) + self.cont_latent_loss)
            total_loss = tf.identity(pre_total_loss,name = "total_loss")
            self.total_loss, self.total_loss_update = tf.metrics.mean( total_loss, name = "metric")
            self.running_vars.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "total_loss/metric"))  
            total_loss_summary_op    = tf.summary.scalar('total_loss',
                                                         self.total_loss)
            summary_op_list.append(total_loss_summary_op)
            self.tf_metrics_tensor_dict.update(total_loss = self.total_loss)


        ### Accuracy word level e.g. "i eat the payment pad pad pad" vs "we accept the meal pad pad pad" --> equals accuracy of 4/7
        with tf.name_scope("accuracy"):
    
            if self.repr_type == "one-hot":
                #prob_split = tf.nn.softmax(self.hard_logit_split,axis=2)
                prob_split = tf.nn.softmax(self.logit_split,axis=2)
                ### Now get the predictions as indices
                predictions_idx_split = tf.argmax(prob_split,axis=2)
                labels_idx_split = tf.argmax(self.x_split,axis=2)
                ### Now squeeze to remove last dimension of size 1
                self.predictions = tf.squeeze(predictions_idx_split, name = "prediction")
                self.labels = tf.squeeze(labels_idx_split, name = "label")
   

            if "word2vec" in self.repr_type:
                #logit_split_reshaped = utils.reshape(self.hard_logit_split, [[0, 1], 2])
                logit_split_reshaped = utils.reshape(self.logit_split, [[0, 1], 2])
                x_split_reshaped = utils.reshape(self.x_split, [[0, 1], 2])
                normed_logit_split_reshaped = tf.nn.l2_normalize(logit_split_reshaped, axis = 1) 
                normed_x_split_reshaped = tf.nn.l2_normalize(x_split_reshaped, axis = 1)
                
                cosine_similarity_logit = tf.matmul(normed_logit_split_reshaped, tf.transpose(self.normed_embedding_matrix, [1,0]))
                cosine_similarity_x = tf.matmul(normed_x_split_reshaped, tf.transpose(self.normed_embedding_matrix, [1,0]))
                ### Finally compute the argmax for each element of the batch, this will denote the idx of the correct and predicted word indices
                predictions = tf.argmax(cosine_similarity_logit, axis = 1)
                self.predictions = tf.reshape(predictions, shape=(-1, self.sent_len), name = "prediction")
                labels = tf.argmax(cosine_similarity_x, axis = 1)
                self.labels = tf.reshape(labels, shape=(-1, self.sent_len), name = "label")


            self.accuracy, self.accuracy_update = tf.metrics.accuracy( labels = self.labels, predictions = self.predictions, name = "metric")
            self.running_vars.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "accuracy/metric"))
            accuracy_summary_op      = tf.summary.scalar('word_level_accuracy',
                                                         self.accuracy)
            summary_op_list.append(accuracy_summary_op)
            self.tf_metrics_tensor_dict.update(accuracy = self.accuracy)

            ### Define initializer in order to initialize/reset all running_vars
            self.running_vars_initializer = tf.variables_initializer(var_list = self.running_vars)

        with tf.name_scope("optimizer"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
                gradients, variables = zip(*optimizer.compute_gradients(total_loss))

                ### add variables to vae model
                self.gradients_global_norm = tf.global_norm(gradients)
                grads_and_vars = zip(gradients, variables)
                grads_and_vars_not_none = []
                for grad,var in grads_and_vars:
                    if grad == None:
                        pass
                    else:
                        grads_and_vars_not_none.append((grad,var))
                self.gradients, self.variables = zip(*grads_and_vars_not_none)
                self.optimizer = optimizer.apply_gradients(zip(gradients, variables))


        ### Merge all summary ops to a single summary_op
        self.summary_op = tf.summary.merge(summary_op_list)
        
        
    def partial_fit(self, sess, x_data, embedding_matrix):
        if self.is_continuous:
            return_values                       =  sess.run(self.list_single_disc_latent_loss_update + 
                                                            [self.optimizer, 
                                                            self.reconstr_loss_update, 
                                                            self.disc_latent_loss_update, 
                                                            self.cont_latent_loss_update, 
                                                            self.total_loss_update, 
                                                            self.accuracy_update, 
                                                            self.summary_op],
                                                            feed_dict={self.x : x_data, self.phase : True, self.embedding_matrix : embedding_matrix})
        else:
            return_values                       =  sess.run(self.list_single_disc_latent_loss_update + 
                                                           [self.optimizer,
                                                            self.reconstr_loss_update, 
                                                            self.disc_latent_loss_update, 
                                                            self.total_loss_update, 
                                                            self.accuracy_update, 
                                                            self.summary_op],
                                                            feed_dict={self.x : x_data, self.phase : True, self.embedding_matrix : embedding_matrix})

        summary_str = return_values[-1]
        return summary_str


    def partial_test(self, sess, x_data, embedding_matrix):
        if self.is_continuous:
            return_values                   =       sess.run(self.list_single_disc_latent_loss_update + 
                                                            [self.reconstr_loss_update,
                                                            self.disc_latent_loss_update, 
                                                            self.cont_latent_loss_update,
                                                            self.total_loss_update, 
                                                            self.accuracy_update, 
                                                            self.summary_op],
                                                            feed_dict={self.x : x_data, self.phase : False, self.embedding_matrix : embedding_matrix})
        else:
            return_values                   =      sess.run(self.list_single_disc_latent_loss_update + 
                                                           [self.reconstr_loss_update, 
                                                            self.disc_latent_loss_update, 
                                                            self.total_loss_update, 
                                                            self.accuracy_update, 
                                                            self.summary_op],
                                                            feed_dict={self.x : x_data, self.phase : False, self.embedding_matrix : embedding_matrix})

        summary_str = return_values[-1]
        return summary_str
    

    ### Runs any dict of tensor variables from tf.metrics. API (WHICH DO NOT NEED A FEED DICT) given a session(sess)
    def run_tf_metric_tensors(self, sess, tf_metric_tensor_dict):
        outputs = {}
        for key,tf_metric_tensor in tf_metric_tensor_dict.items():
            output = sess.run(tf_metric_tensor)
            outputs[key] = output
        return outputs


    def predict(self, sess, x_data, embedding_matrix):
        ### gives the predictions given data. These predictions are simply indices which denote the token using the id-to-token-dict.
        ### x -> argmax predictions
        return sess.run(self.predictions,
                        feed_dict={self.x : x_data, self.phase : False, self.embedding_matrix : embedding_matrix})

    def transform(self, sess, x_data, embedding_matrix):
        ### Transform data by mapping it into the latent space.
        ### x -> soft_complete_samples, hard_complete_samples
        return sess.run([self.soft_complete_samples, self.hard_complete_samples],
                         feed_dict={self.x: x_data, self.phase : False, self.embedding_matrix : embedding_matrix })

    def return_batch(self, X, batch_indices): 
        batch = X[batch_indices]
        return batch


    def train(self, sess, manager, saver, args):

        ### Plot Steps Buffer List
        plot_steps = []
        value_buffers = [ [] for i in range(self.num_disc_latents)]
        gf_accuracies_buffer_test = []
        MIGs_test = []

        gf_accuracies_buffer_train = []
        MIGs_train = []

        ### Define path towards experiment log directory
        exp_logdir = args.logs_dir + self.experiment_name
        ### Make sure we do not mix existing logs_dir with new logs by deleting existing ones
        if os.path.exists(exp_logdir):
            shutil.rmtree(exp_logdir)

        ### Now as usual create filewriters
        train_writer = tf.summary.FileWriter(exp_logdir +  "/train", sess.graph)
        test_writer = tf.summary.FileWriter(exp_logdir + "/test")
     
        
        ### Fetch variables for train and test cases from data manager    
        n_train_samples  = manager.train_size
        n_test_samples   = manager.test_size
  
        print("n_train_samples:", n_train_samples)
        print("n_test_samples:", n_test_samples)
 
        ### Calculate amount of batches for training and testing
        n_train_batches  = n_train_samples // self.batch_size 
        n_test_batches   = n_test_samples // self.batch_size
        ### Case: 
        if n_test_samples > 10000:
            ### Reduce the number of used test batches to only 10%
            n_test_batches = int(0.1*n_test_batches)
        else:
            pass

        print("number used test samples:",self.batch_size * n_test_batches)
        
        ### Fetch train and test data
        train_indices    = manager.train_indices  ###these indices are based on the whole dataset(X_test and X_train) 
        test_indices     = manager.test_indices   ###these indices are based on the whole dataset(X_test and X_train)    
        X_train          = manager.X_train
        X_test           = manager.X_test      
        ### 1.so that we dont test over same array every time
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        ### Fetch 10000 samples for metric training        
        metric_indices_train = np.random.choice(train_indices, 5000, replace=False)
        metric_X_train = manager.dataset[metric_indices_train]

        ### Fetch 10000 samples for metric testing        
        metric_indices_test = np.random.choice(test_indices, 5000, replace=False)
        metric_X_test = manager.dataset[metric_indices_test]
        ### Also do fetch the corresponding ground truth generative factors gf-array
        GFs_test = manager.latents_classes[metric_indices_test]
        GFs_train = manager.latents_classes[metric_indices_train]
        ### Calculate the inv_gf_entropies for normalization purposes
        inv_GF_entropies = utils.calc_inv_entropies(latents_sizes = manager.latents_sizes, n_used_tuples = self.n_used_tuples)

        ### Determine randomly and only once, those indices we will use for evaluation purposes, to make the measure less noisy use at least 5000 values
        recon_indices_test = np.random.choice(test_indices, 5000, replace=False)
        recon_indices_test = list(recon_indices_test)

        recon_indices_train = np.random.choice(train_indices, 5000, replace=False)
        recon_indices_train = list(recon_indices_train)        


        ### Fetch ground-truth-sentences for evaluation (this is only needed once. therefore we do it now)
        gt_sentences_test = copy.deepcopy(manager.ground_truth(recon_indices_test))
        gt_sentences_train = copy.deepcopy(manager.ground_truth(recon_indices_train))        

        ### remove "<PAD>" and "." from gt_sentences
        for entry in gt_sentences_test:
            while entry[-1] in ["<PAD>","."]:
                del entry[-1]

        
        ### remove "<PAD>" and "." from gt_sentences
        for entry in gt_sentences_train:
            while entry[-1] in ["<PAD>","."]:
                del entry[-1]


        ### Load embedding matrix
        embedding_matrix = manager.embedding_matrix
          
        ### step is the global step counter (number of so far processed X_train batches),
        ### we can see it in tensorboard on the x-axis 
        step = 0
        epoch = 0
        ### Dump initial disc losses and epoch into buffers
        plot_steps.append(epoch)

        ### Calcualte Initial GF accuracies for Later Normalizing the gf_accuracies array
        accuracies_test = utils.gf_accuracy(self,sess,manager,recon_indices_test,gt_sentences_test)
        gf_accuracies_buffer_test.append(accuracies_test)
 
        ### Calcualte Initial GF accuracies for Later Normalizing the gf_accuracies array
        accuracies_train = utils.gf_accuracy(self,sess,manager,recon_indices_train,gt_sentences_train)
        gf_accuracies_buffer_train.append(accuracies_train)

#####################################################################################################################################################################

        ### Normalize and note that this function utils.normalize_by_first_row expects an np.array as input and also outputs same type again!
        normalized_current_gf_accuracies_test = utils.normalize_by_first_row(np.array(gf_accuracies_buffer_test)) 
        current_gf_accuracies_test = normalized_current_gf_accuracies_test[-1] ### normalize by value of first row and current value is simply the last row/element
        dict_accuracies_test = dict(zip(manager.list_of_gf_strings,current_gf_accuracies_test))
        print("GF accuracies TEST set", dict_accuracies_test)

        ### Determine the current average GF accuracy
        avg_GF_accuracy_test = np.mean(current_gf_accuracies_test)
        print("GF average TEST accuracy",avg_GF_accuracy_test)
        avg_GF_accuracy_str = tf.Summary(value=[tf.Summary.Value(tag="Average GF-accuracy", simple_value=avg_GF_accuracy_test),])
        test_writer.add_summary(avg_GF_accuracy_str, epoch)

        ### Normalize and note that this function utils.normalize_by_first_row expects an np.array as input and also outputs same type again!
        normalized_current_gf_accuracies_train = utils.normalize_by_first_row(np.array(gf_accuracies_buffer_train))
        current_gf_accuracies_train = normalized_current_gf_accuracies_train[-1] ### normalize by value of first row and current value is simply the last row/element
        dict_accuracies_train = dict(zip(manager.list_of_gf_strings,current_gf_accuracies_train))
        print("GF accuracies TRAIN set", dict_accuracies_train)

        ### Determine the current average GF accuracy
        avg_GF_accuracy_train = np.mean(current_gf_accuracies_train)
        print("GF average TRAIN accuracy", avg_GF_accuracy_train)
        avg_GF_accuracy_str = tf.Summary(value=[tf.Summary.Value(tag="Average GF-accuracy", simple_value=avg_GF_accuracy_train),])
        train_writer.add_summary(avg_GF_accuracy_str, epoch)
   
##########################################################################################################################################################################

        ### Define tensorboard summaries for the GF-accuracies
        ### Create customized summary for GF accuracies
        GF_tags = ["obj-verb-tuple","obj-sing-pl","gender","subj-sing-pl","sent-type","1st-2nd-3rd-pers","pos-neg","verb-tense","verb-style"]
        GF_summary_str_list = []
        for idx,gf_tag in enumerate(GF_tags):
            GF_summary_str_list.append(tf.Summary(value=[tf.Summary.Value(tag="GF-accuracy-"+gf_tag, simple_value=current_gf_accuracies_test[idx]),]))

        ### Add current values in normalized_current_gf_accuracies
        for gf_summary_str in GF_summary_str_list:
            test_writer.add_summary(gf_summary_str, epoch)

        GF_summary_str_list = []
        for idx,gf_tag in enumerate(GF_tags):
            GF_summary_str_list.append(tf.Summary(value=[tf.Summary.Value(tag="GF-accuracy-"+gf_tag, simple_value=current_gf_accuracies_train[idx]),]))

        ### Add current values in normalized_current_gf_accuracies
        for gf_summary_str in GF_summary_str_list:
            train_writer.add_summary(gf_summary_str, epoch)
       
##########################################################################################################################################################################
 
        ### Evaluate initial metric
        ### the following two lines have to be executed in every epoch once!
        code_test = self.transform(sess, metric_X_test, embedding_matrix)[1] ### [1] because it is returning soft_samples (which are close to one-hot) and hard_samples (which are exactly one-hot) 
        argmax_code_test = utils.calc_argmax_code(code_test, self.split_indices)
        
        np.random.shuffle(argmax_code_test)
     
        MIGs_test.append(utils.MIG_score(GFs_test, argmax_code_test, inv_GF_entropies))

        ### Create customized summary for MIG-metric
        MIG_summary_str = tf.Summary(value=[tf.Summary.Value(tag="MIG", simple_value=MIGs_test[-1]),])  ###always add the most current value within the list (i.e. the last one by [-1])

        ### Add current current MIG value to summary
        test_writer.add_summary(MIG_summary_str, epoch)

        ### Evaluate initial metric
        ### the following two lines have to be executed in every epoch once!
        code_train = self.transform(sess, metric_X_train, embedding_matrix)[1] ### [1] because it is returning soft_samples (which are close to one-hot) and hard_samples (which are exactly one-hot) 
        argmax_code_train = utils.calc_argmax_code(code_train, self.split_indices)

        np.random.shuffle(argmax_code_train)

        MIGs_train.append(utils.MIG_score(GFs_train, argmax_code_train, inv_GF_entropies))

        ### Create customized summary for MIG-metric
        MIG_summary_str = tf.Summary(value=[tf.Summary.Value(tag="MIG", simple_value=MIGs_train[-1]),])  ###always add the most current value within the list (i.e. the last one by [-1])

        ### Add current current MIG value to summary
        train_writer.add_summary(MIG_summary_str, epoch)

##########################################################################################################################################################################

        for buffer_idx, buffer in enumerate(value_buffers):
            ### Fetch tensor metric of single disc latent loss of specific idx
            single_disc_latent_loss = self.single_disc_latent_losses[buffer_idx]
            ### Return the values
            value = sess.run(single_disc_latent_loss)
            ### Note down this value in the value_buffer list of lists
            value_buffers[buffer_idx].append(value)
        
        ### Do single loop for train-data  in order to have a value at step 0 (before we start training)
        for batch_idx in range(n_train_batches):
            batch_indices      = np.arange(self.batch_size*batch_idx, self.batch_size*(batch_idx+1))
            train_batch         = self.return_batch(X_train, batch_indices)
 
            ### shuffle along one axis, then transpose to shuffle along second axis, finally back transpose to original shape 
            np.random.shuffle(train_batch)
            train_batch = np.transpose(train_batch)
            np.random.shuffle(train_batch)
            train_batch = np.transpose(train_batch)
          

            train_summary_str   = self.partial_test(sess, train_batch, embedding_matrix)

        outputs = self.run_tf_metric_tensors(sess, self.tf_metrics_tensor_dict)
        train_string = ", ".join(["train "+str(key)+": "+str(round(val,3)) for (key,val) in outputs.items()])
        print("Intitial train metrics:\n")
        print(train_string)
   
        ### Resetting all running vars
        sess.run(self.running_vars_initializer)
#        print("Metrics after reset as sanity-check:", self.run_tf_metric_tensors(sess,self.tf_metrics_tensor_dict))
         
        ### Do single loop for test-data in order to have a values at step 0 (before we start training)
        print(n_test_batches)
        for batch_idx in range(n_test_batches):
            batch_indices      = np.arange(self.batch_size*batch_idx, self.batch_size*(batch_idx+1)) 
            test_batch         = self.return_batch(X_test, batch_indices)
            test_summary_str   = self.partial_test(sess, test_batch, embedding_matrix)

        outputs = self.run_tf_metric_tensors(sess, self.tf_metrics_tensor_dict)
        test_string = ", ".join(["test "+str(key)+": "+str(round(val,3)) for (key,val) in outputs.items()])
        print("Intitial test metrics:\n")
        print(test_string)

        ### Resetting all running vars
        sess.run(self.running_vars_initializer)
 
        ### Write initial accuracies of train and test data to file-writer
        train_writer.add_summary(train_summary_str, epoch)
        test_writer.add_summary(test_summary_str, epoch)

        ### Create lists of gf and code names for plotting the corr_mat later on
        gf_names = manager.list_of_gf_strings
        n_code_dims = argmax_code_train.shape[1]
        code_names = ["c"+str(i) for i in range(n_code_dims)]

        ### For comparision purposes save corr matrix at the beginning of training
        ### Create corr_matr between code and gf
        plt.figure(figsize=(10,5))
        plt.title("GF vs. Code Correlation Test Set")
        R = utils.MI_mat(GFs_test, argmax_code_test, inv_GF_entropies)
        ax = plt.gca()
        cax = ax.matshow(R,vmin=0,vmax=1)
        plt.colorbar(cax)
        plt.grid(True)
        xticks = np.arange(0,len(code_names),1)
        yticks = np.arange(0,len(gf_names),1)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(code_names)
        ax.set_yticklabels(gf_names)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        image_summary = tf.summary.image("Starting MI Correlation Test", image, max_outputs=1)
        image_summary_op = sess.run(image_summary)
        test_writer.add_summary(summary=image_summary_op, global_step=epoch)


        plt.figure(figsize=(10,5))
        plt.title("GF vs. Code Correlation Train Set")
        R = utils.MI_mat(GFs_train, argmax_code_train, inv_GF_entropies)
        ax = plt.gca()
        cax = ax.matshow(R,vmin=0,vmax=1)
        plt.colorbar(cax)
        plt.grid(True)
        xticks = np.arange(0,len(code_names),1)
        yticks = np.arange(0,len(gf_names),1)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(code_names)
        ax.set_yticklabels(gf_names)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        image_summary = tf.summary.image("Starting MI Correlation Train", image, max_outputs=1)
        image_summary_op = sess.run(image_summary)
        train_writer.add_summary(summary=image_summary_op, global_step=epoch)


        print("Training cycle starts now!")
        ### Training cycle
        for epoch in range(self.epoch_size):
            ### Shuffle train indices
            np.random.shuffle(train_indices)
        
            # Loop over all batches for training
            for batch_idx in range(n_train_batches):
                ### Generate train batch
                batch_indices     = np.arange(self.batch_size*batch_idx, self.batch_size*(batch_idx+1))          
                train_batch       = self.return_batch(X_train, batch_indices)
                train_summary_str = self.partial_fit(sess, train_batch, embedding_matrix)
                ### Increase step after having processed single batch 
                step += 1

            outputs = self.run_tf_metric_tensors(sess, self.tf_metrics_tensor_dict)
            train_string = ", ".join(["train "+str(key)+": "+str(round(val,3)) for (key,val) in outputs.items()])
            
            ### Reset running_vars
            sess.run(self.running_vars_initializer) 
#            print("Metrics after reset as sanity-check:", self.run_tf_metric_tensors(sess, self.tf_metrics_tensor_dict)) 
         
            ### Loop over all batches for testing
            for batch_idx in range(n_test_batches):
                batch_indices        = np.arange(self.batch_size*batch_idx, self.batch_size*(batch_idx+1))
                test_batch           = self.return_batch(X_test, batch_indices)
                test_summary_str     = self.partial_test(sess, test_batch, embedding_matrix)         
                    
            ### Fetch test metrics before we reset running_vars
            outputs = self.run_tf_metric_tensors(sess, self.tf_metrics_tensor_dict)
            test_string = ", ".join(["test "+key+": "+str(round(val,3)) for (key,val) in outputs.items()])

            ### Note down step and single disc loss values
            for buffer_idx, buffer in enumerate(value_buffers):
                single_disc_latent_loss = self.single_disc_latent_losses[buffer_idx]
                value = sess.run(single_disc_latent_loss)
                value_buffers[buffer_idx].append(value)

            ### Reset running_vars
            sess.run(self.running_vars_initializer)
#            print("Metrics after reset as sanity-check:", self.run_tf_metric_tensors(sess, self.tf_metrics_tensor_dict)) 

        
            # Printing out statistics of current epoch
            print("Epoch {} statistics:\n".format(epoch+1))

            print(train_string)
            print(test_string)
            print("Now reconstruction examples from the test set.")
            ### Keep track of sentences evolution of test dataset(can we generalize?), this is done after every epoch
            input_indices = np.random.choice(test_indices, 10, replace=False)
            input_indices = list(input_indices)
            utils.reconstruct_sentences(self, sess, input_indices, epoch, manager)
        
            ### Calcualte GF accuracies
            accuracies_test = utils.gf_accuracy(self,sess,manager,recon_indices_test,gt_sentences_test)
            accuracies_train = utils.gf_accuracy(self,sess,manager,recon_indices_train,gt_sentences_train)

            plot_steps.append(epoch+1)
            gf_accuracies_buffer_test.append(accuracies_test)
            gf_accuracies_buffer_train.append(accuracies_train)
            ### Normalize and note that this function utils.normalize_by_first_row expects an np.array as input and also outputs same type again!
            normalized_current_gf_accuracies_test = utils.normalize_by_first_row(np.array(gf_accuracies_buffer_test))
            current_gf_accuracies_test = normalized_current_gf_accuracies_test[-1] ### normalize by value of first row and current value is simply the last row/element

            dict_accuracies_test = dict(zip(manager.list_of_gf_strings,current_gf_accuracies_test))
            print("GF TEST accuracies", dict_accuracies_test)

            ### Determine the current average GF accuracy
            avg_GF_accuracy_test = np.mean(current_gf_accuracies_test)
            print("GF TEST average accuracy", avg_GF_accuracy_test)
            avg_GF_accuracy_str = tf.Summary(value=[tf.Summary.Value(tag="Average GF-accuracy", simple_value=avg_GF_accuracy_test),])
            test_writer.add_summary(avg_GF_accuracy_str, epoch)

            ### Normalize and note that this function utils.normalize_by_first_row expects an np.array as input and also outputs same type again!
            normalized_current_gf_accuracies_train = utils.normalize_by_first_row(np.array(gf_accuracies_buffer_train))
            current_gf_accuracies_train = normalized_current_gf_accuracies_train[-1] ### normalize by value of first row and current value is simply the last row/element

            dict_accuracies_train = dict(zip(manager.list_of_gf_strings,current_gf_accuracies_train))
            print("GF TRAIN accuracies", dict_accuracies_train)

            ### Determine the current average GF accuracy
            avg_GF_accuracy_train = np.mean(current_gf_accuracies_train)
            print("GF TRAIN average accuracy",avg_GF_accuracy_train)
            avg_GF_accuracy_str = tf.Summary(value=[tf.Summary.Value(tag="Average GF-accuracy", simple_value=avg_GF_accuracy_train),])
            train_writer.add_summary(avg_GF_accuracy_str, epoch)


            ### Define tensorboard summaries for the GF-accuracies
            ### Create customized summary for GF accuracies
            GF_tags = ["obj-verb-tuple","obj-sing-pl","gender","subj-sing-pl","sent-type","1st-2nd-3rd-pers","pos-neg","verb-tense","verb-style"]
            GF_summary_str_list = []
            for idx,gf_tag in enumerate(GF_tags):
                GF_summary_str_list.append(tf.Summary(value=[tf.Summary.Value(tag="GF-accuracy-"+gf_tag, simple_value=current_gf_accuracies_test[idx]),]))

            ### Add current values in normalized_current_gf_accuracies
            for gf_summary_str in GF_summary_str_list:
                test_writer.add_summary(gf_summary_str, epoch+1)

            GF_summary_str_list = []
            for idx,gf_tag in enumerate(GF_tags):
                GF_summary_str_list.append(tf.Summary(value=[tf.Summary.Value(tag="GF-accuracy-"+gf_tag, simple_value=current_gf_accuracies_train[idx]),]))

            ### Add current values in normalized_current_gf_accuracies
            for gf_summary_str in GF_summary_str_list:
                train_writer.add_summary(gf_summary_str, epoch+1)


            ### the following two lines have to be executed in every epoch once!
            code_test = self.transform(sess, metric_X_test, embedding_matrix)[1] ### [1] because it is returning soft_samples (which are close to one-hot) and hard_samples (which are exactly one-hot) 
            argmax_code_test = utils.calc_argmax_code(code_test, self.split_indices)

            np.random.shuffle(argmax_code_test)

            MIGs_test.append(utils.MIG_score(GFs_test, argmax_code_test, inv_GF_entropies))

            ### Create customized summary for MIG-metric
            MIG_summary_str = tf.Summary(value=[tf.Summary.Value(tag="MIG", simple_value=MIGs_test[-1]),])  ###always add the most current value within the list (i.e. the last one by [-1])

            ### Add current current MIG value to summary
            test_writer.add_summary(MIG_summary_str, epoch+1)

            ### Evaluate initial metric
            ### the following two lines have to be executed in every epoch once!
            code_train = self.transform(sess, metric_X_train, embedding_matrix)[1] ### [1] because it is returning soft_samples (which are close to one-hot) and hard_samples (which are exactly one-hot) 
            argmax_code_train = utils.calc_argmax_code(code_train, self.split_indices)

            np.random.shuffle(argmax_code_train)

            MIGs_train.append(utils.MIG_score(GFs_train, argmax_code_train, inv_GF_entropies))

            ### Create customized summary for MIG-metric
            MIG_summary_str = tf.Summary(value=[tf.Summary.Value(tag="MIG", simple_value=MIGs_train[-1]),])  ###always add the most current value within the list (i.e. the last one by [-1])


            ### Add current current MIG value to summary
            train_writer.add_summary(MIG_summary_str, epoch+1)

            ### Last but not least after every epoch we can write to file-writers:
            train_writer.add_summary(train_summary_str, epoch+1)
            test_writer.add_summary(test_summary_str, epoch+1)


            ### Finally before starting a new epoch save checkpoint! This is done after every epoch
            ### setting "write_meta_graph" to False as we only want to save the .meta file once at the very first saving step, 
            ### we do not need to recreate the .meta file in every epoch we save the session, as the graph does not change after creation 
            saver.save(sess, args.checkpoints_dir + self.experiment_name+ "/" + "checkpoint", global_step = epoch, write_meta_graph = False)
        

            ### Finally we decrease the temperature a little bit so that we can can smoothly deform into  a categorical distribution
            ### this is done every 5 epohcs
            if (epoch % 5 == 0) and (epoch != 0):
                self.temp = self.temp * 0.9
                print("NEW TEMPERATURE VALUE: {}.".format(self.temp))
            else:
                pass

 

        ### At the very end of the training lets generate the single latent loss plot
        plt.figure(figsize=(10,5)) ### figize=(width,height)
        for index,values in enumerate(value_buffers):
            loss_name = str(index+1) + '_nr_dims_'+ str(self.latent_spec['disc'][index])  
            curve_label = "single_disc_latent_loss_" + loss_name
            plt.plot(plot_steps,values,label=curve_label)
        dim_single_code = self.code_dim_list[0]
        plt.plot(plot_steps,len(plot_steps)*[np.log(dim_single_code)] ,':', label="max. single dim loss log("+str(dim_single_code)+")", color="k")
        plt.grid(True)
        plt.title("Single Discrete Latent Loss")
        plt.xlabel("Epoch")
        plt.ylabel("KL-loss")
        plt.legend(loc='lower right')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        image_summary = tf.summary.image("single_disc_latent_losses", image, max_outputs=1)
        image_summary_op = sess.run(image_summary)
        train_writer.add_summary(image_summary_op)

##################################### TEST ########################################


        ### At the very end of the training lets generate plot of gf_accuracy evolutions
        gf_accuracies_array_test = np.array(gf_accuracies_buffer_test)
        ### Normalize
        normalized_gf_accuracies_array_test = utils.normalize_by_first_row(gf_accuracies_array_test)
        plt.figure(figsize=(10,5)) 
        for curve_idx,curve_label in enumerate(manager.list_of_gf_strings):
            ### Fetch the right column for GF factor with index "curve_idx"
            gf_values = normalized_gf_accuracies_array_test[:,curve_idx]
            plt.plot(plot_steps,gf_values,label=curve_label) 
       
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Test Accuracy")
        plt.title("Generative Factors Test-Accuracies")
        plt.legend(loc="upper left")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        image_summary = tf.summary.image("Generative Factors Accuracies Test", image, max_outputs=1)
        image_summary_op = sess.run(image_summary)
        test_writer.add_summary(summary=image_summary_op,global_step=epoch+1)

        ### At the very end of the training lets generate plot of MIG metric evolution
        MIGs_test = np.array(MIGs_test)
        plt.figure(figsize=(10,5))
        plt.plot(MIGs_test)
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Test MIG")
        plt.title("Evolution of Mutual Information Gap Test")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        image_summary = tf.summary.image("MIG scores Test", image, max_outputs=1)
        image_summary_op = sess.run(image_summary)
        test_writer.add_summary(summary=image_summary_op,global_step=epoch+1)

        ### Create MI_matr between code and gf
        plt.figure(figsize=(10,5))
        plt.title("GF vs. Code MI Correlation Test Set")
        R = utils.MI_mat(GFs_test, argmax_code_test, inv_GF_entropies)
        ax = plt.gca()
        cax = ax.matshow(R,vmin=0,vmax=1)
        plt.colorbar(cax)
        plt.grid(True)
        xticks = np.arange(0,len(code_names),1)
        yticks = np.arange(0,len(gf_names),1)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(code_names)
        ax.set_yticklabels(gf_names)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        image_summary = tf.summary.image("Final MI Correlation Test", image, max_outputs=1)
        image_summary_op = sess.run(image_summary)
        test_writer.add_summary(summary=image_summary_op,global_step=epoch+1)

############################################### TRAIN ################################################
        ### At the very end of the training lets generate plot of gf_accuracy evolutions
        gf_accuracies_array_train = np.array(gf_accuracies_buffer_train)
        ### Normalize
        normalized_gf_accuracies_array_train = utils.normalize_by_first_row(gf_accuracies_array_train)
        plt.figure(figsize=(10,5))
        for curve_idx,curve_label in enumerate(manager.list_of_gf_strings):
            ### Fetch the right column for GF factor with index "curve_idx"
            gf_values = normalized_gf_accuracies_array_train[:,curve_idx]
            plt.plot(plot_steps,gf_values,label=curve_label)

        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Train Accuracy")
        plt.title("Generative Factors Train-Accuracies")
        plt.legend(loc="upper left")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        image_summary = tf.summary.image("Generative Factors Accuracies Train", image, max_outputs=1)
        image_summary_op = sess.run(image_summary)
        train_writer.add_summary(summary=image_summary_op,global_step=epoch+1)

        ### At the very end of the training lets generate plot of MIG metric evolution
        MIGs_train = np.array(MIGs_train)
        plt.figure(figsize=(10,5))
        plt.plot(MIGs_train)
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Train MIG")
        plt.title("Evolution of Mutual Information Gap Train Set")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        image_summary = tf.summary.image("MIG scores Train", image, max_outputs=1)
        image_summary_op = sess.run(image_summary)
        train_writer.add_summary(summary=image_summary_op,global_step=epoch+1)

        ### Create MI_matr between code and gf
        plt.figure(figsize=(10,5))
        plt.title("GF vs. Code Correlation Train Set")
        R = utils.MI_mat(GFs_train, argmax_code_train, inv_GF_entropies)
        ax = plt.gca()
        cax = ax.matshow(R,vmin=0,vmax=1)
        plt.colorbar(cax)
        plt.grid(True)
        xticks = np.arange(0,len(code_names),1)
        yticks = np.arange(0,len(gf_names),1)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(code_names)
        ax.set_yticklabels(gf_names)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        image_summary = tf.summary.image("Final MI Correlation Train", image, max_outputs=1)
        image_summary_op = sess.run(image_summary)
        train_writer.add_summary(summary=image_summary_op,global_step=epoch+1)

