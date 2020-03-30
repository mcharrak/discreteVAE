import tensorflow as tf
import numpy as np
import utils
layers = tf.contrib.layers
import shutil
import os
import copy
import io
import matplotlib
### Set matplotlib not to use Xwindows backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class VAE(object):
  """ Beta Variational Auto Encoder. """
  
  def __init__(self, args, manager):

    # define data parameters, extract these from manager-input 
    self.input_dim = manager.input_dim
    self.sent_len = manager.sent_len      
    self.emb_dim = int(self.input_dim / self.sent_len)   # Define network hyper parameters
    self.emb_matrix_shape = manager.embedding_matrix.shape
    self.train_split = args.train_split
    
    # define network hyper parameters, extract these from args-input
    self.beta = args.beta
    self.lr = args.lr
    self.z_dim = args.z_dim


    if args.activation_fn == "elu":
        self.activation_fn = tf.nn.elu
    if args.activation_fn == "relu":
        self.activation_fn = tf.nn.relu

    self.filters_per_layer_list = args.filters_per_layer_list
    self.kernel_width_per_layer_list = args.kernel_width_per_layer_list
    self.enc_units_per_layer = args.enc_units_per_layer_list
    self.dec_units_per_layer_list = args.dec_units_per_layer_list

    
    self.epoch_size = args.epoch_size    
    self.batch_size = args.batch_size
    self.repr_type = args.repr_type
    self.enc_type = args.enc_type
    self.dec_type = args.dec_type
    self.n_used_tuples = args.n_used_tuples
    ### Pruning threshold for disentanglement score
    self.prun_thold = args.prun_thold

    ### Give an experiment name
    self.experiment_name = (self.repr_type + '_beta_' + str(self.beta) +
                            '_lr_' + str(self.lr) + '_n_tuples_' + str(args.n_used_tuples) + "_enc_type_" + args.enc_type + "_dec_type_" + args.dec_type +
                            '_split_' + str(self.train_split) +  '_n_epoch_' + str(self.epoch_size))

    ### Create autoencoder network
    self._create_network()
    
    ### Define loss function and corresponding optimizer
    self._create_loss_optimizer()


  def _sample_z(self, z_mean, z_log_sigma_sq):
    with tf.variable_scope("reparametrize"):
      eps_shape = tf.shape(z_mean)
      eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
      # z = mu + sigma * epsilon
      z = tf.add(z_mean,
               tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps), name = "z")
      return z


  def _create_mlp_encoder(self, x, phase, reuse=False):
      with tf.variable_scope("enc", reuse=reuse):
          layer = x
          layer_nr = 1
          for units in self.enc_units_per_layer:
               layer = layers.fully_connected(inputs = layer, num_outputs = units,
                                              activation_fn = self.activation_fn,
                                              scope = 'fc'+str(layer_nr))
               layer = layers.batch_norm(inputs = layer,
                                         is_training = phase,
                                         activation_fn = None,
                                         scope = 'fc'+str(layer_nr)+"BN")
               layer_nr = (layer_nr + 1)
          
          z_mean = layers.fully_connected(inputs=layer,
                                           num_outputs=self.z_dim,
                                           activation_fn = None,
                                           scope='mean')
          z_log_sigma_sq = layers.fully_connected(inputs=layer,
                                                   num_outputs=self.z_dim,
                                                   activation_fn = None,
                                                   scope='log_sigma_sq')
          
          return (z_mean, z_log_sigma_sq)


  """ 
  def _create_mlp_encoder(self, x, phase, reuse=False):
    with tf.variable_scope("enc", reuse=reuse):
      layer1 = layers.fully_connected(inputs=x,
                                      num_outputs=1024,
                                      activation_fn = None,
                                      scope='fc1')
      layer1_normalized = layers.batch_norm(inputs=layer1,
                                            is_training=phase,
                                            activation_fn = tf.nn.elu,
                                            scope="fc1BN")
      layer2 = layers.fully_connected(inputs=layer1_normalized,
                                      num_outputs=512,
                                      activation_fn = None,
                                      scope='fc2')
      layer2_normalized = layers.batch_norm(inputs=layer2,
                                            is_training=phase,
                                            activation_fn = tf.nn.elu,
                                            scope="fc2BN")
      layer3 = layers.fully_connected(inputs=layer2_normalized,
                                      num_outputs=256,
                                      activation_fn = None,
                                      scope='fc3')
      layer3_normalized = layers.batch_norm(inputs=layer3,
                                            is_training=phase,
                                            activation_fn = tf.nn.elu,
                                            scope="fc3BN")
      z_mean = layers.fully_connected(inputs=layer3_normalized,
                                    num_outputs=self.z_dim,
                                    activation_fn = None,
                                    scope='mean')
      z_log_sigma_sq = layers.fully_connected(inputs=layer3_normalized,
                                           num_outputs=self.z_dim,
                                           activation_fn = None,
                                           scope='log_sigma_sq')
      return (z_mean, z_log_sigma_sq)
  """



  def _create_cnn_encoder(self, x, phase, reuse=False):
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
      self.conv3_shape = conv3.get_shape().as_list()      
      print("This is what conv3 looks like after flatten:", conv3)      
      print("Shape of conv3:", self.conv3_shape)

        
      z_mean = layers.fully_connected(inputs            = conv3,
                                      num_outputs       = self.z_dim,
                                      activation_fn     = None,
                                      scope             = 'mean')

      z_log_sigma_sq = layers.fully_connected(inputs    = conv3,
                                           num_outputs  = self.z_dim,
                                           activation_fn= None,
                                           scope        = 'log_sigma_sq')

      return (z_mean, z_log_sigma_sq)


  def _create_mlp_decoder(self, z, reuse=False):
    with tf.variable_scope("dec", reuse = reuse):
      ### set layer to input and create layer counter
      layer = z 
      layer_nr = 1
      ### self.dec_units_per_layer_list might look like: [128,512,512] --> three layers
      for units in self.dec_units_per_layer_list :
          layer = layers.fully_connected(inputs = layer, num_outputs = units, activation_fn = self.activation_fn, scope = 'fc'+str(layer_nr))
          layer_nr = layer_nr + 1

      logit = layers.fully_connected(inputs = layer, num_outputs = self.input_dim, activation_fn = None, scope='logit')
      return logit


  def _create_cnn_decoder(self, z, reuse=False):
    with tf.variable_scope("dec", reuse=reuse):
      
      dim_features = self.filters_per_layer_list[0] ### this is equivalent to 512
      
      ### Blow up dimension from 20 (hidden_dim) to 128 (dimension of last encoder convolutional layer output)
      deconv0 = layers.fully_connected(inputs = z, num_outputs = dim_features, activation_fn = None, scope="deconv_prep")
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
    ### splitting sentence of into 7 subtensors (word-wise) and bringing batch dimension back to the first position of the shape [batch, sent_len , emb_dim]
    self.x_split = tf.transpose(tf.convert_to_tensor(tf.split(value=self.x, num_or_size_splits = self.sent_len, axis = 1)), [1,0,2])
    self.embedding_matrix = tf.placeholder(tf.float32, shape = self.emb_matrix_shape, name = "embedding_matrix")
    self.normed_embedding_matrix = tf.nn.l2_normalize(self.embedding_matrix, axis = 1) 
     
    with tf.variable_scope("vae"):
      ### select the right encoder as set with the argparser object in main file
      if self.enc_type == 'mlp':
          z_mean, z_log_sigma_sq = self._create_mlp_encoder(self.x, self.phase) # (Batch Dimension, Hidden Dimension)
      if self.enc_type == 'cnn':
          z_mean, z_log_sigma_sq = self._create_cnn_encoder(self.x, self.phase) # (Batch Dimension, Hidden Dimension)          

      ### now lets give the z_mean and z_log_sigma_sq some names for later inference session running in evaluation mode
      self.z_mean = tf.identity(z_mean, name = "z_mean")
      self.z_log_sigma_sq = tf.identity(z_log_sigma_sq, name = "z_log_sigma_sq")
      self.z_mean_dimensionwise = tf.reduce_mean(self.z_mean, axis=0, name = "z_mean_dimensionwise") # (Hidden Dimension,)
      ### prepare variance (sigma_sq) and standard deviation(sigma) for the class function .code_statistics()
      z_sigma_sq = tf.exp(self.z_log_sigma_sq, name = "z_sigma_sq") # (Batch, Hidden Dimension)
      z_sigma = tf.sqrt(z_sigma_sq, name = "z_sigma") # (Batch Dimension,Hidden Dimension)
 
      ### now assign to class-variable (with self.)
      self.z_sigma_sq_dimensionwise = tf.reduce_mean(z_sigma_sq, axis = 0, name = "z_sigma_sq_dimensionwise") #(Hidden Dimension,) -> mean removes Batch Dimension
      self.z_sigma_dimensionwise = tf.reduce_mean(z_sigma, axis = 0, name = "z_sigma_dimensionwise") # (Hidden Dimension,) -> mean removes Batch Dimension

      ### Draw one sample z from Gaussian distribution
      ### z = mu + sigma * epsilon
      self.z = self._sample_z(self.z_mean, self.z_log_sigma_sq)
      
      if self.dec_type == 'mlp':
          logit = self._create_mlp_decoder(self.z)
      if self.dec_type == 'cnn':
          logit = self._create_cnn_decoder(self.z)
          
      ### now lets give the logit output some name for later inference session running in evaluation mode
      self.logit = tf.identity(logit, name = "logit")
      self.logit_split = tf.transpose(tf.convert_to_tensor(tf.split(value = self.logit, num_or_size_splits = self.sent_len, axis = 1), name="logit_split" ), [1,0,2])
      ### logit shape: (?, 13468) logit_split shape: (?, 7, 1924)



  def _create_loss_optimizer(self):
     
    ### Create list with all running_vars count/total variables which we will extend in the following 
    self.running_vars = []

    ### Handy tf.metrics variables collected in list and runable when needed
    self.tf_metrics_tensor_dict = {}
    ### Existing summary_op collector
    summary_op_list = []

    ### Creat list for fetching the single kld losses as running metrics
    self.single_kld_losses = []
    ### Create list for tracking single kld losses updates
    self.list_single_kld_loss_update = []

    ### Creat list for fetching the means as running metrics
    self.list_means = []
    ### Create list for tracking mean updates
    self.list_means_update = []

    ### Creat list for fetching the means as running metrics
    self.list_variances = []
    ### Create list for tracking mean updates
    self.list_variances_update = []


    ### Posterior tracking
    with tf.name_scope("posterior_means"):
      ### Add dimension such that we can split object into list of tensors
      expanded_means = tf.expand_dims(self.z_mean_dimensionwise, axis=1)
      ### Now split into list of tensors
      split_means = tf.split(value = expanded_means, num_or_size_splits=self.z_dim, axis=0)
      ### Logging of single means
      for index, mean in enumerate(split_means):
          mean_name = 'z_' + str(index)
          single_mean, single_mean_update = tf.metrics.mean(mean, name=mean_name)
          ### Append each single mean-update into list of runnable tensor_updates   
          self.list_means_update.append(single_mean_update)
          ### Append each single mean-value into list of runnable tensors
          self.list_means.append(single_mean)
          self.running_vars.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "posterior_means/" + mean_name))
          single_mean_summary_op = tf.summary.scalar('mean_' + mean_name , single_mean)
          summary_op_list.append(single_mean_summary_op)


    with tf.name_scope("posterior_variances"):
      ### Add dimension such that we can split object into list of tensors
      expanded_variances = tf.expand_dims(self.z_sigma_sq_dimensionwise, axis=1)
      ### Now split into list of tensors
      split_variances = tf.split(value = expanded_variances, num_or_size_splits=self.z_dim, axis=0)
      ### Logging of single disc latent losses
      for index, variance in enumerate(split_variances):
          variance_name = 'z_' + str(index)
          single_variance, single_variance_update = tf.metrics.mean(variance, name=variance_name)
          ### Append each single variance-update into list of runnable tensor_updates
          self.list_variances_update.append(single_variance_update)
          ### Append each single variance-value into list of runnable tensors
          self.list_variances.append(single_variance)
          self.running_vars.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "posterior_variances/" + variance_name))
          single_variance_summary_op = tf.summary.scalar('variance_' + variance_name , single_variance)
          summary_op_list.append(single_variance_summary_op)


    ### Reconstruction loss
    with tf.name_scope("reconstr_loss"):  
      if self.repr_type == 'one-hot':
        reconstr_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logit_split, labels=self.x_split) #collapses axis=2
        reconstr_loss = tf.reduce_sum(reconstr_loss, axis=1) # this is the error-value PER SENTENCE #collapses axis=1

      if "word2vec" in self.repr_type:
        reconstr_loss = tf.square(tf.subtract(self.logit_split, self.x_split)) #this is SE (SQUARRED ERRORS) on axis =2
        ### Now lets get rid of the vector dimension 300
        reconstr_loss = tf.reduce_sum( reconstr_loss, axis = 2) #this is SSE per word (SUM OF SQUARRED ERRORS) #collapses axis=2
        ### Now lets get rid of the sentence length 7 by adding up all SSE values -> to get Sum of SSE (SSSE)
        reconstr_loss = tf.reduce_sum( reconstr_loss, axis=1) #this is the error-value per sentence (sum over all 7 SSE values in this sentence) #collapses axis=1
        
      reconstr_loss = tf.reduce_mean(reconstr_loss, name = "reconstr_loss") #this is equivalent to tf.reduce_sum(reconstr_loss) / batch_size -> WHICH IS THE AVERAGE error-value PER SENTENCE
      self.reconstr_loss, self.reconstr_loss_update = tf.metrics.mean( reconstr_loss, name = "metric")
      self.running_vars.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "reconstr_loss/metric"))

      reconstr_loss_summary_op      = tf.summary.scalar('reconstr_loss', self.reconstr_loss)
      summary_op_list.append(reconstr_loss_summary_op)
      self.tf_metrics_tensor_dict.update(reconstr_loss = self.reconstr_loss)

    ### Latent loss/total_kld (alias total KL divergence)
    with tf.name_scope("latent_loss"):
      klds = -0.5 * ( 1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq) ) # this is a 2D-array of size (Batch x z_dim)
      self.dimensionwise_kld = tf.reduce_mean(klds, axis = 0, name = "dimensionwise_kld") # this is a 1D-array of size (1 x z_dim)
      avg_kld = tf.reduce_mean(tf.reduce_mean(klds, axis = 1), name = "mean_kld") # this is a 1D-scalar of size (1 x 1) and the relationship to total_kld is total_kld = z_dim*total_kld
      total_kld = tf.reduce_mean(tf.reduce_sum(klds, axis = 1), name = "total_kld") # this is a scalar of size (1 x 1) and MUST BE USED FOR TOTAL LOSS OF VAE (i.e. multiplied by beta-value)
      ### total_kld == latent_loss
      self.latent_loss, self.latent_loss_update = tf.metrics.mean( total_kld, name = "metric")
      self.running_vars.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "latent_loss/metric"))
      latent_loss_summary_op   = tf.summary.scalar('latent_loss', self.latent_loss)
      summary_op_list.append(latent_loss_summary_op)
      self.tf_metrics_tensor_dict.update(latent_loss = self.latent_loss)

      ### average kld including all dims == mean_kld
      self.mean_kld, self.mean_kld_update = tf.metrics.mean( avg_kld, name = "metric_mean_kld")
      self.running_vars.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "latent_loss/metric_mean_kld"))
      mean_kld_summary_op   = tf.summary.scalar('mean_kld', self.mean_kld)
      summary_op_list.append(mean_kld_summary_op)
      self.tf_metrics_tensor_dict.update(mean_kld = self.mean_kld)


      ### Add dimension such that we can split object into list of tensors
      expanded_dimensionwise_kld = tf.expand_dims(self.dimensionwise_kld,axis=1)
      ### Now split into list of tensors
      split_klds = tf.split(value = expanded_dimensionwise_kld, num_or_size_splits=self.z_dim, axis=0)
      ### Logging of single disc latent losses
      for index, single_kld in enumerate(split_klds):
          kld_name = 'z_' + str(index)
          single_kld_loss, single_kld_loss_update = tf.metrics.mean(single_kld, name=kld_name)
  
          ### Append each single kld loss update value into list of runnable tensor_updates   
          self.list_single_kld_loss_update.append(single_kld_loss_update)
 
          ### Append each single disc latent loss value into list of runnable tensors
          self.single_kld_losses.append(single_kld_loss)

          self.running_vars.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "latent_loss/" + kld_name))
          single_kld_loss_summary_op = tf.summary.scalar('single_kld_loss_' + kld_name , single_kld_loss)
          summary_op_list.append(single_kld_loss_summary_op)
    

    ### Total Loss as final beta-VAE loss
    with tf.name_scope("total_loss"):
      total_loss = tf.add(reconstr_loss, (self.beta * total_kld), name = "total_loss")
      self.total_loss, self.total_loss_update = tf.metrics.mean( total_loss, name = "metric")
      self.running_vars.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "total_loss/metric"))
      total_loss_summary_op    = tf.summary.scalar('total_loss',
                                                         self.total_loss)
      summary_op_list.append(total_loss_summary_op)
      self.tf_metrics_tensor_dict.update(total_loss = self.total_loss)


    ### Accuracy word level e.g. "i eat the payment pad pad pad" vs "we accept the meal pad pad pad" --> equals accuracy of 4/7
    with tf.name_scope("accuracy"):

      if self.repr_type == "one-hot":
        prob_split = tf.nn.softmax(self.logit_split,axis=2)
        ### Now get the predictions as indices
        predictions_idx_split = tf.argmax(prob_split,axis=2)
        labels_idx_split = tf.argmax(self.x_split,axis=2)
        ### Now squeeze to remove last dimension of size 1
        self.predictions = tf.squeeze(predictions_idx_split, name = "prediction")
        self.labels = tf.squeeze(labels_idx_split, name = "label")

      if "word2vec" in self.repr_type:
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

      print("These are all running_vars:", self.running_vars)
      ### Define initializer in order to initialize/reset all running_vars
      self.running_vars_initializer = tf.variables_initializer(var_list = self.running_vars)


    with tf.name_scope("optimizer"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          ### Ensures that we execute the update_ops before performing the train_step
          ### Eaxmple line: train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)       
          optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
          gradients, variables = zip(*optimizer.compute_gradients(total_loss))
#          gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
#          gradients = [ None if gradient is None else tf.clip_by_norm(gradient, 5.0) for gradient in gradients]
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

#      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

    ######################### TENSORBOARD SUMMARY OPS ############################

    ### Define variables to be saved in summary (and then displayed with TensorBoard)
#    reconstr_loss_summary_op = tf.summary.scalar('reconstr_loss',       self.reconstr_loss)
#    latent_loss_summary_op   = tf.summary.scalar('latent_loss',         self.latent_loss)
#    total_loss_summary_op    = tf.summary.scalar('total_loss',          self.total_loss)
#    accuracy_summary_op      = tf.summary.scalar('word_level_accuracy', self.accuracy)
    ### Merge all summary ops to a single summary_op
#    self.summary_op          = tf.summary.merge([reconstr_loss_summary_op, latent_loss_summary_op, total_loss_summary_op, accuracy_summary_op])
                                                

    ### optional gradient global norm summary for debugging purposes
#    gradients_global_norm_summary_op = tf.summary.scalar('gradient_global_norm', self.gradients_global_norm)
#    self.debug_summary_op            = tf.summary.merge([gradients_global_norm_summary_op]) 
    

    ### Logging dimensionwise sclars from dimensionwise_kld , z_sigma, and z_mean (sclar by scalar) THIS STILL HAS TO BE DONE!
    #dimensionwise_kld_summary_op = [tf.summary.scalar('%s_dim_%d' % (kld.name,index), kld) for kld in self.dimensionwise_kld]  
 

  def printing_debug(self, sess, x_data, embedding_matrix):
      gradients, variables =  sess.run((self.gradients, self.variables),
                                      feed_dict={self.x : x_data, self.phase : True, self.embedding_matrix : embedding_matrix})
      return gradients, variables


  def partial_fit(self, sess, x_data, embedding_matrix):
      return_values                       =  sess.run(self.list_single_kld_loss_update + self.list_means_update + self.list_variances_update + 
                                                           [self.optimizer,
                                                            self.reconstr_loss_update,
                                                            self.latent_loss_update,
															self.mean_kld_update, 
                                                            self.total_loss_update,
                                                            self.accuracy_update,
                                                            self.summary_op],
                                                            feed_dict={self.x : x_data, self.phase : True, self.embedding_matrix : embedding_matrix})

      summary_str = return_values[-1]
      return summary_str


  def partial_test(self, sess, x_data, embedding_matrix):

      return_values                     =       sess.run(self.list_single_kld_loss_update + self.list_means_update + self.list_variances_update +
                                                           [self.reconstr_loss_update,
                                                            self.latent_loss_update,
															self.mean_kld_update, 
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


  def reconstruct(self, sess, x_data, embedding_matrix):
      ###  Reconstruct given data.
      ###  Original VAE output: x -> logit_split
      return sess.run(self.logit_split,
                      feed_dict={self.x: x_data, self.phase : False, self.embedding_mtrix : embedding_matrix})
  
  def predict(self, sess, x_data, embedding_matrix):
       ### gives the predictions given data. These predictions are simply indices which denote the token using the id-to-token-dict.
       ### x -> argmax predictions
       return sess.run(self.predictions,
                       feed_dict={self.x : x_data, self.phase : False, self.embedding_matrix : embedding_matrix})

  def transform(self, sess, x_data, embedding_matrix):
      ### Transform data by mapping it into the latent space.
      ### x -> z, z_mean, z_log_sigma_sq (=z_log_var)
      return sess.run([self.z, self.z_mean, self.z_log_sigma_sq],
                       feed_dict={self.x: x_data, self.phase : False, self.embedding_matrix : embedding_matrix })

  def generate(self, sess, z_data):
      ### Generate data by sampling from latent space.
      ### z -> logit_split
      return sess.run(self.logit_split,
                      feed_dict={self.z: z_data, self.phase : False})

  def get_means_and_vars(self, sess, x_data, embedding_matrix):
      ### Transform data by mapping it into dimension-wise mean and dimension-wise variacne.
      return sess.run([self.z_mean_dimensionwise, self.z_sigma_sq_dimensionwise],
                       feed_dict={self.x : x_data, self.phase : False, self.embedding_matrix : embedding_matrix})

    
  def generate_vote(self, sess, L_sentences, factor_index):  
    z, _, z_log_sigma_sq  = self.transform(sess, L_sentences)
    z_sigma_sq = tf.exp(self.z_log_sigma_sq) # (L x Hidden Dimension)
    z_sigma = tf.sqrt(z_sigma_sq) # (L x Hidden Dimension) 
    z_sigma_dimensionwise = tf.reduce_mean(z_sigma, axis = 0) # (Hidden Dimension,) in effect it is 1-Dimensional
    
    # now check if all latents still collapsed 
    booleans = tf.greater(z_sigma_dimensionwise, self.prun_thold) #booleans is a vector or True/False entries like this
    # if True: this position has a z_sigma value larger prun_thold)
    # if False: this position as a z_sigma value smaller than prun_thold
   
    # fetch the boolean value for all_collapsed by running a session
    all_collapsed = sess.run(tf.reduce_all(booleans))
    if(all_collapsed):
        d_prime = -1
        vote = (d_prime, factor_index)
    else:
        # if we get into this else-section, this means, that AT LEAST A SINGLE latent is NOT COLLAPSED AND THEREFORE ALLOWED TO VOTE!
        # now let us 1) calculate the std-vector of z, 2) normalize z by the std-vector and finally 3) calculate the emp-var-vector
        s = tf.keras.backend.std(z, axis = 0, keepdims = True)
        normalized_z = tf.divide(z, s)
        pre_emp_vars = tf.keras.backend.var(normalized_z, axis = 0, keepdims=True)
        # NOW WE HAVE TO SET THE EMPIRICAL_VARS entries where z_sigma_dimensionwise > prun_thold UP TO A HIGH VALUE (e.g. 5.0)
        # only by doing this, we are able to exclude them from voting
        # now we create the vector emp_vars_prun_values which we later on ADD ONTO empirical_vars in order to prun the voting rights of collapsed(invalid) latents
        emp_vars_prun_values = tf.cast(booleans, tf.float32)
        emp_vars_prun_values = tf.multiply(emp_vars_prun_values, 5)
        emp_vars = tf.add(pre_emp_vars, emp_vars_prun_values)
        # get the min-value
        min_value = tf.reduce_min(emp_vars)
        # get indices where entry equal to min-value
        min_indices = tf.where(tf.equal(emp_vars, min_value))
        # in case we ended up with several min-indices, randomly choose one of it   
        # now let us shuffle min indices before we choose one of those values
        min_indices = tf.random_shuffle(min_indices)
        min_idx = min_indices[0]
        # finally assing d_prime by running the session and fetching the output
        d_prime = sess.run(min_idx)
        vote = (d_prime, factor_index)
    return vote

  def majority_vote_classifier(self, votes, manager, M):
      hidden_dims = self. z_dim
      factor_dims = manager.n_generative_factors
      # create empty TF array which we will fill up in a loop:
      V_matrix = tf.zeros(shape = (hidden_dims, factor_dims))
      return 0

  def d_score(self):
    d_score = 0
    return d_score


  def return_batch(self, X, batch_indices): 
      batch = X[batch_indices]
      return batch

  def train(self, sess, manager,saver, args):
      ### Plot Steps Buffer List
      plot_steps = []
    
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

      #set values for train and test cases    
      n_train_samples  = manager.train_size
      n_test_samples   = manager.test_size

      print("n_train_samples:", n_train_samples)
      print("n_test_samples:", n_test_samples)

      #calculate amount of batches for training and testing
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
      train_indices    = manager.train_indices  
      test_indices     = manager.test_indices    
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

      #load embedding matrix
      embedding_matrix = manager.embedding_matrix

      # step is the global step counter (number of so far processed X_train batches),
      # we can see it in tensorboard on the x-axis 
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

########################################################################################################################################################


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

######################################################## FETCH MEAN AND Z_SIGMA ########################################################################



      


########################################################################################################################################################

      ### Evaluate initial metric
      ### the following two lines have to be executed in every epoch once!
      code_test = self.transform(sess, metric_X_test, embedding_matrix)[1] ### [1] because it is returning soft_samples (which are close to one-hot) and hard_samples (which are exactly one-hot) 
      MIGs_test.append(utils.MIG_score(GFs_test, code_test, inv_GF_entropies))

      ### Create customized summary for MIG-metric
      MIG_summary_str = tf.Summary(value=[tf.Summary.Value(tag="MIG", simple_value=MIGs_test[-1]),])  ###always add the most current value within the list (i.e. the last one by [-1])

      ### Add current current MIG value to summary
      test_writer.add_summary(MIG_summary_str, epoch)

      ### Evaluate initial metric
      ### the following two lines have to be executed in every epoch once!
      code_train = self.transform(sess, metric_X_train, embedding_matrix)[1] ### [1] because it is returning soft_samples (which are close to one-hot) and hard_samples (which are exactly one-hot) 
      MIGs_train.append(utils.MIG_score(GFs_train, code_train, inv_GF_entropies))

      ### Create customized summary for MIG-metric
      MIG_summary_str = tf.Summary(value=[tf.Summary.Value(tag="MIG", simple_value=MIGs_train[-1]),])  ###always add the most current value within the list (i.e. the last one by [-1])

      ### Add current current MIG value to summary
      train_writer.add_summary(MIG_summary_str, epoch)

########################################################################################################################################################

      ### Do single loop for train-data  in order to have a value at step 0 (before we start training)
      for batch_idx in range(n_train_batches):
          batch_indices      = np.arange(self.batch_size*batch_idx, self.batch_size*(batch_idx+1))
          train_batch         = self.return_batch(X_train, batch_indices)
          train_summary_str   = self.partial_test(sess, train_batch, embedding_matrix)

      outputs = self.run_tf_metric_tensors(sess, self.tf_metrics_tensor_dict)
      train_string = ", ".join(["train "+str(key)+": "+str(round(value,3)) for (key,value) in outputs.items()])
      print("Intitial train metrics:\n")
      print(train_string)
    
      #resetting all running vars
      sess.run(self.running_vars_initializer)

#      print("Metrics after reset as sanity-check:", self.run_tf_metric_tensors(sess, self.tf_metrics_tensor_dict))     
 
      # Do single loop for test-data in order to have a values at step 0 (before we start training)
      for batch_idx in range(n_test_batches):
          batch_indices      = np.arange(self.batch_size*batch_idx, self.batch_size*(batch_idx+1)) 
          test_batch         = self.return_batch(X_test, batch_indices)
          test_summary_str   = self.partial_test(sess, test_batch, embedding_matrix)

      outputs = self.run_tf_metric_tensors(sess, self.tf_metrics_tensor_dict)
      test_string = ", ".join(["test "+str(key)+": "+str(round(value,3)) for (key,value) in outputs.items()])
      print("Intitial test metrics:\n")
      print(test_string)
      
      #resetting all running vars
      sess.run(self.running_vars_initializer)

      ### Write initial accuracies of train and test data to file-writer
      train_writer.add_summary(train_summary_str, epoch)
      test_writer.add_summary(test_summary_str, epoch)

      ### Create lists of gf and code names for plotting the corr_mat later on
      gf_names = manager.list_of_gf_strings
      n_code_dims = code_train.shape[1]
      code_names = ["c"+str(i) for i in range(n_code_dims)]


      ### For comparision purposes save corr matrix at the beginning of training
      ### Create corr_matr between code and gf
      plt.figure(figsize=(10,5))
      plt.title("GF vs. Code Correlation Test Set")
      R = utils.MI_mat(GFs_test, code_test, inv_GF_entropies)
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
      R = utils.MI_mat(GFs_train, code_train, inv_GF_entropies)
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
      # Training cycle
      for epoch in range(self.epoch_size):
        # Shuffle train indices
        np.random.shuffle(train_indices)
        
        # Loop over all batches for training
        for batch_idx in range(n_train_batches):
            # Generate train batch
            batch_indices     = np.arange(self.batch_size*batch_idx, self.batch_size*(batch_idx+1))          
            train_batch       = self.return_batch(X_train, batch_indices)
            train_summary_str = self.partial_fit(sess, train_batch, embedding_matrix)
            # increase step after having processed single batch 
            step += 1

        ###  Fetch train metrics before we reset running_vars
        outputs = self.run_tf_metric_tensors(sess, self.tf_metrics_tensor_dict)
        train_string = ", ".join(["train "+str(key)+": "+str(round(value,3)) for (key,value) in outputs.items()])

        # now reset running_vars
        sess.run(self.running_vars_initializer) 

        ### Loop over all batches for testing
        ### Prevent looping over all n_test_batches because in case n_tuples is large, and train_split is small, this might take a while
        ### 1.so that we dont test over same array every time
        np.random.shuffle(X_test)

        for batch_idx in range(n_test_batches):
            batch_indices        = np.arange(self.batch_size*batch_idx, self.batch_size*(batch_idx+1))
            test_batch           = self.return_batch(X_test, batch_indices)
            test_summary_str     = self.partial_test(sess, test_batch, embedding_matrix)         
                
        # Fetch test metrics before we reset running_vars
#        test_reconstr_loss, test_latent_loss, test_total_loss, test_accuracy = self.run_tf_metric_tensors(sess, tf_metrics_tensor_list) 
        ### Fetch test metrics before we reset running_vars
        outputs = self.run_tf_metric_tensors(sess, self.tf_metrics_tensor_dict)
        test_string = ", ".join(["test "+key+": "+str(round(value,3)) for (key,value) in outputs.items()])

        #now reset running vars
        sess.run(self.running_vars_initializer)

        ### Last but not least after every epoch we can write to file-writers:
        train_writer.add_summary(train_summary_str, epoch+1)
        test_writer.add_summary(test_summary_str, epoch+1)

        ### Printing out statistics of current epoch
        print("Epoch {} statistics:\n".format(epoch+1))

        print(train_string)
        print(test_string)
        print("Now reconstruction examples from the test set.")

        # Keep track of sentences evolution of test dataset(can we generalize?), this is done after every epoch
        input_indices = np.random.choice(test_indices, 10, replace=False)
        input_indices = list(input_indices)
        utils.reconstruct_sentences(self,sess,input_indices,epoch,manager)
     
##################################################################################################################################################################
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

##################################################################################################################################################################

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

##################################################################################################################################################################

        ### the following two lines have to be executed in every epoch once!
        code_test = self.transform(sess, metric_X_test, embedding_matrix)[1] ### [1] because it is returning soft_samples (which are close to one-hot) and hard_samples (which are exactly one-hot) 
        MIGs_test.append(utils.MIG_score(GFs_test, code_test, inv_GF_entropies))

        ### Create customized summary for MIG-metric
        MIG_summary_str = tf.Summary(value=[tf.Summary.Value(tag="MIG", simple_value=MIGs_test[-1]),])  ###always add the most current value within the list (i.e. the last one by [-1])

        ### Add current current MIG value to summary
        test_writer.add_summary(MIG_summary_str, epoch+1)

        ### Evaluate initial metric
        ### the following two lines have to be executed in every epoch once!
        code_train = self.transform(sess, metric_X_train, embedding_matrix)[1] ### [1] because it is returning soft_samples (which are close to one-hot) and hard_samples (which are exactly one-hot) 
        MIGs_train.append(utils.MIG_score(GFs_train, code_train, inv_GF_entropies))

        ### Create customized summary for MIG-metric
        MIG_summary_str = tf.Summary(value=[tf.Summary.Value(tag="MIG", simple_value=MIGs_train[-1]),])  ###always add the most current value within the list (i.e. the last one by [-1])

        ### Add current current MIG value to summary
        train_writer.add_summary(MIG_summary_str, epoch+1)

        ### Last but not least after every epoch we can write to file-writers:
        train_writer.add_summary(train_summary_str, epoch+1)
        test_writer.add_summary(test_summary_str, epoch+1)

        # Finally before starting a new epoch save checkpoint! This is done after every epoch
        # setting "write_meta_graph" to False as we only want to save the .meta file once at the very first saving step, 
        # we do not need to recreate the .meta file in every epoch we save the session, as the graph does not change after creation 
        print("now saving checkpoint for epoch number {}.".format(epoch))
        saver.save(sess, args.checkpoints_dir + self.experiment_name+ "/" + "checkpoint"  , global_step = epoch, write_meta_graph = False)
        
    
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
      R = utils.MI_mat(GFs_test, code_test, inv_GF_entropies)
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
      image_summary = tf.summary.image("MI Correlation Test", image, max_outputs=1)
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
      R = utils.MI_mat(GFs_train, code_train, inv_GF_entropies)
      print("R-matrix:",R)
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
      image_summary = tf.summary.image("MI Correlation Train", image, max_outputs=1)
      image_summary_op = sess.run(image_summary)
      train_writer.add_summary(summary=image_summary_op,global_step=epoch+1)
