import numpy as np
import tensorflow as tf
import argparse
import os
import time
import shutil
from sklearn.metrics.cluster.supervised import mutual_info_score ### MI metric for discrete code
from sklearn.feature_selection.mutual_info_ import _compute_mi_cd ### MI metric for continuous code
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
from scipy.stats import pearsonr
from pattern.en import singularize, pluralize, lexeme ### singularize, pluralize to check for object, lexeme to get all possible verb style (see,saw,seeing,sees,seen)

####################################################  beta TC VAE  DISENTANGLEMENT METRIC  ############################################################# 



### Determine the normalizing inverse entropy value for each gf
def calc_inv_entropies(latents_sizes, n_used_tuples=None):
    if n_used_tuples is not None:
       latents_sizes[0] = n_used_tuples
    else:
        pass
    entropies = np.log(latents_sizes)
    inv_entropies = (1/entropies)
    return inv_entropies


### Calculate total MIG by averaging over all MIG scores and normalizing by the max value for each gf variable (in effect its entropy=
def total_MIG(MIG_scores):
    total_MIG = np.mean(MIG_scores)
    return total_MIG


### calculate the Mutual information gap between the top 2 contributing code variaables
def MIG_score(gf,code,inv_GF_entropies, random=False):
    code_is_disc = False
    code_is_cont = False
    if "int" in code.dtype.name:
        code_is_disc = True
    elif "float" in code.dtype.name:
        code_is_cont = True
    else:
        print("Code data type must be int or float!")
    if random:
        np.random.shuffle(code)
    else:
        pass
    MIG_scores = []
    for gf_idx in range(gf.shape[1]):
        gf_column = gf[:,gf_idx]
        GF_norm_value = inv_GF_entropies[gf_idx]
        MI_scores = []
        for code_idx in range(code.shape[1]):
            code_column = code[:,code_idx]
            ### Case discrete VAE
            if code_is_disc:
                MI = ( mutual_info_score(code_column,gf_column) * GF_norm_value)
            ### Case continuous VAE
            elif code_is_cont:
                MI = ( _compute_mi_cd(code_column,gf_column,n_neighbors=3) * GF_norm_value)

            MI_scores.append(MI)

        MI_scores = np.array(MI_scores)
        MI_scores = np.sort(MI_scores)
        top2_MI_scores = MI_scores[-2:]
        MIG_score = abs(top2_MI_scores[0] - top2_MI_scores[1])
        MIG_scores.append(MIG_score)
    MIG_scores = np.array(MIG_scores)
    MIG = total_MIG(MIG_scores)
    return MIG

### Calculate the argmax_mapped code
def calc_argmax_code(code,split_indices):
    n_rows = code.shape[0]
    n_cols = len(split_indices)+1 ### #1 because if we split an array with 2 given indices we end up with 3 seperate subarrays
    argmax_code = []
    for row in code:
        splits = np.split(row,split_indices)
        for split in splits:
            argmax = np.argmax(split)
            argmax_code.append(argmax)
    argmax_code = np.array(argmax_code).reshape(n_rows,n_cols)
    return argmax_code

####################################################  FACTOR VAE DISENTANGLEMENT METRIC  #############################################################   
 
def generate_vote(model, sess, L_sentences, factor_index, prun_thold=0.75):
  #1 get representation and the z_log_sigma_sq
  z, _, z_log_sigma_sq = model.transform(sess, L_sentences) # z is (L,z_dim)
  #2 calculate empirical standard deviation
  s = z.std(axis = 0)  #s is (1,z_dim)
  #3 normalize by empirical standard deviation
  normalized_z = z/s
  #4 take empirical variance over all L sentences
  empirical_vars = normalized_z.var(axis = 0) # empirical_variances is (1,z_dim)

  #5 before choosing argmin in empirical variances we have to prune away latent variables which are collapsed to prior
  #5.1 calculate dimensionwise z_sigma (standard deviation)
  z_sigma_sq = np.exp(z_log_sigma_sq)
  z_sigma = np.sqrt(z_sigma_sq)
  z_sigma_dimensionwise = np.mean(z_sigma, axis = 0)
  #5.2 determine invalid latents position 
  invalid_latents = np.where(z_sigma_dimensionwise > prun_thold)[0]
  #6 modify empirical variances, this is the "pruning step"
  # NOW WE HAVE TO SET THE EMPIRICAL_VARS entries where z_sigma_dimensionwise > prun_thold UP TO 2.0
  # only by doing this, we are able to exclude them from voting
  empirical_vars[invalid_latents] = 5.0
  # in case the latents have not fully developed yet as all of them start from a collapsed sigma=1 case
  if len(invalid_latents) == model.z_dim:
      # no latent variables have developed yet, therefore choose randomly to keep score low around 1/n_generative_factors
#      d_prime = np.random.choice(model.z_dim)
      # let us set a signal to indicate, that still all latents are invalid
      d_prime = -1
      #create vote tuple pair
      vote = (d_prime, factor_index)

  else:
      # tie-breaking with np.random choice in case of several min-values -> in order to prevent deterministical choice of first index of all min-values
      # 1st retrieve indices where the condition is true, using np.where which returns a tuple condition.nonzero(), it contains in the first position the indices where condition is True.
      # we fetch first element with [0] as a possible output of np.where looks like this: 
      # In [85]: a = np.array([0.05, 0.04, 0.03, 0.01, 0.67, 0.05, 0.01, 0.03, 0.87, 0.01])
      # In [86]: np.where( a == a.min() )
      # Out[86]: (array([3, 6, 9]),)
      min_indices = np.where ( empirical_vars == empirical_vars.min() )[0]
      # in case we ended up with several min-indices, randomly choose one of it 
      d_prime = np.random.choice( min_indices )
      #6 create vote tuple pair
      vote = (d_prime, factor_index)


  # THIS VOTE IS A SINGLE TRAINING POINT FOR THE MAJORITY VOTE CLASSIFIER
  return vote

#d_prime is in range 0 to z_dim
#factor_index is in range 0 to 9()
#def majority_vote_classifier(votes, list_of_valid_latents, manager, z_dim = 20):
def majority_vote_classifier(votes, manager, model, M):
#    print("Number of votes:", len(votes))
    hidden_dimensions = model.z_dim
    factor_dimensions = manager.n_generative_factors
    V_matrix = np.zeros(shape=(hidden_dimensions,factor_dimensions),dtype=int)
#    voting_latents = list(set.intersection(*map(set,list_of_valid_latents)))
    for vote in votes:
        if vote[0] == -1:
            pass
        else:
            V_matrix[vote[0],vote[1]] += 1
    largest_values_each_row = np.amax(V_matrix,axis=1)
    print("The V_matrix:\n", V_matrix)
#    print("Largest values each row:", largest_values_each_row)
    correct_sum = np.sum(largest_values_each_row)
    print("Value of correct sum:", correct_sum)
#    print("Sum of all entries in V_matrix:", np.sum(V_matrix))
    accuracy = correct_sum / M 
    return accuracy


# L = samples per vote
# M = number of total votes/training points  
def disentanglement_score(model, manager, sess, L, M):
      votes = []
      for m in range(M):

          #1 get random factor_index and sample a value/form of it
          factor_index = np.random.randint(manager.n_generative_factors)
          form_value = np.random.randint(manager.latents_sizes[factor_index])
          # print("Factor index: {} and Form value: {}".format(factor_index,form_value))
          #2 get L = 100 sentences with fixed factor k (between 0 and 9) and fix form value
          L_sentences = manager.get_L_sentences(L, factor_index, form_value)
          #decode_sentences(L_sentences, manager)
          
          #ignore votes which contain d_prime == 1 because this means, that we still
          #do not have a z_dim which is non-collapsed (i.e. z_sigma < prun_thold
          vote = generate_vote(model, sess, L_sentences, factor_index) 
          votes.append(vote)
      score = majority_vote_classifier(votes, manager, model, M)
      return score

################################################################# END OF DISENTANGLEMENT METRICS ###############################################################

############################### Calculating correlation matrix between ground-truth (GF) and hidden-state(latent code) ########################################

def corr_mat(gf,code):
    ### GF
    n_gf_dims = gf.shape[1]
    ### Code
    n_code_dims = code.shape[1]
    ### Create empty corr_mat of correct shape
    R = np.empty(shape=(n_gf_dims,n_code_dims))     

    for gf_idx in range(n_gf_dims):
        for code_idx in range(n_code_dims):
            R[gf_idx,code_idx] = abs(pearsonr(gf[:,gf_idx], code[:,code_idx])[0]) ### [0] because pearsonr returns (corr,p-value)
    return R

###############################################################################################################################################################

############################### Calculating correlation matrix between ground-truth (GF) and hidden-state(latent code) ########################################

def MI_mat(gf,code,inv_GF_entropies):
    ### determine if we should use MI(disc,disc) or MI(disc,cont)
    code_is_disc = False
    code_is_cont = False
    if "int" in code.dtype.name:
        code_is_disc = True
    elif "float" in code.dtype.name:
        code_is_cont = True
    else:
        print("Code data type must be int or float!")

    ### Number of GF variables
    n_gf_dims = gf.shape[1]
    ### Number of code variables
    n_code_dims = code.shape[1]
    ### Create empty corr_mat of correct shape
    MI = np.empty(shape=(n_gf_dims,n_code_dims))

    for gf_idx in range(gf.shape[1]):
        gf_column = gf[:,gf_idx]
        GF_norm_value = inv_GF_entropies[gf_idx]
        for code_idx in range(code.shape[1]):
            code_column = code[:,code_idx]
            ### Case discrete VAE
            if code_is_disc:
                single_mi = ( mutual_info_score(code_column,gf_column) * GF_norm_value)
            ### Case continuous VAE
            elif code_is_cont:
                single_mi =  ( _compute_mi_cd(code_column,gf_column,n_neighbors=3) * GF_norm_value)
            #print("single mi for gf_idx_{} and code_idx_{} is:{}".format(gf_idx,code_idx,single_mi))
            ### Now place single_mi value into the MI-"corr"-matrix
            MI[gf_idx,code_idx] = single_mi
    return MI

###############################################################################################################################################################

def reconstruct_sentences(model, sess, input_indices, epoch, manager):
    print('Testing current model language reconstruction')
    input_sentences = manager.get_sentences(input_indices)
    original_sentences = manager.ground_truth(input_indices)
    # directly fetch the predictions with model.predict
    # the output of model.reconstruct is logit_split vector
    predictions = model.predict(sess, input_sentences, manager.embedding_matrix)
    list_of_sentences = []
    for prediction in predictions:
        sentence = [manager.id_to_token[entry] for entry in prediction]
        sentence = " ".join(sentence)
        list_of_sentences.append(sentence)
    print("Current reconstructions for epoch {} look like these:".format(epoch+1))
    for idx, entry in enumerate(list_of_sentences):
        print("\n")
        print("Original Sentence:", " ".join(original_sentences[idx]))
        print("Reconstr Sentence:", entry)

### Returns a list of lists; each list contains words(as single-string) of the sentence        
def return_recon_sent(model,sess,recon_indices,manager,random=False):
    """ Input to this function <recon_indices> is simply a list of single indices 
        which describe the location of the sentence in the main raw dataset """
    recon_input_sentences = manager.get_sentences(recon_indices)    
    predictions = model.predict(sess,recon_input_sentences,manager.embedding_matrix)
    if random:
        print("shuffling the predictions")
        np.random.shuffle(predictions)
    else:
        pass
    recon_sentences = []
    for prediction in predictions:
        ### sentence is a list of words(strings)
        sentence = [manager.id_to_token[entry] for entry in prediction]
        while sentence[-1] in ["<PAD>","."]:
            del sentence[-1]
        recon_sentences.append(sentence)
    return recon_sentences



### Following fucntions are for gf accuracies metric ### ------------------------------------------------------------------------------------------

### Obj-Verb-Tuple
def check_gf1(gt_sent,recon_sent):
    ### Check for verb-object tuple; heuristic of matching singluar version and
    ### Same first 3 letters for the verb as shortest verb is of length 3 e.g. Hug :)
    gt_verb = gt_sent[-3]

    verbs = lexeme(gt_verb) ###returns a list of conjug. verbs
   # print("gt sent is:", gt_sent)
   # print("recon sent is:", recon_sent) 
#    print("list of gt verb alternatives:",verbs)
    gt_obj = gt_sent[-1]
   # print("gt obj is:",gt_obj)
   # print("last word in recon sent:",recon_sent[-1])
   # print("gt verb is:", gt_verb)
    objs = [singularize(gt_obj), pluralize(gt_obj)] ###return a list with pos1 sing.obj and pos2 pl.obj. 
    score = 0.0
    if any(i in verbs for i in recon_sent):
        ###  Reward learning the verb
        score = score + 0.5
    if any(i in objs for i in recon_sent):
        ### Reward learning the object
        score = score + 0.5
    
    return score


### Obj-pl./sing.    
def check_gf2(gt_sent,recon_sent):
    ### 1. extrct ground truth object
    gt_obj = gt_sent[-1][-1]
    ### 2. detect if gt-object is sinular or plural version
    gt_objs = [singularize(gt_obj), pluralize(gt_obj)]
    if (gt_obj == gt_objs[0]):
        gt_is_plural = False
    elif (gt_obj == gt_objs[1]):
        gt_is_plural = True
    gt_is_plural = (gt_sent[-1][-1] == "s")
    ### if any of the last 3 words contains a "s" at the end, we are happy to call it plural
    ###                           LAST-WORD                SECOND-TO-LAST-WORD            THIRD-TO-LAST-WORD 
#    recon_is_plural = ( (recon_sent[-1][-1] == "s") or (recon_sent[-2][-1] == "s") or (recon_sent[-3][-1] == "s"))
    ### if any of the last 3 words contains a "s" at the end, we are happy to call it plural
    ###                           LAST-WORD                SECOND-TO-LAST-WORD
    recon_is_plural = ( (recon_sent[-1][-1] == "s") or (recon_sent[-2][-1] == "s") )
    if gt_is_plural == recon_is_plural:
        return 1
    else:
        return 0

###new gender measuring technique
def check_gf3(gt_sent,recon_sent):
    ### 1st check if the ground truth sentence can represent gender (i.e. if it contains he/she it can)
    gt_male = ("he" in gt_sent)
    gt_fem  = ("she" in gt_sent)
    if (gt_male or gt_fem):
        can_repr = 1
    else:
        can_repr = 0
    
    if gt_male:
        if "he" in recon_sent:
            return_value = 1
        else:
            return_value = 0
    elif gt_fem:
        if "she" in recon_sent:
            return_value = 1
        else:
            return_value = 0
    else:
        return_value = 0

    return (return_value, can_repr)
    
### subj_sing./pl.
def check_gf4(gt_sent,recon_sent):
    can_repr = 1 

    sing_indicator = ["i","he","she"]
    plural_indicator = ["we","they"]
 
    gt_is_sing = any(i in sing_indicator for i in gt_sent)
    gt_is_plural = any(i in plural_indicator for i in gt_sent) 

    if gt_is_plural:
        recon_is_plural = any(i in plural_indicator for i in recon_sent)
        if gt_is_plural == recon_is_plural:
            return_value = 1
        else:
            return_value = 0

    elif gt_is_sing:
        recon_is_sing = any(i in sing_indicator for i in recon_sent) 
        if gt_is_sing == recon_is_sing:
            return_value = 1
        else:
            return_value = 0

    else:
        ### The case where we cannot determine if sing./pl. version of subject ("you" cases)
        can_repr = 0
        return_value = 0

    return (return_value, can_repr)


### sent_type    
def check_gf5(gt_sent,recon_sent):
    question_indicators = ["was","were","am","did","does","do","is","are","will"]
    first_gt_word = gt_sent[0]
    first_recon_word = recon_sent[0]

    gt_is_question = (first_gt_word in question_indicators)
    recon_is_question = (first_recon_word in question_indicators)

    ### if both are false, then this will be true again :)
    if (gt_is_question == recon_is_question):
        return_value = 1
    else:
        return_value = 0
    
    return return_value

### 1st/2nd/3rd person
def check_gf6(gt_sent,recon_sent):
    first_pers_cond = ["i","we"]
    second_pers_cond = ["you"]
    third_pers_cond = ["they","he","she"]
    ### gt is 1st person
    if any(i in first_pers_cond for i in gt_sent):
        if any(i in first_pers_cond for i in recon_sent):
            return 1
        else:
            return 0
    ### gt is 2nd person
    elif any(i in second_pers_cond for i in gt_sent):
        if any(i in second_pers_cond for i in recon_sent):
            return 1
        else:
            return 0
    ### gt is 3rd person
    elif any(i in third_pers_cond for i in gt_sent):
        if any(i in third_pers_cond for i in recon_sent):
            return 1
        else:
            return 0
  

### pos_neg_verb
def check_gf7(gt_sent,recon_sent):
    gt_is_neg = "not" in gt_sent
    recon_is_neg = "not" in recon_sent
    if gt_is_neg == recon_is_neg:
        return 1
    else:
        return 0


### verb_tense
def check_gf8(gt_sent,recon_sent):
    fut_indicator = ["will"]
    past_indicator = ["did","were","was"]
    comb_indicator = fut_indicator + past_indicator

    ### Case ground truth is future tense
    if any(i in fut_indicator for i in gt_sent):
        if any(i in fut_indicator for i in recon_sent):
            return 1
        else:
            return 0
              
    ### Case ground truth is past tense
    elif any(i in past_indicator for i in gt_sent):
        if any(i in past_indicator for i in recon_sent):
            return 1
        else:
            return 0
 
    ### All other cases ground truth MUST! be present tense
    else:
        ### following 2 vars (lines) are just initialization to make them even exist
        ### Check if recon_sent is NOT future and NOT past
        if any(i in comb_indicator for i in recon_sent):
            recon_is_present = False
        else:
            recon_is_present = True

        ### if this statement below is true, then both have learnt present tense
        if recon_is_present:
            return 1
        else:
            return 0


### verb_style
def check_gf9(gt_sent,recon_sent):
    gt_is_progressive = ("ing" in gt_sent[-3])
    ### [:-2] because in a properly reconstructed sentence, the last two words are "the" and the object itself    
    if any("ing" in word for word in recon_sent[:-2]):
        recon_is_progressive = True
    else:
        recon_is_progressive = False
        
    if gt_is_progressive == recon_is_progressive:
        return 1
    else:
        return 0
    
def gf_accuracy(model,sess,manager,recon_indices,gt_sentences,random=False):
    recon_sentences = return_recon_sent(model,sess,recon_indices,manager,random=random)
    ### Matches is a "holder" for match scores (boolean)
    matches = []
    gend_counter = 0
    subj_counter = 0    
    for idx,gt_sent in enumerate(gt_sentences):
        match = []
        recon_sent = recon_sentences[idx]
        match.append(check_gf1(gt_sent,recon_sent))
        match.append(check_gf2(gt_sent,recon_sent))

        ### GENDER
        (gend_score, can_repr_gend) = check_gf3(gt_sent,recon_sent)   
        gend_counter = gend_counter + can_repr_gend 
        match.append(gend_score)       
        ### 1st/2nd/3rd PERSON
        (subj_score, can_repr_subj) = check_gf4(gt_sent,recon_sent)
        subj_counter = subj_counter + can_repr_subj
        match.append(subj_score)        

        match.append(check_gf5(gt_sent,recon_sent))
        match.append(check_gf6(gt_sent,recon_sent))
        match.append(check_gf7(gt_sent,recon_sent))
        match.append(check_gf8(gt_sent,recon_sent))
        match.append(check_gf9(gt_sent,recon_sent))         
        matches.append(match)
    
    ### convert matches from list into a np.array
    matches = np.array(matches)
    accuracies = list(np.mean(matches,axis=0))

    n_samples = matches.shape[0]

    gender_accuracy = accuracies[2]
    corrected_gender_accuracy = gender_accuracy*(n_samples/gend_counter)
    accuracies[2] = corrected_gender_accuracy

    subj_accuracy = accuracies[3]
    corrected_subj_accuracy = subj_accuracy*(n_samples/subj_counter)
    accuracies[3] = corrected_subj_accuracy

    return accuracies


### ------------------------------------------------------------------------------------------------------------------------------------------------------------


### Normalzing accuracies (which are in range 0-1)
def normalize_by_first_row(in_array):
    normalizing_values = in_array[0,:]
    #print("this is what the first row looks like:", normalizing_values)
    ones = np.ones_like(in_array)
    ### Normalize in_array
    #print("this is what the current row looks like:", in_array[-1,:])
    normalized_in_array = (in_array - normalizing_values) / (ones - normalizing_values)
    return normalized_in_array


#this function "decode_predictions" is expecting predictions in form of a list of vectors, 
#where each vector already contains 7 ID-values which we can then look up in the id_to_token_dict
def decode_predictions(predictions, id_to_token_dict):

    list_of_sentences = []
    for prediction in predictions: 
        sentence = [id_to_token_dict[entry] for entry in prediction]
        sentence = " ".join(sentence)
        list_of_sentences.append(sentence)

    print("Reconstruction:")
    for sentence in list_of_sentences:
        print(sentence)


#input "L_sentences" is a numpy array of shape (L x input_dim) , therefore we have to split the input_dim
#into a a list of vectors (one-hot here as we can see that we still have to retrieve the idx with np.argmax)        
def decode_sentences(L_sentences, manager):
    print("Printing L sentences!")
    print("Shape of L_sentences before split:", L_sentences.shape)
    for row in L_sentences:
        split = np.split(row,7) # split is a list of arrays
        print("Shape of a row:", row.shape)   
        sentence = [manager.id_to_token[np.argmax(vector)] for vector in split]
        sentence = " ".join(sentence)
        print("\n")
        print("Sentence with fixed factor k:", sentence)


def str2bool(cmd_line_flag):
    # for parsing boolean values with argparse
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if cmd_line_flag.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif cmd_line_flag.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def create_checkpoint_saver(sess, args, model):
    saver = tf.train.Saver(max_to_keep = args.keep_n_ckpts, save_relative_paths = True)
    exp_checkpointdir = args.checkpoints_dir + model.experiment_name
    tf.train.export_meta_graph(filename= exp_checkpointdir + "/" + "checkpoint.meta")
    if not os.path.exists(exp_checkpointdir):
        os.mkdir(exp_checkpointdir)
#    else:
        ### 1. Remove experiment directory
#        shutil.rmtree(exp_checkpointdir)
        ### 2. Create experiment directory
#        os.mkdir(exp_checkpointdir)
    return saver


### ---------------------------------------------------------------------------------------------------------------------------------

### Helper functions for tensorflow shape extraction and reshaping

def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims

def reshape(tensor, dims_list):
  shape = get_shape(tensor)
  dims_prod = []
  for dims in dims_list:
    if isinstance(dims, int):
      dims_prod.append(shape[dims])
    elif all([isinstance(shape[d], int) for d in dims]):
      dims_prod.append(np.prod([shape[d] for d in dims]))
    else:
      dims_prod.append(np.prod([shape[d] for d in dims]))
  tensor = tf.reshape(tensor, dims_prod)
  return tensor

### ----------------------------------------------------------------------------------------------------------------------------------

### The code comes from stackoverflow : https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python/35199035#35199035
class Timer:
  def __init__(self):
    #when the timer gets initialized, it is already having a time-stamp initialized
    self.start_stamp = time.time()

  def restart(self):
    #we can restart the time in case it has an old time-stamp   
    self.start_stamp = time.time()

  def get_past_time(self):
    #we can get the past time by simply getting the current time-stamp and substracting the  old time-stamp
    end_stamp = time.time() #value of end is in seconds
    minutes, seconds = divmod(end_stamp - self.start_stamp, 60) #this (returns minutes, seconds)
    hours, minutes = divmod(minutes, 60) #this returns (hours, minutes)
    time_str = "%02d:%02d:%02d" % (hours, minutes, seconds)
    return time_str
  
