import argparse
import subprocess
import numpy as np

def main(args):
    ### Create learning_rates list
    
    learning_rates = [5.0e-5]
    ### Create disc_gammas list
#    disc_gammas = [0.0, 1.5]
    disc_gammas = [1.5]    

    ### Start experiment loops
#    for multiply in range(5,130,20):
    for multiply in range(125,130,20):
        ### for first dimension always use 110% of number of available verb-obj-tuples
        z_liste = [int(1.1*args.n_used_tuples)] + [5] * multiply
        z_string = ",".join(str(entry) for entry in z_liste)
        for lr in learning_rates:
            for disc_gamma in disc_gammas:
                subprocess.call(["python", "main_joint_vae.py", 
                                "--batch_size",      str(args.batch_size), 
                                "--lr",              str(lr) , 
                                "--epoch_size",      str(args.epoch_size),  
                                "--repr_type",       args.repr_type, 
                                "--train_split",     str(args.train_split), 
                                "--n_used_tuples",   str(args.n_used_tuples), 
                                "--disc_gamma",      str(disc_gamma), 
                                "--z_dims_str",      z_string])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Experiments')

    ### Define properties and hyperparameters of experiments
    parser.add_argument('--repr_type', default='word2vec50d', type=str, help='use of one-hot or  word2vec embedding; options are <one-hot>,<word2vec300d>,<word2vec50d>')
    parser.add_argument('--train_split', default=0.75, type=float, help='what fraction of dataset for training purpose')
    parser.add_argument('--n_used_tuples', default=1100  , type=int,
                        help='how many (verb,object)-tuples should training and testing set contain (max. is 1100')
    parser.add_argument('--epoch_size', default=100, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=100)


    args = parser.parse_args()
    main(args)
