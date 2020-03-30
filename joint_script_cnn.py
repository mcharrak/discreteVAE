import argparse
import subprocess
import numpy as np

def main(args):
    ### Create learning_rates list
    
    learning_rates = [5.0e-5]
    ### Create disc_gammas list
#    disc_gammas = [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    disc_gammas = [5.0, 10.0]

    ### Start experiment loops
    for lr in learning_rates:
        for disc_gamma in disc_gammas:
            subprocess.call(["python", "main_joint_vae.py", 
                             "--batch_size",      str(args.batch_size), 
                             "--lr",              str(lr) , 
                             "--epoch_size",      str(args.epoch_size),  
                             "--repr_type",       args.repr_type,
                             "--enc_type",        args.enc_type,
                             "--dec_type",        args.dec_type, 
                             "--train_split",     str(args.train_split), 
                             "--n_used_tuples",   str(args.n_used_tuples), 
                             "--disc_gamma",      str(disc_gamma)]) 
                            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Experiments')

    ### Define properties and hyperparameters of experiments
    parser.add_argument('--enc_type', default='cnn')
    parser.add_argument('--dec_type', default='cnn')
    parser.add_argument('--repr_type', default='word2vec50d', type=str, help='use of one-hot or  word2vec embedding; options are <one-hot>,<word2vec300d>,<word2vec50d>')
    parser.add_argument('--train_split', default=0.75, type=float, help='what fraction of dataset for training purpose')
    parser.add_argument('--n_used_tuples', default=1100  , type=int,
                        help='how many (verb,object)-tuples should training and testing set contain (max. is 1100')
    parser.add_argument('--epoch_size', default=100, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=100)


    args = parser.parse_args()
    main(args)
