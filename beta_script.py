import argparse
import subprocess
import numpy as np

def main(args):
    ### Create learning_rates list
    
    learning_rates = [5.0e-5]
    ### Create betas list
    betas = [7.5, 12.5, 15.0, 17.5, 20.0]#, 25.0, 30.0, 50.0]




    ### Start experiment loops
    for lr in learning_rates:
        for beta in betas:
            subprocess.call(["python", "main_beta_vae.py",
                             "--batch_size",      str(args.batch_size),
                             "--lr",              str(lr) ,
                             "--epoch_size",      str(args.epoch_size),
                             "--repr_type",       args.repr_type,
                             "--enc_type",        args.enc_type,
                             "--dec_type",        args.dec_type,
                             "--train_split",     str(args.train_split),
                             "--n_used_tuples",   str(args.n_used_tuples),
                             "--beta", str(beta)])



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Beta Experiments')

    ### Define properties and hyperparameters of experiments
    parser.add_argument('--enc_type', default='mlp')
    parser.add_argument('--dec_type', default='mlp')
    parser.add_argument('--repr_type', default='word2vec50d', type=str, help='use of one-hot or  word2vec embedding; options are <one-hot>,<word2vec300d>,<word2vec50d>')
    parser.add_argument('--train_split', default=0.75, type=float, help='what fraction of dataset for training purpose')
    parser.add_argument('--n_used_tuples', default=1100  , type=int,
                        help='how many (verb,object)-tuples should training and testing set contain (max. is 1100')
    parser.add_argument('--epoch_size', default=100, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=100)

    args = parser.parse_args()
    main(args)
