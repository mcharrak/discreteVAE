# Deep Learning for Natural Language Processing (NLP) using Variational Autoencoders (VAE)

This repository contains the code and datasets used for the thesis "Deep Learning for Natural Language Processing (NLP) using Variational Autoencoders (VAE)"
which can be found online here: <https://pub.tik.ee.ethz.ch/students/2018-FS/MA-2018-22.pdf>.

The goal of this project is to:

* create a dataset similar to dSprites dataset (<https://github.com/deepmind/dsprites-dataset>) where the factors of variation are knonwn but in this case for the natural language domain instead of graphics/image domain
* develop deep generative models using various deep learning architectures (MLP, CNN, RNN) as feature extractors for encoding and decoding within the variational encoder (VAE) and autoencoder (AE) framework
* learn disentangled and interpretable natural language representations using latent variable modles (specifically VAEs)


In this project we create our own dataset from scratch based on verb-object pairs (see spreadsheet) with factors of variation such as grammar (grammatical number object, gender, grammatical number subject, verb tense, and style), sentence structure (sentence type), and syntax (verb-object tuple and sentence negation). The Cartesian product of all possible combinations of our generative factor values results in a dataset size of N = 576,000.

We develop various deep generative model pipelines using the discrete versions of VAE/beta-VAE/AE frameworks by employing the Gumbel-Softmax trick with various values for beta e.g. beta = 0 for AE, beta = 1 for VAE, and beta-VAE for all other non-negative beta values.
We use RNNs, CNNs, and MLPs as encoder respectively decoder functions. Further, we implement various disentanglement metric measures such as variance based disentanglement metric (<https://arxiv.org/pdf/1802.05983.pdf>) and mutual information based disentanglement metric (<https://arxiv.org/pdf/1802.04942.pdf>).
