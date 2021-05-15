# Deep Learning for Natural Language Processing (NLP) using Variational Autoencoders (VAE)

This repository contains the code and datasets used for the thesis "Deep Learning for Natural Language Processing (NLP) using Variational Autoencoders (VAE)"
which can be found online here: <https://pub.tik.ee.ethz.ch/students/2018-FS/MA-2018-22.pdf>.

The goal of this project is to:

* create a new dataset (similar to dSprites dataset (<https://github.com/deepmind/dsprites-dataset>)) where the factors of variation are knonwn but for the natural language domain instead of graphics/image domain, we coin this dataset as dSentences
* develop deep generative models using various deep learning architectures (MLP, CNN, RNN) as feature extractors for encoder and decoder in the variational autoencoder (VAE) and autoencoder (AE) framework
* learn disentangled and interpretable natural language text representations using latent variable modles (especially VAEs)


In this project we create a new custom disentanglement dataset for NLP from scratch based on verb-object pairs (see spreadsheet) with factors of variation being: grammar (grammatical number object, gender, grammatical number subject, verb tense, and style), sentence structure (sentence type), and syntax (verb-object tuple and sentence negation). The Cartesian product of all possible combinations of these generative factor values results in a dataset size of N = 576,000 samples.

We develop deep generative model pipelines using the discrete latent space versions of VAE/beta-VAE/AE frameworks by employing the Gumbel-Softmax trick and different values for beta (e.g., beta = 0 for the AE and beta = 1 for the VAE); and beta-VAE for all non-negative beta.
We use RNNs, CNNs, and MLPs as encoder respectively decoder functions. Further, we implement various disentanglement metric measures such as variance based disentanglement metric (<https://arxiv.org/pdf/1802.05983.pdf>) and mutual information based disentanglement metric (<https://arxiv.org/pdf/1802.04942.pdf>).
