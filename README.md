# GAN_Mg_Alloy
Generative Adversarial Networks (GANs)

This repository contains a Generative Adversarial Network (GAN) model trained upon a magnesium (Mg) alloy dataset.
The purpose of this repository is to provide the alloy design and materials community a useful tool for ‘inverse design’ of Mg-alloys. Inverse design 
GANs consist of a generator and a discriminator neural network. These networks operate through a simultaneous learning process from the complex high-dimensional probability distribution of input data, with one model (the generator) focused on generating a random search space and the other model (the discriminator) committed to differentiate between generated samples and real data.
In using the model, to replace any other alloy dataset, one needs to define "CHEMICALS" and "CATEGORICALS" variables,for the normalize_chemicals and normalize_categoricals functions.
To carry out inverse design of Mg-alloys, one is required to ‘narrow down’ generated samples with pre-defined target properties.

The utilisation of the inverse design Mg-alloy GAN model requires some proficiency in Python, along with calling upon data in the root folder (https://github.com/katrina-coder/Magnesium-alloys-database).

In order to access our other machine learning tools relevant to Mg-alloys (that are user friendly GUI based tools, that require no coding experience), you may wish to also explore:

Forward feed Mg-alloy property predictor:

https://colab.research.google.com/drive/1sp4_YP7qtUc2S-GO7BJ299x8Atux9-3z#scrollTo=eWerj79uRPyX


Mg-alloy predictor based on Bayesian Optimisation:

https://colab.research.google.com/drive/1wR0bQnxdAVUurH879dQNZ5wVzc0jGeD1#scrollTo=B5qe8hjnEyxe
