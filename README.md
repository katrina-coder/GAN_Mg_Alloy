# GAN_Mg_Alloy
Generative Adversarial Networks (GANs)


This repository contains a GAN model trained for magnesium (Mg) alloy dataset.


GANs consist of a generator and a discriminator neural network. These networks operate through a simultaneous learning process of the complex high-dimensional probability distribution of input data, with one model (the generator) focused on generating a random search space and the other model (the discriminator) committed to differentiate between generated samples and real data. 


To replace any other alloy dataset, you need to define "CHEMICALS" and "CATEGORICALS" variables for the normalize_chemicals and normalize_categoricals functions.


To do the alloy inverse design, you need to narrow down the generated samples with pre-defined target properties.
