# Adversarial Images for Variational Autoencoders

To be presented at the Adversarial Training Workshop at NIPS 2016, Barcelona.

Arxiv link will be posted in the near future!

Please cite our work:
> Pedro Tabacof, Julia Tavares, and Eduardo Valle. Adversarial Images for Variational Autoencoders. Adversarial Training Workshop, NIPS. 2016.

## Requirements

[Theano](http://deeplearning.net/software/theano/)

[Lasagne](https://github.com/Lasagne/Lasagne)

[Parmesan](https://github.com/casperkaae/parmesan)

## Notebooks

To reproduce our experiments, simply run the notebooks. 

There are some options that can be readily changed, the most important one being ```do_train_model```: Set it to True to train the model from scratch, or to False to use the pretrained models in the params folder.

## Files

### Experiments

adv: Adversarial images for (variational) autoencoders

clf: Adversarial images for classifier experiments

### Architectures

ae: Deterministic autoencoders

vae: Variational autoencoders

### Datasets

mnist: MNIST dataset

svhn: SVHN dataset

### Pretrained weights

params: Pretrained AEs, VAEs and CLFs

### Results

results: folder with CSVs containing the experiments results -- to be used for plotting

results.ipynb: Plotting results from the folder above

