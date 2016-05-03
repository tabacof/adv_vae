# -*- coding: utf-8 -*-
"""
Created on Tue May 03 11:12:17 2016

"""
import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import numpy as np
import lasagne
from parmesan.distributions import log_normal2, kl_normal2_stdnormal
from parmesan.layers import SimpleSampleLayer
from parmesan.datasets import load_cifar10, load_mnist_realval
import time, shutil, os
import scipy
import pylab as plt
from read_write_model import *
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cross_validation import train_test_split

filename_script = os.path.basename(os.path.realpath(__file__))

class Dataset:
    Cifar, Mnist, Olivetti = range(3)

#settings
dataset = Dataset.Cifar
do_train_model = True
batch_size = 100
nhidden = 500
nonlin_enc = T.nnet.softplus
nonlin_dec = T.nnet.softplus
latent_size = 100
analytic_kl_term = True
lr = 0.0003
num_epochs = 1
results_out = os.path.join("results", os.path.splitext(filename_script)[0])

# Original and target images
orig_img = 13
target_img = 1

np.random.seed(1234) # reproducibility

if dataset == Dataset.Cifar:
    model_filename = "cifar_model_gaussian"
elif dataset == Dataset.Mnist:
    model_filename = "mnist_model_gaussian"
elif dataset == Dataset.Olivetti:
    model_filename = "olivetti_model_gaussian"
else:
    raise Exception('Unsupported dataset')

# Setup outputfolder logfile etc.
if not os.path.exists(results_out):
    os.makedirs(results_out)
shutil.copy(os.path.realpath(__file__), os.path.join(results_out, filename_script))
logfile = os.path.join(results_out, 'logfile.log')

#SYMBOLIC VARS
sym_x = T.matrix()
sym_x_out = T.matrix()
sym_lr = T.scalar('lr')

#Helper functions
def bernoullisample(x):
    return np.random.binomial(1,x,size=x.shape).astype(theano.config.floatX)

### LOAD DATA

if dataset == Dataset.Cifar:
    print "Using CIFAR10 dataset"
    train_x, train_y, test_x, test_y = load_cifar10(normalize=False,dequantify=False)
    train_x = train_x.reshape(train_x.shape[0],-1)  # reshape so RGB data is all in one dimension
    test_x = test_x.reshape(test_x.shape[0],-1)
    mean = 120.7076 # Calculated using training set
    std = 64.150093 # Calculated using training set
    del train_y, test_y
elif dataset == Dataset.Mnist:
    print "Using real valued MNIST dataset"
    train_x, train_t, valid_x, valid_t, test_x, test_t = load_mnist_realval()
    del train_t, valid_t, test_t    
    train_x = np.concatenate([train_x, valid_x])
    mean = np.mean(train_x).astype(theano.config.floatX)
    std = np.std(train_x).astype(theano.config.floatX)
elif dataset == Dataset.Olivetti:
    train_aux = fetch_olivetti_faces()
    train_x, test_x, train_y, test_y = train_test_split(train_aux.images,train_aux.target, test_size=0.20, random_state=1234)
    del train_aux, train_y, test_y
    mean = np.mean(train_x).astype(theano.config.floatX)
    std = np.std(train_x).astype(theano.config.floatX)
else:
    raise Exception('Unsupported dataset')

train_x = (train_x - mean)/std
test_x = (test_x - mean)/std
    
train_x = train_x.astype(theano.config.floatX)
test_x = test_x.astype(theano.config.floatX)

nfeatures=train_x.shape[1]
n_train_batches = train_x.shape[0] / batch_size
n_test_batches = test_x.shape[0] / batch_size

#setup shared variables
sh_x_train = theano.shared(train_x, borrow=True)
sh_x_test = theano.shared(test_x, borrow=True)

### RECOGNITION MODEL q(z|x)
l_in = lasagne.layers.InputLayer((batch_size, nfeatures))
l_noise = lasagne.layers.BiasLayer(l_in, b = np.zeros(nfeatures, dtype = np.float32), name = "NOISE")
l_noise.params[l_noise.b].remove("trainable")
l_enc_h1 = lasagne.layers.DenseLayer(l_noise, num_units=nhidden, nonlinearity=nonlin_enc, name='ENC_DENSE1')
l_enc_h1 = lasagne.layers.DenseLayer(l_enc_h1, num_units=nhidden, nonlinearity=nonlin_enc, name='ENC_DENSE2')

l_mu = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_Z_MU')
l_log_var = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_Z_LOG_VAR')

#sample the latent variables using mu(x) and log(sigma^2(x))
l_z = SimpleSampleLayer(mean=l_mu, log_var=l_log_var)

### GENERATIVE MODEL p(x|z)
l_dec_h1 = lasagne.layers.DenseLayer(l_z, num_units=nhidden, nonlinearity=nonlin_dec, name='DEC_DENSE2')
l_dec_h1 = lasagne.layers.DenseLayer(l_dec_h1, num_units=nhidden, nonlinearity=nonlin_dec, name='DEC_DENSE1')
l_dec_x_mu = lasagne.layers.DenseLayer(l_dec_h1, num_units=nfeatures, nonlinearity=nonlin_dec, name='DEC_X_MU')
l_dec_x_log_var = lasagne.layers.DenseLayer(l_dec_h1, num_units=nfeatures, nonlinearity=nonlin_dec, name='DEC_X_LOG_VAR')

# Get outputs from model
# with noise
z_train, z_mu_train, z_log_var_train, x_mu_train, x_log_var_train = lasagne.layers.get_output(
    [l_z, l_mu, l_log_var, l_dec_x_mu, l_dec_x_log_var], sym_x, deterministic=False
)

# without noise
z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, x_log_var_eval = lasagne.layers.get_output(
    [l_z, l_mu, l_log_var, l_dec_x_mu, l_dec_x_log_var], sym_x, deterministic=True
)

#Calculate the loglikelihood(x) = E_q[ log p(x|z) + log p(z) - log q(z|x)]
def latent_gaussian_x_bernoulli(z, z_mu, z_log_var, x_mu, x_log_var, x):
    """
    Latent z       : gaussian with standard normal prior
    decoder output : bernoulli

    When the output is bernoulli then the output from the decoder
    should be sigmoid. The sizes of the inputs are
    z: (batch_size, num_latent)
    z_mu: (batch_size, num_latent)
    z_log_var: (batch_size, num_latent)
    x_mu: (batch_size, num_features)
    x: (batch_size, num_features)
    """
    kl_term = kl_normal2_stdnormal(z_mu, z_log_var).sum(axis=1)
    log_px_given_z = log_normal2(x, x_mu, x_log_var).sum()
    LL = T.mean(-kl_term + log_px_given_z)

    return LL

# TRAINING LogLikelihood
LL_train = latent_gaussian_x_bernoulli(
    z_train, z_mu_train, z_log_var_train, x_mu_train, x_log_var_train, sym_x)

# EVAL LogLikelihood
LL_eval = latent_gaussian_x_bernoulli(
    z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, x_log_var_eval, sym_x)


params = lasagne.layers.get_all_params([l_dec_x_mu, l_dec_x_log_var], trainable=True)
for p in params:
    print p, p.get_value().shape

### Take gradient of Negative LogLikelihood
grads = T.grad(-LL_train, params)

# Add gradclipping to reduce the effects of exploding gradients.
# This speeds up convergence
clip_grad = 1
max_norm = 5
mgrads = lasagne.updates.total_norm_constraint(grads,max_norm=max_norm)
cgrads = [T.clip(g,-clip_grad, clip_grad) for g in mgrads]

#Setup the theano functions
sym_batch_index = T.iscalar('index')
batch_slice = slice(sym_batch_index * batch_size, (sym_batch_index + 1) * batch_size)

updates = lasagne.updates.adam(cgrads, params, learning_rate=sym_lr)

train_model = theano.function([sym_batch_index, sym_lr], LL_train, updates=updates,
                                  givens={sym_x: sh_x_train[batch_slice]},)

test_model = theano.function([sym_batch_index], LL_eval,
                                  givens={sym_x: sh_x_test[batch_slice]},)

def train_epoch(lr):
    costs = []
    for i in range(n_train_batches):
        cost_batch = train_model(i, lr)
        costs += [cost_batch]
    return np.mean(costs)

def test_epoch():
    costs = []
    for i in range(n_test_batches):
        cost_batch = test_model(i)
        costs += [cost_batch]
    return np.mean(costs)

if do_train_model:
    # Training Loop
    for epoch in range(num_epochs):
        start = time.time()

        #shuffle train data, train model and test model
        np.random.shuffle(train_x)
        sh_x_train.set_value(train_x)

        train_cost = train_epoch(lr)
        test_cost = test_epoch()

        t = time.time() - start

        line =  "*Epoch: %i\tTime: %0.2f\tLR: %0.5f\tLL Train: %0.3f\tLL test: %0.3f\t" % ( epoch, t, lr, train_cost, test_cost)
        print line
        with open(logfile,'a') as f:
            f.write(line + "\n")
    
    print "Write model data"
    write_model(l_dec_x_mu, model_filename)
else:
    read_model(l_dec_x_mu, model_filename)
    
def kld(mean1, log_var1, mean2, log_var2):
    mean_term = (T.exp(0.5*log_var1) + (mean1-mean2)**2.0)/T.exp(0.5*log_var2)
    return mean_term + log_var2 - log_var1 - 0.5


# Autoencoder outputs
mean, log_var, reconstruction = lasagne.layers.get_output(
    [l_mu, l_log_var, l_dec_x_mu], inputs = sym_x, deterministic=True)
    
# Adversarial confusion cost function
    
# Mean squared reconstruction difference
#adv_target  = T.vector()
#adv_confusion = lasagne.objectives.squared_error(reconstruction, adv_target).sum()
# KL divergence between latent variables
adv_mean =  T.vector()
adv_log_var = T.vector()
adv_confusion = kld(mean, log_var, adv_mean, adv_log_var).sum()

# Adversarial regularization
C = T.scalar()
adv_reg = C*lasagne.regularization.l2(l_noise.b)
# Total adversarial loss
adv_loss = adv_confusion + adv_reg
adv_grad = T.grad(adv_loss, l_noise.b)
#adv_function = theano.function([sym_x, adv_target, C], [adv_loss, adv_grad])

# Function used to optimize the adversarial noise
adv_function = theano.function([sym_x, adv_mean, adv_log_var, C], [adv_loss, adv_grad])

# Set the adversarial noise to zero
l_noise.b.set_value(np.zeros(nfeatures).astype(np.float32))

# Get latent variables of the target
adv_mean_log_var = theano.function([sym_x], [mean, log_var])
adv_mean_values, adv_log_var_values = adv_mean_log_var(train_x[target_img][np.newaxis, :])
# Plot original reconstruction
adv_plot = theano.function([sym_x], reconstruction)

def show_image(img, title=""): # expects flattened image 
    if dataset == Dataset.Cifar:
        plt.figure(figsize=(0.5,0.5))
        img = img.reshape(32,32,3)
        plt.imshow(img)
    elif dataset == Dataset.Mnist:
        img = img.reshape(28,28)
        plt.imshow(img, cmap='gray')
    elif dataset == Dataset.Olivetti:
        img = img.reshape(64,64)
        plt.imshow(img, cmap='gray')
    else:
        raise Exception('Unsupported dataset')
    
    img *= std
    img += mean
    plt.title(title)
    plt.axis("off")
    plt.show()

#show_image(train_x[orig_img])
#show_image(adv_plot(train_x[orig_img][np.newaxis, :]))

# Initialize the adversarial noise for the optimization procedure
l_noise.b.set_value(np.random.uniform(-1e-5, 1e-5, nfeatures).astype(np.float32))

# Optimization function for L-BFGS-B
def fmin_func(x):
    l_noise.b.set_value(x.astype(np.float32))
    #f, g = adv_function(train_x[orig_img][np.newaxis,:], train_x[target_img], 1.0)
    f, g = adv_function(train_x[orig_img][np.newaxis,:], adv_mean_values.squeeze(), adv_log_var_values.squeeze(), 1.0)
    return float(f), g.astype(np.float64)
    
# Noise bounds (pixels cannot exceed 0-1)
bounds = zip(-train_x[orig_img], 1-train_x[orig_img])
# L-BFGS-B optimization to find adversarial noise
x, f, d = scipy.optimize.fmin_l_bfgs_b(fmin_func, l_noise.b.get_value(), fprime = None, bounds = bounds, factr = 10, m = 25)

# Plotting results
show_image(train_x[orig_img], "Original image")
show_image(train_x[target_img] , "Target image")
show_image(x, "Adversarial noise")
show_image((train_x[orig_img]+x), "Adversarial image")
show_image(adv_plot(train_x[orig_img][np.newaxis, :]), "Reconstructed adversarial image")

# Adversarial noise norm
print((x**2.0).sum())
