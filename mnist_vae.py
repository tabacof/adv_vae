# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 17:25:13 2016

@author: tabacof
"""

# Implements a variational autoencoder as described in Kingma et al. 2013
# "Auto-Encoding Variational Bayes"
import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import numpy as np
import lasagne
from parmesan.distributions import log_stdnormal, log_normal2, log_bernoulli, kl_normal2_stdnormal
from parmesan.layers import SimpleSampleLayer
from parmesan.datasets import load_mnist_realval, load_mnist_binarized
import time, shutil, os
import scipy
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import cPickle as cPkl

filename_script = os.path.basename(os.path.realpath(__file__))

#settings
do_train_model = True
model_filename = "mnist_model"
nplots = 5
dataset = 'fixed'
batch_size = 100
nhidden = 400
nonlin_enc = lasagne.nonlinearities.rectify
nonlin_dec = lasagne.nonlinearities.rectify
latent_size = 100
analytic_kl_term = True
lr = 0.0002
num_epochs = 50
results_out = os.path.join("results", os.path.splitext(filename_script)[0])

np.random.seed(1234) # reproducibility

# Setup outputfolder logfile etc.
if not os.path.exists(results_out):
    os.makedirs(results_out)
shutil.copy(os.path.realpath(__file__), os.path.join(results_out, filename_script))
logfile = os.path.join(results_out, 'logfile.log')

#SYMBOLIC VARS
sym_x = T.matrix()
sym_lr = T.scalar('lr')

#Helper functions
def bernoullisample(x):
    return np.random.binomial(1,x,size=x.shape).astype(theano.config.floatX)

### LOAD DATA
if dataset is 'sample':
    print "Using real valued MNIST dataset to binomial sample dataset after every epoch "
    train_x, train_t, valid_x, valid_t, test_x, test_t = load_mnist_realval()
    del train_t, valid_t, test_t
    preprocesses_dataset = bernoullisample
else:
    print "Using fixed binarized MNIST data"
    train_x, valid_x, test_x = load_mnist_binarized()
    preprocesses_dataset = lambda dataset: dataset #just a dummy function

#concatenate train and validation set
train_x = np.concatenate([train_x, valid_x])

train_x = train_x.astype(theano.config.floatX)
test_x = test_x.astype(theano.config.floatX)

nfeatures=train_x.shape[1]
n_train_batches = train_x.shape[0] / batch_size
n_test_batches = test_x.shape[0] / batch_size

#setup shared variables
sh_x_train = theano.shared(preprocesses_dataset(train_x), borrow=True)
sh_x_test = theano.shared(preprocesses_dataset(test_x), borrow=True)

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
l_dec_x_mu = lasagne.layers.DenseLayer(l_dec_h1, num_units=nfeatures, nonlinearity=lasagne.nonlinearities.sigmoid, name='DEC_X_MU')

# Get outputs from model
# with noise
z_train, z_mu_train, z_log_var_train, x_mu_train = lasagne.layers.get_output(
    [l_z, l_mu, l_log_var, l_dec_x_mu], sym_x, deterministic=False
)

# without noise
z_eval, z_mu_eval, z_log_var_eval, x_mu_eval = lasagne.layers.get_output(
    [l_z, l_mu, l_log_var, l_dec_x_mu], sym_x, deterministic=True
)


#Calculate the loglikelihood(x) = E_q[ log p(x|z) + log p(z) - log q(z|x)]
def latent_gaussian_x_bernoulli(z, z_mu, z_log_var, x_mu, x, analytic_kl_term):
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
    if analytic_kl_term:
        kl_term = kl_normal2_stdnormal(z_mu, z_log_var).sum(axis=1)
        log_px_given_z = log_bernoulli(x, x_mu, eps=1e-6).sum(axis=1)
        LL = T.mean(-kl_term + log_px_given_z)
    else:
        log_qz_given_x = log_normal2(z, z_mu, z_log_var).sum(axis=1)
        log_pz = log_stdnormal(z).sum(axis=1)
        log_px_given_z = log_bernoulli(x, x_mu, eps=1e-6).sum(axis=1)
        LL = T.mean(log_pz + log_px_given_z - log_qz_given_x)
    return LL

# TRAINING LogLikelihood
LL_train = latent_gaussian_x_bernoulli(
    z_train, z_mu_train, z_log_var_train, x_mu_train, sym_x, analytic_kl_term)

# EVAL LogLikelihood
LL_eval = latent_gaussian_x_bernoulli(
    z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, sym_x, analytic_kl_term)


params = lasagne.layers.get_all_params([l_dec_x_mu], trainable=True)
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
                                  givens={sym_x: sh_x_train[batch_slice], },)

test_model = theano.function([sym_batch_index], LL_eval,
                                  givens={sym_x: sh_x_test[batch_slice], },)
                                  
plot_results = theano.function([sym_batch_index], x_mu_eval,
                                  givens={sym_x: sh_x_test[batch_slice]},)

PARAM_EXTENSION = 'params'

def read_model_data(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('./', '%s.%s' % (filename, PARAM_EXTENSION))
    with open(filename, 'r') as f:
        data = cPkl.load(f)
    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    """Pickles the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, PARAM_EXTENSION)
    with open(filename, 'w') as f:
        cPkl.dump(data, f)

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
        sh_x_train.set_value(preprocesses_dataset(train_x))

        if nplots:
            np.random.shuffle(test_x)
            sh_x_test.set_value(preprocesses_dataset(test_x))
            
            results = plot_results(0)
            plt.figure(figsize=(3, nplots))
            for i in range(0,nplots):
                plt.subplot(nplots,2,(i+1)*2-1)
                plt.imshow(test_x[i].reshape(28,28), cmap='Greys_r')
                if i == 0:
                    plt.title("Input")
                plt.axis('off')
                plt.subplot(nplots,2,(i+1)*2)
                plt.imshow(results[i].reshape(28, 28), cmap='Greys_r')
                if i == 0:
                    plt.title("Output")
                plt.axis('off')
            plt.savefig(results_out+"/epoch_"+str(epoch)+".pdf", bbox_inches='tight')
                
        train_cost = train_epoch(lr)
        test_cost = test_epoch()

        t = time.time() - start

        line =  "*Epoch: %i\tTime: %0.2f\tLR: %0.5f\tLL Train: %0.3f\tLL test: %0.3f\t" % ( epoch, t, lr, train_cost, test_cost)
        print line
        with open(logfile,'a') as f:
            f.write(line + "\n")
    
    print "Write model data"
    write_model_data(l_dec_x_mu, model_filename)
else:
    print "Load model data"
    read_model_data(l_dec_x_mu, model_filename)
       
# Adversarial image generation for VAEs
def kld(mean1, log_var1, mean2, log_var2):
    mean_term = (T.exp(0.5*log_var1) + (mean1-mean2)**2.0)/T.exp(0.5*log_var2)
    return mean_term + log_var2 - log_var1 - 0.5

# Autoencoder outputs
mean, log_var, reconstruction = lasagne.layers.get_output(
    [l_mu, l_log_var, l_dec_x_mu], inputs = sym_x, deterministic=True)
    
# Adversarial confusion cost function
    
# Mean squared reconstruction difference
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

# Function used to optimize the adversarial noise
adv_function = theano.function([sym_x, adv_mean, adv_log_var, C], [adv_loss, adv_grad])

# Helper to plot reconstructions    
adv_plot = theano.function([sym_x], reconstruction)
    
def show_mnist(fig, img, i, title=""):
    ax = fig.add_subplot(3, 2, i)
    ax.imshow(img.copy().reshape(28,28), cmap='Greys_r')
    ax.set_title(title)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.grid(b=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax
    
def adv_test(orig_img = 0, target_img = 1, C = 1, plot = True):
    # Set the adversarial noise to zero
    l_noise.b.set_value(np.zeros(nfeatures).astype(np.float32))
    
    # Get latent variables of the target
    adv_mean_log_var = theano.function([sym_x], [mean, log_var])
    adv_mean_values, adv_log_var_values = adv_mean_log_var(test_x[target_img][np.newaxis, :])
        
    # Plot original image and its reconstruction
    if plot:
        fig = plt.figure(figsize=(6,10))
        ax = show_mnist(fig, test_x[orig_img], 1, "Input")
        ax.set_ylabel("Original")
        show_mnist(fig, adv_plot(test_x[orig_img][np.newaxis]), 2, "Output")

    # Initialize the adversarial noise for the optimization procedure
    l_noise.b.set_value(np.random.uniform(-1e-5, 1e-5, nfeatures).astype(np.float32))
    
    # Optimization function for L-BFGS-B
    def fmin_func(x):
        l_noise.b.set_value(x.astype(np.float32))
        f, g = adv_function(test_x[orig_img][np.newaxis], adv_mean_values.squeeze(), adv_log_var_values.squeeze(), C)
        return float(f), g.astype(np.float64)
        
    # Noise bounds (pixels cannot exceed 0-1)
    bounds = zip(-test_x[orig_img], 1-test_x[orig_img])
    # L-BFGS-B optimization to find adversarial noise
    x, f, d = scipy.optimize.fmin_l_bfgs_b(fmin_func, l_noise.b.get_value(), fprime = None, bounds = bounds, factr = 10, m = 25)
    
    adv_img = adv_plot(test_x[orig_img][np.newaxis])
    
    # Plotting results
    if plot:
        ax = show_mnist(fig, (test_x[orig_img].flatten()+x), 3)
        ax.set_ylabel("Adversarial")
        show_mnist(fig, adv_img, 4)
        show_mnist(fig, x, 5, "Adversarial noise")
        show_mnist(fig, test_x[target_img], 6, "Target image")
        plt.savefig(results_out+"/adv_"+str(orig_img)+"_"+str(target_img)+".pdf", bbox_inches='tight')
        plt.show()
        
    # Adversarial noise norm    
    return np.linalg.norm(x), np.linalg.norm(adv_img - test_x[target_img])

od, ad = adv_test(10, 1231, C=0.75, plot = True)
od, ad = adv_test(2440, 9231, C=0.75, plot = True)
od, ad = adv_test(5003, 6002, C=0.75, plot = True)