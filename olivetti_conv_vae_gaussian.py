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
from parmesan.distributions import log_normal2, kl_normal2_stdnormal
from parmesan.layers import SimpleSampleLayer
import time, shutil, os
import scipy
from scipy.io import loadmat
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from read_write_model import read_model, write_model
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cross_validation import train_test_split

filename_script = os.path.basename(os.path.realpath(__file__))

#settings
do_train_model = True
batch_size = 100
latent_size = 500
analytic_kl_term = True
lr = 0.0002
num_epochs = 100
model_filename = "olivetti_conv_vae"
nplots = 15

results_out = os.path.join("results", os.path.splitext(filename_script)[0])

np.random.seed(1234) # reproducibility

# Setup outputfolder logfile etc.
if not os.path.exists(results_out):
    os.makedirs(results_out)
shutil.copy(os.path.realpath(__file__), os.path.join(results_out, filename_script))
logfile = os.path.join(results_out, 'logfile.log')

#SYMBOLIC VARS
sym_x = T.tensor4()
sym_lr = T.scalar('lr')

### LOAD DATA
print "Using Olivetti dataset"

#svhn_train = loadmat('train_32x32.mat')
#svhn_test = loadmat('test_32x32.mat')
#
#train_x = np.rollaxis(svhn_train['X'], 3).transpose(0,3,1,2).astype(theano.config.floatX)
#test_x = np.rollaxis(svhn_test['X'], 3).transpose(0,3,1,2).astype(theano.config.floatX)
#
#mean = 115.11177966923525
#std = 50.819267906232888
#train_x = (train_x - mean)/std
#test_x = (test_x - mean)/std

train_aux = fetch_olivetti_faces()
train_x, test_x, train_y, test_y = train_test_split(train_aux.images,train_aux.target, test_size=0.20, random_state=42)
del train_aux
train_x = train_x * 255
test_x = test_x * 255
train_mean = np.mean(train_x)
std = np.std(train_x)
train_x = (train_x - train_mean)/std
test_x = (test_x - train_mean)/std

train_x = train_x.astype(theano.config.floatX)
train_x = train_x.reshape(-1,1, 64, 64)
test_x = test_x.astype(theano.config.floatX)
test_x = test_x.reshape(-1, 1, 64, 64)

if test_x.shape[0] < batch_size:
    batch_size = test_x.shape[0]
    print "Using batch size= " + str(batch_size)

n_train_batches = train_x.shape[0] / batch_size
n_test_batches = test_x.shape[0] / batch_size 

#setup shared variables
sh_x_train = theano.shared(train_x, borrow=True)
sh_x_test = theano.shared(test_x, borrow=True)

dim1, dim2, dim3 = 1, 64, 64

### RECOGNITION MODEL q(z|x)
l_in = lasagne.layers.InputLayer((batch_size, dim1, dim2, dim3))
l_noise = lasagne.layers.BiasLayer(l_in, b = np.zeros((dim1, dim2, dim3), dtype = np.float32), shared_axes = 0, name = "NOISE")
l_noise.params[l_noise.b].remove("trainable")
l_enc_h1 = lasagne.layers.Conv2DLayer(l_noise, num_filters = dim2, filter_size = 4, stride = 2, nonlinearity = lasagne.nonlinearities.elu, name = 'ENC_CONV1')
l_enc_h1 = lasagne.layers.Conv2DLayer(l_enc_h1, num_filters = 128, filter_size = 4, stride = 2, nonlinearity = lasagne.nonlinearities.elu, name = 'ENC_CONV2')
l_enc_h1 = lasagne.layers.Conv2DLayer(l_enc_h1, num_filters = 512, filter_size = 4, stride = 2, nonlinearity = lasagne.nonlinearities.elu, name = 'ENC_CONV3')
l_enc_h1 = lasagne.layers.DenseLayer(l_enc_h1, num_units=1024, nonlinearity=lasagne.nonlinearities.elu, name='ENC_DENSE2')

l_mu = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_Z_MU')
l_log_var = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_Z_LOG_VAR')

#sample the latent variables using mu(x) and log(sigma^2(x))
l_z = SimpleSampleLayer(mean=l_mu, log_var=l_log_var)

### GENERATIVE MODEL p(x|z)
l_dec_h1 = lasagne.layers.DenseLayer(l_z, num_units=1024, nonlinearity=lasagne.nonlinearities.elu, name='DEC_DENSE1')
l_dec_h1 = lasagne.layers.ReshapeLayer(l_dec_h1, (batch_size, -1, 4, 4))
l_dec_h1 = lasagne.layers.TransposedConv2DLayer(l_dec_h1, num_filters = 512, crop="same", filter_size = 5, stride = 2, nonlinearity = lasagne.nonlinearities.elu, name = 'DEC_CONV1')
l_dec_h1 = lasagne.layers.TransposedConv2DLayer(l_dec_h1, num_filters = 128, crop="same", filter_size = 5, stride = 2, nonlinearity = lasagne.nonlinearities.elu, name = 'DEC_CONV2')
l_dec_h1 = lasagne.layers.TransposedConv2DLayer(l_dec_h1, num_filters = dim2, filter_size = 5, stride = 2, nonlinearity = lasagne.nonlinearities.elu, name = 'DEC_CONV3')
l_dec_x_mu = lasagne.layers.TransposedConv2DLayer(l_dec_h1, num_filters = 1,filter_size = 36, nonlinearity = lasagne.nonlinearities.identity, name = 'DEC_MU')
l_dec_x_mu = lasagne.layers.ReshapeLayer(l_dec_x_mu, (batch_size, -1))
l_dec_x_log_var = lasagne.layers.TransposedConv2DLayer(l_dec_h1, num_filters = 1,filter_size = 36, nonlinearity = lasagne.nonlinearities.identity, name = 'DEC_LOG_VAR')
l_dec_x_log_var = lasagne.layers.ReshapeLayer(l_dec_x_log_var, (batch_size, -1))

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
def ELBO(z, z_mu, z_log_var, x_mu, x_log_var, x):
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
    log_px_given_z = log_normal2(x.reshape((batch_size, -1)), x_mu, x_log_var).sum(axis=1)
    LL = T.mean(-kl_term + log_px_given_z)

    return LL

# TRAINING LogLikelihood
LL_train = ELBO(z_train, z_mu_train, z_log_var_train, x_mu_train, x_log_var_train, sym_x)

# EVAL LogLikelihood
LL_eval = ELBO(z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, x_log_var_eval, sym_x)


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
        
plot_results = theano.function([sym_batch_index], x_mu_eval,
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

        results = plot_results(0)
        plt.figure(figsize=(2, nplots))
        
        for i in range(0,nplots):
            plt.subplot(nplots,2,(i+1)*2-1)
            plt.imshow((std*test_x[i].reshape(64,64)+train_mean)/255.0, cmap='gray')
            plt.axis('off')
            plt.subplot(nplots,2,(i+1)*2)
            plt.imshow((std*results[i].reshape(64,64)+train_mean)/255.0, cmap='gray')
            plt.axis('off')
            
        plt.savefig(results_out+"/epoch_"+str(epoch)+".pdf", bbox_inches='tight')
        plt.close()
        
        train_cost = train_epoch(lr)
        test_cost = test_epoch()

        t = time.time() - start

        line =  "*Epoch: %i\tTime: %0.2f\tLR: %0.5f\tLL Train: %0.3f\tLL test: %0.3f\t" % ( epoch, t, lr, train_cost, test_cost)
        print line
        with open(logfile,'a') as f:
            f.write(line + "\n")
    
    print "Write model data"
    write_model([l_dec_x_mu, l_dec_x_log_var], model_filename)
else:
    read_model(l_dec_x_mu, model_filename)
    
def kld(mean1, log_var1, mean2, log_var2):
    mean_term = (T.exp(0.5*log_var1) + (mean1-mean2)**2.0)/T.exp(0.5*log_var2)
    return mean_term + log_var2 - log_var1 - 0.5

# Original and target images
orig_img = 13
target_img = 1

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

# Plot original reconstruction
adv_plot = theano.function([sym_x], reconstruction)


def show_img(img, i, title=""): # expects flattened image of shape (3072,) 
    if dim1>1:
        img = img.copy().reshape(dim1,dim2,dim3).transpose(1,2,0)
    else:
        img = img.copy().reshape(dim2,dim3)
    img *= std
    img += train_mean
    img /= 255.0
    plt.subplot(3, 2, i)
    if dim1>1:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis("off")

def adv_test(orig_img = 0, target_img = 1, C = 200.0):
    # Set the adversarial noise to zero
    l_noise.b.set_value(np.zeros((dim1,dim2,dim3)).astype(np.float32))
    
    plt.figure(figsize=(10,10))
    # Get latent variables of the target
    adv_mean_log_var = theano.function([sym_x], [mean, log_var])
    adv_mean_values, adv_log_var_values = adv_mean_log_var(np.tile(test_x[target_img], (batch_size, 1, 1, 1)).reshape(batch_size, dim1,dim2,dim3))
    adv_mean_values = adv_mean_values[0]
    adv_log_var_values = adv_log_var_values[0]

    # Plot original reconstruction    
    show_img(test_x[orig_img], 1, "Original image")
    show_img(adv_plot(np.tile(test_x[orig_img], (batch_size, 1, 1, 1)).reshape(batch_size, dim1,dim2,dim3))[0], 2, "Original reconstruction")

    # Initialize the adversarial noise for the optimization procedure
    l_noise.b.set_value(np.random.uniform(-1e-3, 1e-3, size=(dim1,dim2,dim3)).astype(np.float32))
    
    # Optimization function for L-BFGS-B
    def fmin_func(x):
        l_noise.b.set_value(x.reshape(dim1,dim2,dim3).astype(np.float32))
        f, g = adv_function(np.tile(test_x[orig_img], (batch_size, 1, 1, 1)).reshape(batch_size, dim1,dim2,dim3), adv_mean_values, adv_log_var_values, C)
        return float(f), g.flatten().astype(np.float64)
        
    # Noise bounds (pixels cannot exceed 0-1)
    bounds = zip(-train_mean/std-test_x[orig_img].flatten(), (255.0-train_mean)/std-test_x[orig_img].flatten())
    #bounds = zip(-test_x[orig_img].flatten(), 1-test_x[orig_img].flatten())
    
    # L-BFGS-B optimization to find adversarial noise
    x, f, d = scipy.optimize.fmin_l_bfgs_b(fmin_func, l_noise.b.get_value().flatten(), bounds = bounds, fprime = None, factr = 10, m = 25)
    
    # Plotting results
    show_img(x, 3, "Adversarial noise")
    show_img(test_x[target_img], 4, "Target image")
    show_img((test_x[orig_img].flatten()+x), 5, "Adversarial image")
    show_img(adv_plot(np.tile(test_x[orig_img], (batch_size, 1, 1, 1)).reshape(batch_size, dim1,dim2,dim3))[0], 6, "Adversarial reconstruction")

    plt.show()

    # Adversarial noise norm
    print("Adversarial distortion norm", (x**2.0).sum())
    
adv_test()
