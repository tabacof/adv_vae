# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:45:41 2016

@author: Julia
"""
import os
import lasagne
try:
   import cPickle as pkl
except:
   import pickle as pkl

def read_model(model, filename, extension='params'):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('./params/', '%s.%s' % (filename, extension))
    with open(filename, 'r') as f:
        data = pkl.load(f)
    lasagne.layers.set_all_param_values(model, data)

def write_model(model, filename, extension='params', protocol=2):
    """Pickles the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./params/', filename)
    filename = '%s.%s' % (filename, extension)
    with open(filename, 'w') as f:
        pkl.dump(data, f, protocol)
