#!/usr/bin/env python

import LFPy
import numpy as np
import neuron
import os
import sys
from ipdb import set_trace
import pylab as pl
from os.path import join

import aLFP
from params import *


model = 'ball_n_stick' 
domain = 'white_noise_%s' %model

np.random.seed(1234)

input_idxs = [0, 12]

input_scalings = [0.001, 0.01, 0.1]


aLFP.compare_active_passive(model, input_scalings[0] , input_idxs[1], elec_x, elec_y, elec_z)
