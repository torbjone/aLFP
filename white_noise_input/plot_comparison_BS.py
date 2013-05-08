#!/usr/bin/env python
import os
if not os.environ.has_key('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
import LFPy
import numpy as np
import neuron
import sys
try:
    from ipdb import set_trace
except:
    pass
import pylab as pl
from os.path import join

import aLFP
from params import *

model = 'ball_n_stick' 
domain = 'white_noise_%s' %model

np.random.seed(1234)

input_idxs = [0, 12, 31]

input_scalings = [0.001, 0.01, 0.1]

plot_params = {'ymax': 1000,
               'ymin': 0,
               'yticks': [0, 250, 500, 750, 1000],
               }
#aLFP.compare_active_passive(model, input_scalings[0] , input_idxs[1], 
#                            elec_x, elec_y, elec_z, plot_params)

#sys.exit()
for input_idx in input_idxs:
    for input_scaling in input_scalings:
        print input_idx, input_scaling
        aLFP.compare_active_passive(model, input_scaling , input_idx, 
                                    elec_x, elec_y, elec_z, plot_params)
