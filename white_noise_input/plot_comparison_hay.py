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

model = 'hay' 
domain = 'white_noise_%s' %model

np.random.seed(1234)

input_idxs = [0, 791, 611, 808, 681, 740, 606]

input_scalings = [0.001, 0.01, 0.1, 1.0]

plot_params = {'ymax': 1250,
               'ymin': -250,
#'yticks': [-250, 0, 250, 500, 750, 1000, 1250],
               }


#aLFP.compare_active_passive(model, input_scalings[0] , input_idxs[1], elec_x, elec_y, elec_z, plot_params)
#sys.exit()
for input_idx in input_idxs:
    for input_scaling in input_scalings:
        print input_idx, input_scaling
        aLFP.compare_active_passive(model, input_scaling , input_idx, 
                                    elec_x, elec_y, elec_z, plot_params)


