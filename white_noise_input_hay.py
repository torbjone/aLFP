#!/usr/bin/env python
import os
if not os.environ.has_key('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
try:
    from ipdb import set_trace
except:
    pass
import LFPy
import numpy as np
import neuron
import sys
import pylab as pl
from os.path import join
import cPickle
import aLFP

pl.rcParams.update({'font.size' : 8,
                    'figure.facecolor' : '1',
                    'wspace' : 0.5, 'hspace' : 0.5})
np.random.seed(1234)

model = 'hay'
domain = 'white_noise_%s' %model
neuron_model = join('neuron_models', model)
tstartms = 0
tstopms = 1000
timeres = 2**-5
srate = 1/timeres * 1000

neur_input_params = {'input_place': 'somatic',
                     'input_idx': 0,
                     'input_scaling': 0.001, # Scale input amplitude to fit model
                     }

plotting_params = {'cell_extent': [-600, 600, -400, 1300],
                   }

elec_x = np.array([-100, 0, 100, 0, 0])
elec_y = np.array([-100, 0, 400, 800, 1100])
elec_z = np.zeros(len(elec_x))

neural_sim_dict = {'domain': domain,
                   'model': model,
                   'is_active': True,
                   'load': 0,
                   'output_folder': join(domain, 'neural_sim'),
                   'timeres': timeres,
                   'timeres': timeres,
                   'tstartms': tstartms,
                   'tstopms': tstopms,
                   'ntsteps': round((tstopms) / timeres + 1),
                   'cut_off': 2000, # How many ms to remove from start
                   'elec_x': elec_x,
                   'elec_y': elec_y,
                   'elec_z': elec_z,
                   }
try:
    os.mkdir(domain)
except OSError:
    pass

cell, syn, electrode = aLFP.run_simulation(neural_sim_dict, neur_input_params)
aLFP.plot_all_currents(cell, syn, electrode, neural_sim_dict, plotting_params, neur_input_params)
#pl.show()
#set_trace()
