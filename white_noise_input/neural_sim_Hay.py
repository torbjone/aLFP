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

model = 'hay' 
domain = 'white_noise_%s' %model

np.random.seed(1234)
neuron_model = join('..', 'neuron_models', model)
model_path = join(neuron_model, 'lfpy_version')
LFPy.cell.neuron.load_mechanisms(join(neuron_model, 'mod'))      
LFPy.cell.neuron.load_mechanisms(join('..', 'neuron_models'))      
cut_off = 300
is_active = True
input_idxs = [0, 650]

input_scalings = [0.001, 0.01, 0.1]

rot_params = {'x': -np.pi/2, 
              'y': 0, 
              'z': 0
              }

pos_params = {'xpos': 0, 
              'ypos': 0,
              'zpos': 0,
              }        

if is_active:
    conductance = 'active'
else:
    conductance = 'passive'

cell_params = {
    'morphology' : join(model_path, 'morphologies', 'cell1.hoc'),
    #'rm' : 30000,               # membrane resistance
    #'cm' : 1.0,                 # membrane capacitance
    #'Ra' : 100,                 # axial resistance
    'v_init' : -77,             # initial crossmembrane potential 
    #'e_pas' : -90,              # reversal potential passive mechs
    'passive' : False,           # switch on passive mechs
    'nsegs_method' : 'lambda_f',# method for setting number of segments,
    'lambda_f' : 100,           # segments are isopotential at this frequency
    'timeres_NEURON' : timeres,   # dt of LFP and NEURON simulation.
    'timeres_python' : timeres,
    'tstartms' : 0,          #start time, recorders start at t=0
    'tstopms' : tstopms + cut_off, 
    'custom_code'  : [join(model_path, 'custom_codes.hoc'), \
                      join(model_path, 'biophys3_%s.hoc' % conductance)],
}
ntsteps = round((tstopms - 0) / timeres)
aLFP.initialize_cell(cell_params, pos_params, rot_params, model, elec_x, elec_y, elec_z, ntsteps, model)

#aLFP.run_simulation(cell_params, input_scalings[0], is_active, input_idxs[0], model)


cell_params['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                              join(model_path, 'biophys3_active.hoc')]
aLFP.run_all_simulations(cell_params, True,  model, input_idxs, input_scalings, ntsteps)

cell_params['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                              join(model_path, 'biophys3_passive.hoc')]
aLFP.run_all_simulations(cell_params, False,  model, input_idxs, input_scalings, ntsteps)



#cell_params['custom_fun_args'] = [{'is_active': False}]    
#aLFP.run_all_simulations(cell_params, False,  model, input_idxs, input_scalings)
