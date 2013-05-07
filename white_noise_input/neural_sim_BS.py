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

#from tools import *
    
def active_ball_n_stick(is_active):

    for sec in neuron.h.allsec():
        if is_active:
            sec.insert('hh')
            sec.gnabar_hh = 0.12
            sec.gkbar_hh = 0.036
            sec.gl_hh = 0.0003
            sec.el_hh = -54.3
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 150
        sec.g_pas = 1./ 30000
        sec.e_pas = -65

model = 'ball_n_stick' 
domain = 'white_noise_%s' %model

np.random.seed(1234)
neuron_model = join('..', 'neuron_models', model)

LFPy.cell.neuron.load_mechanisms(join('..', 'neuron_models'))      

cut_off = 100
is_active = True
input_idxs = [0, 12]

input_scalings = [0.001, 0.01, 0.1]

cell_params = {
    'morphology' : join(neuron_model, 'ball_n_stick.hoc'),
    'passive' : False,           # switch on passive mechs
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 100,           
    'timeres_NEURON' : timeres,  
    'timeres_python' :  timeres,
    'tstartms' : 0,          
    'tstopms' : tstopms + cut_off,          
    'custom_fun'  : [active_ball_n_stick],
    'custom_fun_args' : [{'is_active': True}],  
    }        


ntsteps = round((tstopms - 0) / timeres)

pos_params = {'xpos': 0, 
              'ypos': 0,
              'zpos': 0,
              }        

rot_params = {'x': 0, 
              'y': np.pi, 
              'z': 0
              }

aLFP.initialize_cell(cell_params, pos_params, rot_params, model, elec_x, elec_y, elec_z, ntsteps, model)

#aLFP.run_simulation(cell_params, input_scalings[0], is_active, input_idxs[0], model, ntsteps)


cell_params['custom_fun_args'] = [{'is_active': True}]  
aLFP.run_all_simulations(cell_params, True,  model, input_idxs, input_scalings, ntsteps)

cell_params['custom_fun_args'] = [{'is_active': False}]    
aLFP.run_all_simulations(cell_params, False,  model, input_idxs, input_scalings, ntsteps)
