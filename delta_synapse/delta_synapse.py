#!/usr/bin/env python
import os
if not os.environ.has_key('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
    at_stallo = True
else:
    at_stallo = False
import LFPy
import numpy as np
import neuron
import sys
try:
    from ipdb import set_trace
except:
    pass
import pylab as plt
from os.path import join
import aLFP
import pickle

model = 'hay' 
#np.random.seed(1234)
print np.random.random()
if at_stallo:
    neuron_model = join('/home', 'torbness', 'work', 'aLFP', 'neuron_models', model)
    cut_off = 200
    timeres = 2**-6
else:
    neuron_model = join('..', 'neuron_models', model)
    cut_off = 200
    timeres = 2**-6

tstopms = 1000
ntsteps = round((tstopms - 0) / timeres)

ring_dict = {'radiuses': np.array([10., 25, 50., 100., 250., 500., 1000., 2500., 5000.]),
             'numpoints_on_ring': 30,
             'heights': np.array([0., 500., 1000.]),
             }

n_elecs = len(ring_dict['radiuses']) * ring_dict['numpoints_on_ring'] * len(ring_dict['heights'])

elec_x = np.zeros(n_elecs)
elec_y = np.zeros(n_elecs)
elec_z = np.zeros(n_elecs)

idx = 0
for height in ring_dict['heights']:
    for radius in ring_dict['radiuses']:
        for angle_idx in xrange(ring_dict['numpoints_on_ring']):
            angle = angle_idx * 2*np.pi / ring_dict['numpoints_on_ring']
            elec_x[idx] = radius * np.cos(angle)
            elec_z[idx] = radius * np.sin(angle)
            elec_y[idx] = height
            idx += 1

conductance_type = 'passive'
model_path = join(neuron_model, 'lfpy_version')
LFPy.cell.neuron.load_mechanisms(join(neuron_model, 'mod'))      
LFPy.cell.neuron.load_mechanisms(join(neuron_model, '..'))      

rot_params = {'x': -np.pi/2, 
              'y': 0, 
              'z': 0
              }
pos_params = {'xpos': 0, 
              'ypos': 0,
              'zpos': 0,
              }        

cell_params = {
    'morphology' : join(model_path, 'morphologies', 'cell1.hoc'),
    #'rm' : 30000,               # membrane resistance
    #'cm' : 1.0,                 # membrane capacitance
    #'Ra' : 100,                 # axial resistance
    'v_init' : -77,             # initial crossmembrane potential 
    'passive' : False,           # switch on passive mechs
    'nsegs_method' : 'lambda_f',# method for setting number of segments,
    'lambda_f' : 100,           # segments are isopotential at this frequency
    'timeres_NEURON' : timeres,   # dt of LFP and NEURON simulation.
    'timeres_python' : 1,
    'tstartms' : -cut_off,          #start time, recorders start at t=0
    'tstopms' : tstopms, 
    'custom_code'  : [join(model_path, 'custom_codes.hoc'), \
                      join(model_path, 'biophys3_%s.hoc' % conductance_type)],
}

synaptic_params = {'section': ['soma', 'apic', 'dend'],
                   'n' : 1000,
                   'spTimesFun' : LFPy.inputgenerators.stationary_poisson,
                   'args' : [1, 5, cell_params['tstartms'], cell_params['tstopms']]
                   }

def initialize_cell():

    aLFP.initialize_cell(cell_params, pos_params, rot_params, model, 
                         elec_x, elec_y, elec_z, ntsteps, model, 
                         testing=False, make_WN_input=False)


def simulate_single_cell():
    """ One long cell simulation will be used to draw short 
    random sequences of membrane currents to build LFP 
    """  
    conductance_list = ['active', 'Ih_linearized',  'passive_vss']

    aLFP.run_delta_synapse_simulation(cell_params, conductance_list, model, model_path, 
                                   ntsteps, synaptic_params, simulation_idx=int(sys.argv[2]))

def test_plots():
    aLFP.delta_synapse_PSD(model, 'active')
    aLFP.delta_synapse_PSD(model, 'Ih_linearized')
    aLFP.delta_synapse_PSD(model, 'passive_vss')

def ring_plot():
    conductance_list = ['active', 'Ih_linearized', 'passive_vss']
    folder = join('hay')
    filename_root = 'signal_psd'
    input_pos = 'dend'
    #aLFP.average_PSD_on_rings(folder, conductance_list, 'apic', filename_root)
    #aLFP.average_PSD_on_rings(folder, conductance_list, 'dend', filename_root)
    #aLFP.average_PSD_on_rings(folder, conductance_list, 'homogeneous', filename_root)
    #aLFP.new_ring_dist_decay_plot(folder, model, conductance_list, input_pos, ring_dict, elec_x, elec_y, elec_z)
    aLFP.new_ring_plot(folder, model, conductance_list, input_pos, ring_dict, elec_x, elec_y, elec_z)
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python %s <function-name> \n" % sys.argv[0])
        raise SystemExit(1)
    func = eval('%s' % sys.argv[1])
    func()

    
