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
np.random.seed(1234)

if at_stallo:
    neuron_model = join('/home', 'torbness', 'work', 'aLFP', 'neuron_models', model)
    cut_off = 6000
else:
    neuron_model = join('..', 'neuron_models', model)
    cut_off = 100

plot_params = {'ymax': 1250,
               'ymin': -250,
               }

elec_z = np.linspace(-300, 1500, 10)
elec_x = np.zeros(len(elec_z))
elec_y = np.zeros(len(elec_z))

if at_stallo:
    timeres = 2**-5
else:
    timeres = 2**-3

tstopms = 10 * 1000
ntsteps = round((tstopms - 0) / timeres)

population_dict = {'r_limit': 200.,
                   'z_mid': 000,
                   'numcells': 100,
                   'timeres': timeres,
                   'ntsteps': ntsteps,
                   'window_length_ms': 1000, 
                   }

conductance_type = 'passive'
model_path = join(neuron_model, 'lfpy_version')
LFPy.cell.neuron.load_mechanisms(join(neuron_model, 'mod'))      
LFPy.cell.neuron.load_mechanisms(join(neuron_model, '..'))      

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
    'tstartms' : -cut_off,          #start time, recorders start at t=0
    'tstopms' : tstopms, 
    'custom_code'  : [join(model_path, 'custom_codes.hoc'), \
                      join(model_path, 'biophys3_%s.hoc' % conductance_type)],
}

all_synaptic_params = {
    'AMPA':{'section': 'apic',
            'n' : 50,
            'spTimesFun' : LFPy.inputgenerators.stationary_gamma,
            'args' : [cell_params['tstartms'], cell_params['tstopms'], 2, 400],
            },
    'NMDA': {'section' : ['apic'],
             'n' : 2,
             'spTimesFun' : LFPy.inputgenerators.stationary_gamma,
             'args' : [cell_params['tstartms'], cell_params['tstopms'], 2, 400],
             },
    'GABA_A': {'section' : ['soma', 'dend'],
               'n' : 200,
               'spTimesFun' : LFPy.inputgenerators.stationary_gamma,
               'args' : [cell_params['tstartms'], cell_params['tstopms'], 2, 400],
               },                
        }

def simulate_single_cell():
    """ One long cell simulation will be used to draw short 
    random sequences of membrane currents to build LFP 
    """  
    conductance_list = ['passive_vss', 'Ih_linearized', 'Ih_reduced', 'active']
    aLFP.run_population_simulation(cell_params, conductance_list, model, model_path, 
                                   ntsteps, all_synaptic_params, 6)
    
def calc_LFP():
    conductance_list = ['active', 'Ih_linearized', 'Ih_reduced', 'passive_vss']
    neuron_dict = pickle.load(open(join(model, 'neuron_dict.p'), "rb"))
    aLFP.calculate_LFP(neuron_dict, conductance_list, model, population_dict, elec_x, elec_y, elec_z)

    
def create_population():
    """ Creates a dictionary containing positions of all dummy cells in the population """

    aLFP.initialize_dummy_population(population_dict, cell_params,
                    elec_x, elec_y, elec_z, ntsteps, model)
        
if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python %s <function-name> \n" % sys.argv[0])
        raise SystemExit(1)
    func = eval('%s' % sys.argv[1])
    func()
