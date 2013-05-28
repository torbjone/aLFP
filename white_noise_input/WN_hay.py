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
import pylab as pl
from os.path import join
import aLFP
from params import *

model = 'hay' 
domain = 'white_noise_%s' %model

np.random.seed(1234)

if at_stallo:
    neuron_model = join('/home', 'torbness', 'work', 'aLFP', 'neuron_models', model)
else:
    neuron_model = join('..', 'neuron_models', model)

simulation_params = {'rec_imem': True,
                     'rec_icap' : True,
                     'rec_ipas': True,
                     'rec_variables': ['ina', 'ik', 'ica', 'ihcn_Ih'],
                     }
plot_params = {'ymax': 1250,
               'ymin': -250,
               }

n_plots = 10
plot_compartments = np.array(np.linspace(0, 31, n_plots), dtype=int)

input_idxs = [0]#, 791, 611, 808, 681, 740, 606]
input_scalings = [0.]#, 0.001, 0.01, 0.1, 1.0]

def simulate():
    model_path = join(neuron_model, 'lfpy_version')
    LFPy.cell.neuron.load_mechanisms(join(neuron_model, 'mod'))      
    LFPy.cell.neuron.load_mechanisms(join(neuron_model, '..'))      

    cut_off = 1000
    is_active = True
    is_reduced = True
    
    rot_params = {'x': -np.pi/2, 
                  'y': 0, 
                  'z': 0
                  }

    pos_params = {'xpos': 0, 
                  'ypos': 0,
                  'zpos': 0,
                  }        
    conductance_type = 'reduced_with_na'
    
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
                          join(model_path, 'biophys3_%s.hoc' % conductance_type)],
    }
    
    ntsteps = round((tstopms - 0) / timeres)
    aLFP.initialize_cell(cell_params, pos_params, rot_params, model, 
                         elec_x, elec_y, elec_z, ntsteps, model, testing=False)
    #aLFP.run_simulation(cell_params, input_scalings[2], is_active, input_idxs[0], 
    #                    model, ntsteps, simulation_params)
    ## cell_params['custom_code'] = [join(model_path, 'custom_codes.hoc'),
    ##                               join(model_path, 'biophys3_active.hoc')]
    ## aLFP.run_all_simulations(cell_params, model, input_idxs, 
    ##                          input_scalings, ntsteps, simulation_params, 'active')

    ## cell_params['custom_code'] = [join(model_path, 'custom_codes.hoc'),
    ##                               join(model_path, 'biophys3_passive.hoc')]
    ## aLFP.run_all_simulations(cell_params, model, input_idxs, 
    ##                          input_scalings, ntsteps, simulation_params, 'passive')
    
    cell_params['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                  join(model_path, 'biophys3_reduced_with_na.hoc')]
    aLFP.run_all_simulations(cell_params, model, input_idxs, 
                             input_scalings, ntsteps, simulation_params, 'reduced_with_na')

def plot_active():

    aLFP.plot_active_currents(model, 0.01, 0, plot_params, 
                              simulation_params, plot_compartments, 'active')
    sys.exit()
    for input_idx in input_idxs:
        for input_scaling in input_scalings:
            print input_idx, input_scaling
            aLFP.plot_active_currents(model, input_scaling, input_idx, plot_params, 
                                      simulation_params, plot_compartments, 'active')

    
def plot_compare():
    #aLFP.compare_active_passive(model, input_scalings[0] , input_idxs[1], 
    #elec_x, elec_y, elec_z, plot_params)
    #sys.exit()    
    for input_idx in input_idxs:
        for input_scaling in input_scalings:
            print input_idx, input_scaling
            aLFP.compare_active_passive(model, input_scaling , input_idx, 
                                        elec_x, elec_y, elec_z, plot_params)

            
if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python %s <function-name> \n" % sys.argv[0])
        raise SystemExit(1)
    func = eval('%s' % sys.argv[1])
    func()
