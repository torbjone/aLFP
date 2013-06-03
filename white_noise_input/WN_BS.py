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

input_idxs = [0, 12, 31]
input_scalings = [0.0, 0.001, 0.01, 0.1]
model = 'ball_n_stick' 
domain = 'white_noise_%s' %model
np.random.seed(1234)

simulation_params = {'rec_imem': True,
                     'rec_icap': True,
                     'rec_ipas': True,
                     'rec_variables': ['ina', 'ik', 'il_hh'],
                     }

plot_params = {'ymax': 1000,
               'ymin': 0,
               }

#plot_compartments = [0, 5, 10, 15, 20, 25, 30]
n_plots = 6
plot_compartments = [0, 1, 5, 10, 20, 31]#np.array(np.linspace(0, 31, n_plots), dtype=int)

def simulate():
    def active_ball_n_stick(conductance_type):
        if conductance_type == 'passive':
            g_adjust = 0
            g_reduced = 0
        elif conductance_type == 'active':
            g_adjust = 1
            g_reduced = 1
        elif conductance_type == 'reduced':
            g_adjust = 1
            g_reduced = 0
        else: 
            raise RuntimeError, "Wrong conductance type"
            
        for sec in neuron.h.allsec():
            sec.insert('hh')
            sec.gnabar_hh = 0.12 * g_adjust * g_reduced
            sec.gkbar_hh = 0.036 * g_adjust
            sec.gl_hh = 0.0003 * g_adjust * g_reduced
            sec.el_hh = -54.3
            sec.insert('pas')
            sec.cm = 1.0
            sec.Ra = 150
            sec.g_pas = 1./ 30000
            sec.e_pas = -65
    if at_stallo:
        neuron_model = join('/home', 'torbness', 'work', 'aLFP', 'neuron_models', model)
    else:
        neuron_model = join('..', 'neuron_models', model)

    LFPy.cell.neuron.load_mechanisms(join(neuron_model, '..'))      
    cut_off = 100
    conductance_type = 'reduced'

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
        'custom_fun_args' : [{'conductance_type': conductance_type}]
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
    #aLFP.run_simulation(cell_params, input_scalings[0], is_active, input_idxs[0], 
    #                    model, ntsteps, simulation_params)
    #conductance_type = 'active'
    #cell_params['custom_fun_args'] = [{'conductance_type': conductance_type}]  
    #aLFP.run_all_simulations(cell_params, model, input_idxs, input_scalings, ntsteps,
    #                         simulation_params, conductance_type)
    
    conductance_type = 'passive'
    cell_params['custom_fun_args'] = [{'conductance_type': conductance_type}]  
    aLFP.run_all_simulations(cell_params, model, input_idxs, input_scalings, ntsteps,
                             simulation_params, conductance_type)

    conductance_type = 'reduced'
    cell_params['custom_fun_args'] = [{'conductance_type': conductance_type}]  
    aLFP.run_all_simulations(cell_params, model, input_idxs, input_scalings, ntsteps,
                             simulation_params, conductance_type)  



def plot_transfer():
    aLFP.plot_transfer_functions(model, 0.01, 0, plot_params, 
                              simulation_params, plot_compartments, 'passive')
    sys.exit()
    for input_idx in input_idxs:
        for input_scaling in input_scalings:
            print input_idx, input_scaling
            try:
                aLFP.plot_transfer_functions(model, input_scaling, input_idx, plot_params, 
                                          simulation_params, plot_compartments, 'active')
            except:
                continue
    
def plot_active():
    for input_idx in input_idxs:
        for input_scaling in input_scalings:
            print input_idx, input_scaling
            try:
                #aLFP.plot_active_currents(model, input_scaling, input_idx, plot_params, 
                #                          simulation_params, plot_compartments, 'active')
                #sys.exit()
                #aLFP.plot_active_currents(model, input_scaling, input_idx, plot_params, 
                #                          simulation_params, plot_compartments,'reduced')
                aLFP.plot_active_currents(model, input_scaling, input_idx, plot_params, 
                                          simulation_params, plot_compartments, 'passive')
            except:
                continue
def plot_compare():
    #aLFP.compare_active_passive(model, input_scalings[0] , input_idxs[1], 
    #                            elec_x, elec_y, elec_z, plot_params)
    #sys.exit()
    for input_idx in input_idxs:
        for input_scaling in input_scalings:
            print input_idx, input_scaling
            aLFP.compare_active_passive(model, input_scaling , input_idx, 
                                        elec_x, elec_y, elec_z, plot_params)

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python %s <function-name> \nfunction-name can be plot of simulate" 
                         % sys.argv[0])
        raise SystemExit(1)
    func = eval('%s' % sys.argv[1])
    func()
