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

np.random.seed(1234)

if at_stallo:
    neuron_model = join('/home', 'torbness', 'work', 'aLFP', 'neuron_models', model)
    cut_off = 6000
else:
    neuron_model = join('..', 'neuron_models', model)
    cut_off = 1000

simulation_params = {'rec_imem': True,
                     'rec_vmem':True
                     }
plot_params = {'ymax': 1250,
               'ymin': -250,
               }

input_idxs = [0, 791]
input_scalings = [0.15]

epas_array = [-90]#-100,-90, -80, -70]
#vss_array = [-150, -100, -77, -50] 
n_plots = 10
plot_compartments = np.array(np.linspace(0, 1000, n_plots), dtype=int)

def simulate():
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
    conductance_type = 'active'
    
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
    #aLFP.test_static_Vm_distribution(cell_params, model, conductance_type)
    #aLFP.find_static_Vm_distribution(cell_params, model, conductance_type)
    #sys.exit()
    simulate = ['active', 'Ih_linearized', 'passive_vss', 'Ih_reduced'] 
    ntsteps = round((tstopms - 0) / timeres)
    aLFP.initialize_cell(cell_params, pos_params, rot_params, model, 
                         elec_x, elec_y, elec_z, ntsteps, model, testing=False)
    single_run = 0
    if single_run:
        cell_params['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                      join(model_path, 'biophys3_passive_vss.hoc')]
        temp_sim_params = simulation_params.copy()
        if conductance_type == 'passive':
            temp_sim_params['rec_variables'] = []
        elif conductance_type == 'reduced_Ih':
            temp_sim_params['rec_variables'] = ['ihcn_Ih']
        aLFP.run_synaptic_simulation(cell_params, input_scalings[0], input_idxs[0], 
                            model, ntsteps, temp_sim_params, conductance_type, epas=-90)
    for conductance_type in simulate:
        temp_sim_params = simulation_params.copy()            
        cell_params['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                      join(model_path, 'biophys3_%s.hoc' % conductance_type)]
        aLFP.run_all_linearized_simulations(cell_params, model, input_idxs, input_scalings, ntsteps, 
                                          temp_sim_params, conductance_type, input_type='ZAP')

def plot_synaptic():
    for epas in epas_array:
        aLFP.plot_synaptic_currents(model, input_scalings[0], input_idxs[1], plot_params, 
                                    simulation_params, plot_compartments, epas=epas)

def plot_LFPs():
    conductance_list = ['active', 'Ih_linearized', 'passive_vss']
    for input_idx in input_idxs:
        for input_scaling in input_scalings:
            aLFP.compare_LFPs(model, input_scaling, input_idx, elec_x, elec_y, elec_z, plot_params, 
                              conductance_list)
            #sys.exit()
def plot_transfer():
    aLFP.plot_transfer_functions(model, 0.001, 0, plot_params, 
                              simulation_params, plot_compartments, -90)
    sys.exit()
    for input_idx in input_idxs:
        for input_scaling in input_scalings:
            print input_idx, input_scaling
            try:
                aLFP.plot_transfer_functions(model, input_scaling, input_idx, plot_params, 
                                          simulation_params, plot_compartments)
            except:
                continue

def plot_active():
    ifolder = 'hay'
    #aLFP.plot_active_currents(ifolder, 1.0, 0, plot_params, 
    #                          simulation_params, plot_compartments, 'active')
    #sys.exit()
    for input_idx in input_idxs:
        for input_scaling in input_scalings:
            print input_idx, input_scaling          
            #aLFP.plot_active_currents(model, input_scaling, input_idx, plot_params, 
            #                          simulation_params, plot_compartments, 'active')
            #try:
            #    aLFP.plot_active_currents(model, input_scaling, input_idx, plot_params, 
            #                              simulation_params, plot_compartments, 'passive')
            #except:
            #    pass
            #aLFP.plot_active_currents(model, input_scaling, input_idx, plot_params, 
            #                          simulation_params, plot_compartments, 'reduced')
            #try:
            #    aLFP.plot_active_currents(model, input_scaling, input_idx, plot_params,
            #                              simulation_params, plot_compartments, 'reduced_with_na')
            #except:
            #    pass
            #try:
            aLFP.plot_active_currents(model, input_scaling, input_idx, plot_params, 
                                      simulation_params, plot_compartments, 'passive', -90)
            #except:
            #    pass
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
