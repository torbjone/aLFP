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

model = 'hay' 
#np.random.seed(1234)

if at_stallo:
    timeres = 2**-5
    neuron_model = join('/home', 'torbness', 'work', 'aLFP', 'neuron_models', model)
    cut_off = 3000
else:
    neuron_model = join('..', 'neuron_models', model)
    timeres = 2**-4
    cut_off = 3000

simulation_params = {'rec_imem': True,
                     'rec_vmem':True
                     }

print np.random.random()
#input_idx_scale = [[0, 0.001],
#                   [82, 0.001],
#                   [611, 0.001], 
#                   [681, 0.001],
#                   [740, 0.001],
#                   [791, 0.001]]

random_comps = np.random.randint(0,1018, 100)
input_idx_scale =[[comp, 0.001] for comp in random_comps]

ring_dict = {'radiuses': np.array([250., 500., 750., 1000., 1250., 1500.]),
             'numpoints_on_ring': 30,
             'heights': [0, 500, 1000],
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

def simulate_multiple_WN(simulation_idx):
    tstopms = 1000
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
        'tstartms' : 0,          #start time, recorders start at t=0
        'tstopms' : tstopms + cut_off, 
        'custom_code'  : [join(model_path, 'custom_codes.hoc'), \
                          join(model_path, 'biophys3_%s.hoc' % conductance_type)],
    }
    
    simulate = ['active', 'Ih_linearized', 'passive_vss'] 
    ntsteps = round((tstopms - 0) / timeres)
    aLFP.initialize_cell(cell_params, pos_params, rot_params, model, 
                         elec_x, elec_y, elec_z, ntsteps, model, 
                         testing=False, make_WN_input=True, input_idx_scale=input_idx_scale, simulation_idx=simulation_idx)
    
    for conductance_type in simulate:
        temp_sim_params = simulation_params.copy()
        cell_params['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                      join(model_path, 'biophys3_%s.hoc' % conductance_type)]
        #aLFP.run_all_linearized_simulations(cell_params, model, input_idx_scale, ntsteps, 
        #                                    temp_sim_params, conductance_type, 
        #                                    input_type='WN', Ih_distribution='original')

        aLFP.run_multiple_input_simulation(cell_params, model, input_idx_scale, ntsteps,
                                            temp_sim_params, conductance_type, 
                                            input_type='WN', Ih_distribution='original', simulation_idx=simulation_idx)
        
def simulate_WN():
    tstopms = 1000
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
        'tstartms' : 0,          #start time, recorders start at t=0
        'tstopms' : tstopms + cut_off, 
        'custom_code'  : [join(model_path, 'custom_codes.hoc'), \
                          join(model_path, 'biophys3_%s.hoc' % conductance_type)],
    }
    
    simulate = ['active', 'Ih_linearized', 'passive_vss'] 
    ntsteps = round((tstopms - 0) / timeres)
    aLFP.initialize_cell(cell_params, pos_params, rot_params, model, 
                         elec_x, elec_y, elec_z, ntsteps, model, 
                         testing=False, make_WN_input=True, input_idx_scale=input_idx_scale)
    
    for conductance_type in simulate:
        temp_sim_params = simulation_params.copy()
        cell_params['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                      join(model_path, 'biophys3_%s.hoc' % conductance_type)]
        #aLFP.run_all_linearized_simulations(cell_params, model, input_idx_scale, ntsteps, 
        #                                    temp_sim_params, conductance_type, 
        #                                    input_type='WN', Ih_distribution='original')

        aLFP.run_multiple_input_simulation(cell_params, model, input_idx_scale, ntsteps,
                                            temp_sim_params, conductance_type, 
                                            input_type='WN', Ih_distribution='original', average_over_sims=2)
        
def average_circle():
    conductance_list = ['active', 'Ih_linearized', 'passive_vss']
    #for idx_scale in input_idx_scale:
    #    aLFP.average_circle(model, conductance_list, idx_scale, 
    #                        ring_dict, elec_x, elec_y, elec_z)

    if at_stallo: 
        simulation_idx = np.arange(100)
    else:
        simulation_idx = np.arange(3)
    aLFP.average_PSD_over_circle(model, conductance_list, input_idx_scale, 
                                 ring_dict, elec_x, elec_y, elec_z, simulation_idx=simulation_idx)    
        
def plot_setup():
    aLFP.plot_ring(model, elec_x, elec_y, elec_z)

if __name__ == '__main__':

    if len(sys.argv) not in [2, 3]:
        sys.stderr.write("Usage: python %s <function-name> \n" % sys.argv[0])
        raise SystemExit(1)
    func = eval('%s' % sys.argv[1])
    if len(sys.argv) == 3:
        func(int(sys.argv[2]))
    else:
        func()
