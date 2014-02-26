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
import sys
try:
    from ipdb import set_trace
except:
    pass
from os.path import join

import neuron
import aLFP

model = 'shah'
folder = 'shah'

if not os.path.isdir(folder): os.mkdir(folder)

np.random.seed(1234)
if at_stallo:
    neuron_model = join('/home', 'torbness', 'work', 'aLFP', 'neuron_models', model)
    cut_off = 2000
    timeres = 2**-6
else:
    neuron_model = join('..', 'neuron_models', model)
    cut_off = 500
    timeres = 2**-6

num_cells = 1
population_radius = 0

tstopms = 1000
ntsteps = round((tstopms - 0) / timeres)

n_elecs_center = 8
elec_x_center = np.ones(n_elecs_center) * 50
elec_y_center = np.zeros(n_elecs_center)
elec_z_center = np.linspace(-200, 800, n_elecs_center)

n_elecs_lateral = 41
elec_x_lateral = np.linspace(0, 10000, n_elecs_lateral)
elec_y_lateral = np.zeros(n_elecs_lateral)
elec_z_lateral = np.zeros(n_elecs_lateral)

elec_x = np.r_[elec_x_center, elec_x_lateral]
elec_y = np.r_[elec_y_center, elec_y_lateral]
elec_z = np.r_[elec_z_center, elec_z_lateral]

center_idxs = np.arange(n_elecs_center)
lateral_idxs = np.arange(n_elecs_center, n_elecs_lateral + n_elecs_center)

n_elecs = len(elec_x)
#np.save(join(folder, 'elec_x.npy'), elec_x)
#np.save(join(folder, 'elec_y.npy'), elec_y)
#np.save(join(folder, 'elec_z.npy'), elec_z)

model_path = join(neuron_model)

neuron.load_mechanisms(join(neuron_model))
neuron.load_mechanisms(join(neuron_model, '..'))

cell_params = {
    'morphology' : join(model_path, 'geo9068802.hoc'),
    #'rm' : 30000,               # membrane resistance
    #'cm' : 1.0,                 # membrane capacitance
    #'Ra' : 100,                 # axial resistance
    'v_init' : -63,             # initial crossmembrane potential 
    'passive' : False,           # switch on passive mechs
    'nsegs_method' : 'lambda_f',# method for setting number of segments,
    'lambda_f' : 100,           # segments are isopotential at this frequency
    'timeres_NEURON' : timeres,   # dt of LFP and NEURON simulation.
    'timeres_python' : 1,
    'tstartms' : -cut_off,          #start time, recorders start at t=0
    'tstopms' : tstopms,
    'custom_code'  : [join(model_path, 'fig-3c_active.hoc')],
}

simulation_params = {'rec_imem': True,
                     'rec_vmem': True,
                     'rec_isyn': False,
                     'rec_icap' : False,
                     'rec_ipas': False,
                     'rec_variables': []#['ina', 'ik', 'ica', 'i_hd', 'ik_km', 'ik_KahpM95',
                                       #'ik_kad', 'ik_kap', 'ik_kdr'],
                     }

spiketrain_params = {'section': ['tuft'],
                     'n' : 1000,
                     'spTimesFun' : LFPy.inputgenerators.stationary_poisson,
                     'args' : [1, 5, cell_params['tstartms'], cell_params['tstopms']]
                    }


def make_all_input_trains():
    """ Makes all the input spike trains. Totally 1000 / 0.01, since that is the
    maximum number of needed ones"""
    num_trains = int(1000/0.01)
    all_spiketimes = {}
    for idx in xrange(num_trains):
        all_spiketimes[idx] = LFPy.inputgenerators.stationary_poisson(
            1, 5, cell_params['tstartms'], cell_params['tstopms'])[0]
    np.save(join(folder, 'all_spike_trains.npy'), all_spiketimes)


def make_population():
    aLFP.distribute_cells(num_cells, population_radius)

def quickplot_state():
    #aLFP.plot_compare_input_states(folder)
    aLFP.plot_compare_single_input_state(folder, elec_x, elec_y, elec_z)

def simulate_single_cell():
    """ One long cell simulation will be used to draw short 
    random sequences of membrane currents to build LFP 
    """  

    conductance_list = ['active', 'passive', 'hd_and_km', 'only_km', 'only_hd']


    if sys.argv[3] == 'homogeneous':
        spiketrain_params = {'section': ['apic', 'dend'],
                             'n' : 1000,
                             'spTimesFun' : LFPy.inputgenerators.stationary_poisson,
                             'args' : [1, 5, cell_params['tstartms'], cell_params['tstopms']]
                             }
    else:
        spiketrain_params = {'section': [sys.argv[3]],
                             'n' : 1000,
                             'spTimesFun' : LFPy.inputgenerators.stationary_poisson,
                             'args' : [1, 5, cell_params['tstartms'], cell_params['tstopms']]
                             }
    
    correlation = float(sys.argv[2])
    syn_strength = float(sys.argv[6])
    simulation_idx = int(sys.argv[5])
    resting_pot_shift = int(sys.argv[4])
    aLFP.run_CA1_correlated_population_simulation(cell_params, conductance_list, folder, model_path, 
                                              elec_x, elec_y, elec_z, ntsteps, spiketrain_params, 
                                              correlation, num_cells, population_radius, simulation_idx,
                                              syn_strength, resting_pot_shift)


def compare_currents():
    conductance_list = ['passive_-65', 'only_hd_-65', 'only_km_-65', 'hd_and_km_-65', 'active_-65']
    aLFP.compare_cell_currents('shah', 0.001, 0, conductance_list, 'homogeneous')
    #aLFP.compare_cell_currents('probe_shah', 0.1, 0, conductance_list)
    #aLFP.compare_cell_currents('probe_shah', 0.1, 16, conductance_list)
    #aLFP.compare_cell_currents('probe_shah', 0.1, 17, conductance_list)


def is_imem_less_affected_then_vmem():
    """ One long cell simulation will be used to draw short
    random sequences of membrane currents to build LFP
    """
    conductance_list = ['hd_and_km_-80', 'hd_and_km_-65']

    if sys.argv[2] == 'homogeneous':
        spiketrain_params = {'section': ['apic', 'dend'],
                             'n' : 1000,
                             'spTimesFun' : LFPy.inputgenerators.stationary_poisson,
                             'args' : [1, 5, cell_params['tstartms'], cell_params['tstopms']]
                             }
    else:
        spiketrain_params = {'section': [sys.argv[2]],
                             'n' : 1000,
                             'spTimesFun' : LFPy.inputgenerators.stationary_poisson,
                             'args' : [1, 5, cell_params['tstartms'], cell_params['tstopms']]
                             }

    correlation = 1.0
    syn_strength = 0.001
    simulation_idx = 0
    resting_pot_shift = 0
    aLFP.run_CA1_correlated_population_simulation(cell_params, conductance_list, folder, model_path,
                                              elec_x, elec_y, elec_z, ntsteps, spiketrain_params,
                                              correlation, num_cells, population_radius, simulation_idx,
                                              syn_strength, resting_pot_shift)



#aLFP.quickprobe_cell(cell_params, folder)

def probe_cell():

#    cell_params['custom_code'] = [join(model_path, 'fig-3c_mod.hoc')]
#    for shift in [-30, 0, 16, 17]:
#        aLFP.probe_active_currents(cell_params, folder, simulation_params, shift, .1, spiketrain_params,
#                                   'active')
#        #aLFP.plot_cell_probe(folder, simulation_params, .1, shift)

#    cell_params['custom_code'] = [join(model_path, 'fig-3c_passive.hoc')]
#    for shift in [-30, 0, 16, 17]:
#        aLFP.probe_active_currents(cell_params, folder, simulation_params, shift, .1, spiketrain_params,
#                                   'passive')
        #aLFP.plot_cell_probe(folder, simulation_params, .1, shift)

    shift = int(sys.argv[2])
    conductance_type = sys.argv[3]
    print("Running ", shift, conductance_type)
    cell_params['custom_code'] = [join(model_path, 'fig-3c_%s.hoc' % conductance_type)]
    #for shift in [-30, 0, 16, 17]:
    aLFP.probe_active_currents(cell_params, folder, simulation_params, shift, .1, spiketrain_params,
                                   conductance_type)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python %s <function-name> <additional arguments>\n" % sys.argv[0])
        raise SystemExit(1)
    func = eval('%s' % sys.argv[1])
    func()
