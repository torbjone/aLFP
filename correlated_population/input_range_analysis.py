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
    cut_off = 2000
    timeres = 2**-6
else:
    neuron_model = join('..', 'neuron_models', model)
    cut_off = 2000
    timeres = 2**-6

num_cells = 10000
population_radius = 1000

tstopms = 1000
ntsteps = round((tstopms - 0) / timeres)

n_elecs = 8
elec_x = np.zeros(n_elecs)
elec_y = np.zeros(n_elecs)
elec_z = np.linspace(-200, 1200, n_elecs)
#np.save(join(model, 'elec_x.npy'), elec_x)
#np.save(join(model, 'elec_y.npy'), elec_y)
#np.save(join(model, 'elec_z.npy'), elec_z)

model_path = join(neuron_model, 'lfpy_version')
LFPy.cell.neuron.load_mechanisms(join(neuron_model, 'mod'))      
LFPy.cell.neuron.load_mechanisms(join(neuron_model, '..'))      

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
                      join(model_path, 'biophys3_passive.hoc')],
}


def make_population():
    aLFP.distribute_cells(num_cells, population_radius)

def make_all_input_trains():
    """ Makes all the input spike trains. Totally 1000 / 0.01, since that is the
    maximum number of needed ones"""
    num_trains = int(1000/0.01)
    all_spiketimes = {}
    for idx in xrange(num_trains):
        all_spiketimes[idx] = LFPy.inputgenerators.stationary_poisson(
            1, 5, cell_params['tstartms'], cell_params['tstopms'])[0]
    np.save(join(model, 'all_spike_trains.npy'), all_spiketimes)
    
def simulate_single_cell():
    """ One long cell simulation will be used to draw short 
    random sequences of membrane currents to build LFP 
    """  
    conductance_list = ['active', 'Ih_linearized', 'passive_vss']

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
    aLFP.run_correlated_population_simulation(cell_params, conductance_list, model, model_path, 
                                              elec_x, elec_y, elec_z, ntsteps, spiketrain_params, 
                                              correlation, num_cells, population_radius, 
                                              simulation_idx=int(sys.argv[4]))

def input_study():
    aLFP.compare_psd_of_input()
    
def sum_all_signals():
    correlation = float(sys.argv[2])
    input_pos = sys.argv[3]
    conductance_type = sys.argv[4]
    aLFP.sum_signals(model, conductance_type, num_cells, n_elecs, input_pos, correlation)

def sum_population_sizes():
    correlations = [0., 1.0]
    input_pos = 'dend'
    conductance_list = ['active', 'Ih_linearized',  'passive_vss']
    aLFP.population_size_summary(conductance_list, elec_x, elec_y, elec_z)
    #aLFP.population_size_frequency_dependence(conductance_list, input_pos, correlations)
    #aLFP.population_size_amp_dependence(conductance_list, input_pos, correlations)
    #aLFP.sum_signals_population_sizes(model, conductance_list, num_cells, n_elecs,
    #                                    input_pos, correlations, population_radius)
    
def plot():
    correlation = float(sys.argv[2])
    input_pos = sys.argv[3]
    conductance_list = ['active', 'Ih_linearized',  'passive_vss']
    aLFP.plot_correlated_population(model, conductance_list, num_cells, 
                                    elec_x, elec_y, elec_z, input_pos, correlation)
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python %s <function-name> <additional arguments>\n" % sys.argv[0])
        raise SystemExit(1)
    func = eval('%s' % sys.argv[1])
    func()
