__author__ = 'torbjone'

import os
import sys
from os.path import join

if not os.environ.has_key('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
    at_stallo = True
else:
    at_stallo = False
import numpy as np
import pylab as plt
try:
    from ipdb import set_trace
except ImportError:
    pass
import aLFP
import neuron
import LFPy

model = 'shah'
folder = 'WN'
if not os.path.isdir(folder): os.mkdir(folder)

np.random.seed(1234)

if at_stallo:
    neuron_model = join('/home', 'torbness', 'work', 'aLFP', 'neuron_models', model)
    cut_off = 500
else:
    neuron_model = join('..', 'neuron_models', model)
    cut_off = 500

timeres_N = 2**-5
timeres_p = 1
tstopms = 1000
ntsteps_N = round((tstopms - 0) / timeres_N)
ntsteps_p = round((tstopms - 0) / timeres_p)
n_elecs= 8
elec_x = np.ones(n_elecs) * 50
elec_y = np.zeros(n_elecs)
elec_z = np.linspace(-200, 800, n_elecs)

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
    'timeres_NEURON' : timeres_N,   # dt of LFP and NEURON simulation.
    'timeres_python' : timeres_p,
    'tstartms': 0,          #start time, recorders start at t=0
    'tstopms' : tstopms + cut_off,
    'custom_code': [join(model_path, 'fig-3c_active.hoc')],
}
simulation_params = {'rec_imem': True,
                     'rec_vmem': True,
                     }


def static_test():

    for conductance_type in ['active_-60']:
        cell_params['custom_code'] = [join(model_path, 'fig-3c_%s.hoc') % conductance_type]
        aLFP.find_static_Vm_distribution(cell_params, '.', conductance_type)


def initialize():
    aLFP.initialize_cell(cell_params, model, elec_x, elec_y, elec_z,
                         ntsteps_N, folder, make_WN_input=True, testing=False)


def WN_plot():
    for vrest in [-60, -62, -65, -80]:
        for input_idx in [0, 475]:
            for syn_strength in [0.1, 0.01]:
                print vrest, input_idx, syn_strength
                aLFP.plot_WN_CA1(folder, elec_x, elec_y, elec_z, input_idx, vrest, syn_strength)


def WN_sim():
    conductance_list = ['active_-60',
                        'passive_-80', 'passive_-65', 'passive_-62', 'passive_-60',
                        'only_hd_-80', 'only_hd_-65', 'only_hd_-62', 'only_hd_-60',
                        'only_km_-80', 'only_km_-65', 'only_km_-62', 'only_km_-60']
    input_idxs = [0, 475]
    input_scalings = [0.1, 0.01]
    input_shifts = [0.]#[-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5]
    for conductance_type in conductance_list:
        cell_params['custom_code'] = [join(model_path, 'fig-3c_%s.hoc' % conductance_type)]
        aLFP.run_all_CA1_WN_simulations(cell_params, folder, input_idxs, input_scalings, input_shifts, ntsteps_p,
                                        simulation_params, conductance_type)


if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python %s <function-name> \n" % sys.argv[0])
        raise SystemExit(1)
    func = eval('%s' % sys.argv[1])
    func()