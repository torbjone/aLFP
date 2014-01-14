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
    cut_off = 6000
else:
    neuron_model = join('..', 'neuron_models', model)
    cut_off = 2000


timeres = 2**-5
tstopms = 10
ntsteps = round((tstopms - 0) / timeres)


n_elecs= 8
elec_x = np.ones(n_elecs) * 50
elec_y = np.zeros(n_elecs)
elec_z = np.linspace(-200, 800, n_elecs)

model_path = join(neuron_model)
LFPy.cell.neuron.load_mechanisms(join(neuron_model))
LFPy.cell.neuron.load_mechanisms(join(neuron_model, '..'))

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
                     }

def initialize():
    aLFP.initialize_cell(cell_params, model, elec_x, elec_y, elec_z, ntsteps, folder, make_WN_input=True, testing=False)



def WN_sim():
    conductance_list = ['active', 'passive', 'hd_and_km', 'only_km', 'only_hd']
    input_idxs = [0, 475]
    input_scalings = [0.1]
    for conductance_type in conductance_list:
        cell_params['custom_code'] = [join(model_path, 'fig-3c_%s.hoc' % conductance_type)]
        aLFP.run_all_WN_simulations(cell_params, folder, input_idxs, input_scalings, ntsteps,
                                 simulation_params, conductance_type)


if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python %s <function-name> \n" % sys.argv[0])
        raise SystemExit(1)
    func = eval('%s' % sys.argv[1])
    func()