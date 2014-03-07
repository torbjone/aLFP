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
from hay_active_declarations import active_declarations


def simulate(cell_name, input_pos, hold_potential, conductance_type, just_plot):
    model = 'hay'
    if at_stallo:
        neuron_model = join('/home', 'torbness', 'work', 'aLFP', 'neuron_models', model)
    else:
        neuron_model = join('..', 'neuron_models', model)
    model_path = join(neuron_model, 'lfpy_version')
    LFPy.cell.neuron.load_mechanisms(join(neuron_model, 'mod'))      
    LFPy.cell.neuron.load_mechanisms(join(neuron_model, '..'))
    timeres = 2**-4
    tstopms = 100

    cell_params = {
        'morphology': join(model_path, 'morphologies', 'cell1.hoc'),
        'v_init': hold_potential,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': timeres,
        'tstartms': 0,          # start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_code': [join(model_path, 'custom_codes.hoc')],
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'conductance_type': conductance_type,
                             'hold_potential': hold_potential}],
    }
    np.random.seed(1234)
    aLFP.synaptic_reach_simulation(cell_name, cell_params, input_pos,
                                   hold_potential, conductance_type, just_plot)

if __name__ == '__main__':

    just_plot = False
    # simulate('hay', 'soma', -60, 'active', just_plot)
    # simulate('hay', 'apic', -60, 'active', just_plot)
    # simulate('hay', 'soma', -60, 'Ih_linearized', just_plot)
    # simulate('hay', 'apic', -60, 'Ih_linearized', just_plot)
    # simulate('hay', 'soma', -60, 'passive', just_plot)
    # simulate('hay', 'apic', -60, 'passive', just_plot)

    # simulate('hay', 'soma', -70, 'active', just_plot)
    # simulate('hay', 'apic', -70, 'active', just_plot)
    # simulate('hay', 'soma', -70, 'Ih_linearized', just_plot)
    simulate('hay', 'apic', -70, 'Ih_linearized', just_plot)
    # simulate('hay', 'soma', -70, 'passive', just_plot)
    # simulate('hay', 'apic', -70, 'passive', just_plot)

    # simulate('hay', 'soma', -80, 'active', just_plot)
    # simulate('hay', 'apic', -80, 'active', just_plot)
    # simulate('hay', 'soma', -80, 'Ih_linearized', just_plot)
    # simulate('hay', 'apic', -80, 'Ih_linearized', just_plot)
    # simulate('hay', 'soma', -80, 'passive', just_plot)
    # simulate('hay', 'apic', -80, 'passive', just_plot)