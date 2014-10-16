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
from hay_active_declarations import active_declarations as hay_active
from ca1_sub_declarations import active_declarations as ca1_sub_active

neuron_models = join('..', 'neuron_models')

hay_dict = {
    'cellname': 'hay',
    'mechanisms': [join(neuron_models, 'hay', 'mod'), neuron_models],
    'morphology': join(neuron_models, 'hay', 'lfpy_version', 'morphologies', 'cell1.hoc'),
    'custom_code': [join(neuron_models, 'hay', 'lfpy_version', 'custom_codes.hoc')],
    'active_declarations': hay_active,
}

c12861_sub_dict = {
    'cellname': 'c12861',
    'mechanisms': [join(neuron_models, 'ca1_sub')],
    'morphology': join(neuron_models, 'ca1_sub', 'c12861', 'c12861.hoc'),
    'active_declarations': ca1_sub_active,
}
n120_sub_dict = {
    'cellname': 'n120',
    'mechanisms': [join(neuron_models, 'ca1_sub')],
    'morphology': join(neuron_models, 'ca1_sub', 'n120', 'n120.hoc'),
    'active_declarations': ca1_sub_active,
}


def simulate(model_dict, input_pos, hold_potential, just_plot, **kwargs):

    cellname = model_dict['cellname']
    timeres = 2**-4
    tstopms = 100

    cell_params = {
        'morphology': model_dict['morphology'],
        'v_init': hold_potential,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': timeres,
        'tstartms': 0,          # start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_fun': [model_dict['active_declarations']],  # will execute this function
    }

    if model_dict['cellname'] == 'hay':
        cell_params['custom_fun_args'] = [{
                                    'conductance_type': kwargs['conductance_type'],
                                    'hold_potential': hold_potential}]
        cell_params['custom_code'] = model_dict['custom_code']
    elif model_dict['cellname'] == 'c12861' or model_dict['cellname'] == 'n120':
        cell_params['custom_fun_args'] = [{'use_channels': kwargs['use_channels'],
                                            'cellname': cellname,
                                            'hold_potential': hold_potential}]
    np.random.seed(1234)
    aLFP.synaptic_reach_simulation(cellname, cell_params, input_pos,
                                   hold_potential, just_plot, **kwargs)


if __name__ == '__main__':


    aLFP.compare_detectable_volumes(0.005)
    just_plot = False
    #
    # channel_pert = [['Ih', 'Im', 'INaP'], ['Ih', 'INaP'], ['Im', 'INaP']]
    # for di in [c12861_sub_dict, n120_sub_dict]:
    #     [neuron.load_mechanisms(mech) for mech in di['mechanisms']]
    #     for pot in [-60, -70, -80]:
    #         for chan_list in channel_pert:
    #             for input_pos in ['soma', 'apic']:
    #                 print input_pos, pot, chan_list
    #                 simulate(di, input_pos, pot, just_plot, use_channels=chan_list)
    #
    # for di in [hay_dict]:
    #    [neuron.load_mechanisms(mech) for mech in di['mechanisms']]
    #    for pot in [-60, -70, -80]:
    #        for cond in ['passive', 'active', 'Ih_linearized']:
    #            for input_pos in ['apic', 'soma']:
    #                print input_pos, pot, cond
    #                simulate(di, input_pos, pot, just_plot, conductance_type=cond)
