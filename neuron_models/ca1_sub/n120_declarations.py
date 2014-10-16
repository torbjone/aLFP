__author__ = 'torbjone'

import os
from os.path import join
import sys
import neuron
from neuron import h as nrn
import LFPy
import numpy as np
import pylab as plt
import aLFP
from ca1_sub_declarations import active_declarations


def test_steady_state(input_idx, hold_potential):

    timeres = 2**-4
    cut_off = 0
    tstopms = 500
    tstartms = -cut_off
    model_path = 'n120'

    cell_params = {
        'morphology': join(model_path, '%s.hoc' % model_path),
        #'rm' : 30000,               # membrane resistance
        #'cm' : 1.0,                 # membrane capacitance
        #'Ra' : 100,                 # axial resistance
        'v_init': hold_potential,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': timeres,
        'tstartms': tstartms,          # start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'use_channels': ['Im'],
                             'cellname': model_path,
                             'hold_potential': hold_potential}],
    }

    cell = LFPy.Cell(**cell_params)

    apic_stim_idx = cell.get_idx('apic[8]')[0]
    figfolder = join(model_path, 'verifications')
    if not os.path.isdir(figfolder): os.mkdir(figfolder)

    plt.seed(1234)
    apic_tuft_idx = cell.get_closest_idx(-400, 0, -50)
    trunk_idx = cell.get_closest_idx(-100, 0, 0)
    axon_idx = cell.get_idx('axon_IS')[0]
    basal_idx = cell.get_closest_idx(100, 100, 0)
    soma_idx = 0

    print input_idx, hold_potential
    idx_list = np.array([soma_idx, apic_stim_idx, apic_tuft_idx,
                         trunk_idx, axon_idx, basal_idx])

    input_scaling = .01

    sim_params = {'rec_vmem': True,
                  'rec_imem': True,
                  'rec_variables': []}
    cell.simulate(**sim_params)

    simfolder = join(model_path, 'simresults')
    if not os.path.isdir(simfolder): os.mkdir(simfolder)

    simname = join(simfolder, 'simple_%d_%1.3f' % (input_idx, input_scaling))
    if 'use_channels' in cell_params['custom_fun_args'][0] and \
                    len(cell_params['custom_fun_args'][0]['use_channels']) > 0:
        for ion in cell_params['custom_fun_args'][0]['use_channels']:
            simname += '_%s' % ion
    else:
        simname += '_passive'

    if 'hold_potential' in cell_params['custom_fun_args'][0]:
        simname += '_%+d' % cell_params['custom_fun_args'][0]['hold_potential']

    [plt.plot(cell.tvec, cell.vmem[idx, :]) for idx in xrange(len(cell.xmid))]
    # plt.plot(cell.tvec, cell.somav)
    plt.show()
    #plot_cell_steady_state(cell)

if __name__ == '__main__':
    # aLFP.explore_morphology(join('n120', 'n120.hoc'))

    test_steady_state(0, -80)