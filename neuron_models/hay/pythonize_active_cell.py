__author__ = 'torbjone'

""" Test if hay model can be made uniform by tampering with the static currents
"""

from os.path import join
import LFPy
import neuron
import pylab as plt
from hay_active_declarations import *

plt.seed(0)
timeres = 2**-4
cut_off = 0
tstopms = 1000
tstartms = -cut_off


model_path = join('lfpy_version')
neuron.load_mechanisms(join('mod'))
neuron.load_mechanisms('..')
# Synaptic parameters taken from Hendrickson et al 2011
# Excitatory synapse parameters:
synapseParameters_AMPA = {
    'e': 0,                    #reversal potential
    'syntype': 'Exp2Syn',      #conductance based exponential synapse
    'tau1': 1.,                #Time constant, rise
    'tau2': 3.,                #Time constant, decay
    'weight': 0.005,           #Synaptic weight
    'color': 'r',              #for plt.plot
    'marker': '.',             #for plt.plot
    'record_current': True,    #record synaptic currents
}
# Excitatory synapse parameters
synapseParameters_NMDA = {
    'e': 0,
    'syntype': 'Exp2Syn',
    'tau1': 10.,
    'tau2': 30.,
    'weight': 0.005,
    'color': 'm',
    'marker': '.',
    'record_current': True,
}
# Inhibitory synapse parameters
synapseParameters_GABA_A = {
    'e': -80,
    'syntype': 'Exp2Syn',
    'tau1': 1.,
    'tau2': 12.,
    'weight': 0.005,
    'color': 'b',
    'marker': '.',
    'record_current': True
}

# where to insert, how many, and which input statistics
insert_synapses_AMPA_args = {
    'section': 'apic',
    'n': 100,
    'spTimesFun': LFPy.inputgenerators.stationary_gamma,
    'args': [tstartms, tstopms, 0.5, 40, tstartms]
}
insert_synapses_NMDA_args = {
    'section': ['dend', 'apic'],
    'n': 15,
    'spTimesFun': LFPy.inputgenerators.stationary_gamma,
    'args': [tstartms, tstopms, 2, 50, tstartms]
}
insert_synapses_GABA_A_args = {
    'section': 'dend',
    'n': 100,
    'spTimesFun': LFPy.inputgenerators.stationary_gamma,
    'args': [tstartms, tstopms, 0.5, 40, tstartms]
}

cell_params_hoc = {
    'morphology': join(model_path, 'morphologies', 'cell1.hoc'),
    #'rm' : 30000,               # membrane resistance
    #'cm' : 1.0,                 # membrane capacitance
    #'Ra' : 100,                 # axial resistance
    'v_init': -60,             # initial crossmembrane potential
    'passive': False,           # switch on passive mechs
    'nsegs_method': 'lambda_f',  # method for setting number of segments,
    'lambda_f': 100,           # segments are isopotential at this frequency
    'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
    'timeres_python': 1,
    'tstartms': tstartms,          # start time, recorders start at t=0
    'tstopms': tstopms,
    #'custom_code': [join(model_path, 'custom_codes.hoc'),
    #                join(model_path, 'biophys3_active.hoc')],
}

cell_params_py = {
    'morphology': join(model_path, 'morphologies', 'cell1.hoc'),
    #'rm' : 30000,               # membrane resistance
    #'cm' : 1.0,                 # membrane capacitance
    #'Ra' : 100,                 # axial resistance
    'v_init': -70,             # initial crossmembrane potential
    'passive': False,           # switch on passive mechs
    'nsegs_method': 'lambda_f',  # method for setting number of segments,
    'lambda_f': 100,           # segments are isopotential at this frequency
    'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
    'timeres_python': 1,
    'tstartms': tstartms,          # start time, recorders start at t=0
    'tstopms': tstopms,
    'custom_code': [join(model_path, 'custom_codes.hoc')],
    'custom_fun': [active_declarations],  # will execute this function
    #'custom_fun_args': [{'conductance_type': 'active'}],
}


def insert_synapses(synparams, cell, section, n, spTimesFun, args):
    ''' find n compartments to insert synapses onto '''
    idx = cell.get_rand_idx_area_norm(section=section, nidx=n)
    #Insert synapses in an iterative fashion
    for i in idx:
        synparams.update({'idx' : int(i)})
        # Some input spike train using the function call
        spiketimes = spTimesFun(args[0], args[1], args[2], args[3], args[4])

        # Create synapse(s) and setting times using the Synapse class in LFPy
        s = LFPy.Synapse(cell, **synparams)
        s.set_spike_times(spiketimes)


def plot_cell(cell, name):
    plt.close('all')
    plt.figure(figsize=[10, 5])
    plt.subplot(121, aspect='equal')
    img = plt.scatter(cell.xmid, cell.zmid, c=cell.vmem[:, -1], edgecolor='none')
    plt.axis('off')
    plt.colorbar(img, orientation='horizontal', ticks=img.get_clim())

    plt.subplot(122)
    plt.plot(cell.tvec, cell.vmem[0, :])
    plt.savefig('%s.png' % name)


def test_active_mod():

    cell_params_hoc['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                      join(model_path, 'biophys3_active_mod.hoc')]

    cell_params_py['custom_fun_args'] = [{'conductance_type': 'active',
                                          'hold_potential': -70}]
    plt.seed(0)
    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params_py)
    insert_synapses(synapseParameters_AMPA, cell, **insert_synapses_AMPA_args)
    insert_synapses(synapseParameters_NMDA, cell, **insert_synapses_NMDA_args)
    insert_synapses(synapseParameters_GABA_A, cell, **insert_synapses_GABA_A_args)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '2_active_mod_py')

    plt.seed(0)
    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params_hoc)
    insert_synapses(synapseParameters_AMPA, cell, **insert_synapses_AMPA_args)
    insert_synapses(synapseParameters_NMDA, cell, **insert_synapses_NMDA_args)
    insert_synapses(synapseParameters_GABA_A, cell, **insert_synapses_GABA_A_args)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '2_active_mod_hoc')


def test_active_orig():
    plt.seed(0)

    cell_params_hoc['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                      join(model_path, 'biophys3_active.hoc')]
    cell_params_py['custom_fun_args'] = [{'conductance_type': 'active'}]

    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params_hoc)
    insert_synapses(synapseParameters_AMPA, cell, **insert_synapses_AMPA_args)
    insert_synapses(synapseParameters_NMDA, cell, **insert_synapses_NMDA_args)
    insert_synapses(synapseParameters_GABA_A, cell, **insert_synapses_GABA_A_args)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '1_active_hoc')

    plt.seed(0)
    neuron.h('forall delete_section()')

    cell = LFPy.Cell(**cell_params_py)
    insert_synapses(synapseParameters_AMPA, cell, **insert_synapses_AMPA_args)
    insert_synapses(synapseParameters_NMDA, cell, **insert_synapses_NMDA_args)
    insert_synapses(synapseParameters_GABA_A, cell, **insert_synapses_GABA_A_args)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '1_active_py')


def test_active_mod_no_input():

    cell_params_hoc['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                       join(model_path, 'biophys3_active_mod.hoc')]

    cell_params_py['custom_fun_args'] = [{'conductance_type': 'active',
                                           'hold_potential': -70}]
    plt.seed(0)
    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params_py)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '4_active_mod_no_input_py')

    plt.seed(0)
    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params_hoc)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '4_active_mod_no_input_hoc')


def test_active_no_input():
    plt.seed(0)
    cell_params_hoc['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                      join(model_path, 'biophys3_active.hoc')]
    cell_params_py['custom_fun_args'] = [{'conductance_type': 'active'}]

    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params_hoc)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '3_active_no_input_hoc')

    plt.seed(0)
    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params_py)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '3_active_no_input_py')


def test_passive_no_input():

    cell_params_hoc['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                       join(model_path, 'biophys3_passive.hoc')]

    cell_params_py['custom_fun_args'] = [{'conductance_type': 'passive',
                                           'hold_potential': -70}]
    plt.seed(0)
    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params_hoc)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '8_passive_no_input_hoc')

    plt.seed(0)
    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params_py)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '8_passive_no_input_py')


def test_passive():

    cell_params_hoc['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                      join(model_path, 'biophys3_passive.hoc')]

    cell_params_py['custom_fun_args'] = [{'conductance_type': 'passive',
                                          'hold_potential': -70}]
    plt.seed(0)
    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params_hoc)
    insert_synapses(synapseParameters_AMPA, cell, **insert_synapses_AMPA_args)
    insert_synapses(synapseParameters_NMDA, cell, **insert_synapses_NMDA_args)
    insert_synapses(synapseParameters_GABA_A, cell, **insert_synapses_GABA_A_args)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '7_passive_hoc')

    plt.seed(0)
    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params_py)
    insert_synapses(synapseParameters_AMPA, cell, **insert_synapses_AMPA_args)
    insert_synapses(synapseParameters_NMDA, cell, **insert_synapses_NMDA_args)
    insert_synapses(synapseParameters_GABA_A, cell, **insert_synapses_GABA_A_args)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '7_passive_py')


def test_linearized_no_input():

    cell_params_hoc['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                       join(model_path, 'biophys3_Ih_linearized_mod.hoc')]

    cell_params_py['custom_fun_args'] = [{'conductance_type': 'Ih_linearized',
                                          'hold_potential': -70}]

    # plt.seed(0)
    # neuron.h('forall delete_section()')
    # cell = LFPy.Cell(**cell_params_hoc)
    # cell.simulate(rec_vmem=True, rec_imem=True)
    # plot_cell(cell, '6_linearized_no_input_hoc')

    plt.seed(0)
    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params_py)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '6_linearized_no_input_py')


def test_linearized():

    cell_params_hoc['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                       join(model_path, 'biophys3_Ih_linearized_mod.hoc')]

    cell_params_py['custom_fun_args'] = [{'conductance_type': 'Ih_linearized',
                                          'hold_potential': -70}]

    plt.seed(0)
    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params_hoc)
    insert_synapses(synapseParameters_AMPA, cell, **insert_synapses_AMPA_args)
    insert_synapses(synapseParameters_NMDA, cell, **insert_synapses_NMDA_args)
    insert_synapses(synapseParameters_GABA_A, cell, **insert_synapses_GABA_A_args)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '5_linearized_hoc')

    plt.seed(0)
    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params_py)
    insert_synapses(synapseParameters_AMPA, cell, **insert_synapses_AMPA_args)
    insert_synapses(synapseParameters_NMDA, cell, **insert_synapses_NMDA_args)
    insert_synapses(synapseParameters_GABA_A, cell, **insert_synapses_GABA_A_args)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plot_cell(cell, '5_linearized_py')


if __name__ == '__main__':

    # test_active_mod()
    # test_active_orig()
    # test_active_mod_no_input()
    # test_active_no_input()
    # test_passive_no_input()
    # test_passive()
    test_linearized_no_input()
    # test_linearized()
