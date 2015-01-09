__author__ = 'torbjone'
#!/usr/bin/env python
'''
Returns cell object
'''
import LFPy
import numpy as np
from os.path import join
import neuron
import pylab as plt


def _make_WN_input(cell, max_freq):
    """ White Noise input ala Linden 2010 is made """
    tot_ntsteps = round((cell.tstopms - cell.tstartms)/\
                  cell.timeres_NEURON + 1)
    I = np.zeros(tot_ntsteps)
    tvec = np.arange(tot_ntsteps) * cell.timeres_NEURON
    plt.seed(1234)
    for freq in xrange(1, max_freq + 1):
        I += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())
    return I

def insert_white_noise(cell, input_idx, weight):

    max_freq = 500
    input_array = weight * _make_WN_input(cell, max_freq)

    noise_vec = neuron.h.Vector(input_array)
    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print "Input i" \
                      "nserted in ", sec.name()
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if syn is None:
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noise_vec.play(syn._ref_amp, cell.timeres_NEURON)
    return cell, [syn, noise_vec]


def insert_synapse(cell, input_idx):
    # Define synapse parameters
    synapse_parameters = {
        'idx': input_idx,
        'e': 0.,                   # reversal potential
        'syntype': 'ExpSyn',       # synapse type
        'tau': 2.,                # syn. time constant
        'weight': .1,            # syn. weight
        'record_current': True,
    }

    # Create synapse and set time of synaptic input
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([5.]))
    return cell, synapse

def infinite_axon_quasi_active(**kwargs):
    neuron.h.distance(0, 0)
    for sec in neuron.h.allsec():
        sec.nseg = 21
        sec.insert("QA")
        sec.V_r_QA = -70.
        sec.tau_w_QA = 30.
        sec.Ra = 150.
        sec.cm = 1.0
        sec.g_pas_QA = 1 / 30000.
        sec.g_w_QA = 2 / 30000.
        for seg in sec:
            if neuron.h.distance(seg.x) <= 0:
                seg.mu_QA = sec.g_w_QA / sec.g_pas_QA * kwargs['mu_factor_1']
            else:
                seg.mu_QA = sec.g_w_QA / sec.g_pas_QA * kwargs['mu_factor_2']

def return_cell(cell_name, stimuli='synapse', input_idx=0, weight=0.001, mu_factor_1=None, mu_factor_2=None):
    username = os.getenv('USER')
    root_folder = join('/home', self.username, 'work', 'aLFP')
    neuron_models = join(root_folder, 'neuron_models')
    neuron.load_mechanisms(join(neuron_models))
    cell_parameters = {          # various cell parameters,
        'morphology': 'infinite_axon.hoc', # Mainen&Sejnowski, 1996
        'v_init': -70.,    # initial crossmembrane potential
        'passive': False,   # switch on passive mechs
        'nsegs_method': 'lambda_f',
        'lambda_f': 500.,
        'timeres_NEURON': 2.**-3,   # [ms] dt's should be in powers of 2 for both,
        'timeres_python': 2.**-3,   # need binary representation
        'custom_fun': [infinite_axon_quasi_active],
        'custom_fun_args': [{'mu_factor_1': mu_factor_1, 'mu_factor_2': mu_factor_2}],
        'tstartms': 0.,
        'tstopms': 2000.,
    }

    folder = cell_name
    # Create cell
    cell = LFPy.Cell(**cell_parameters)

    if stimuli == 'synapse':
        cell, stim = insert_synapse(cell, input_idx, weight)
    elif stimuli == 'white_noise':
        cell, stim = insert_white_noise(cell, input_idx, weight)
    else:
        raise ValueError("Unknown input stimuli type")
    cell.simulate(rec_imem=True, rec_vmem=True)
    keep_tsteps = len(cell.tvec)/2


    sim_name = '%s_%s_%d_%1.1f_%1.1f' % (cell_name, stimuli, input_idx, mu_factor_1, mu_factor_2)
    np.save(join(folder, 'tvec_%s_%s.npy' % (cell_name, stimuli)), cell.tvec[-keep_tsteps:] - cell.tvec[-keep_tsteps])
    np.save(join(folder, 'imem_%s.npy' % sim_name), cell.imem[:, -keep_tsteps:])
    np.save(join(folder, 'vmem_%s.npy' % sim_name), cell.vmem[:, -keep_tsteps:])
    np.save(join(folder, 'xstart_%s.npy' % cell_name), cell.xstart)
    np.save(join(folder, 'ystart_%s.npy' % cell_name), cell.ystart)
    np.save(join(folder, 'zstart_%s.npy' % cell_name), cell.zstart)
    np.save(join(folder, 'xend_%s.npy' % cell_name), cell.xend)
    np.save(join(folder, 'yend_%s.npy' % cell_name), cell.yend)
    np.save(join(folder, 'zend_%s.npy' % cell_name), cell.zend)
    np.save(join(folder, 'xmid_%s.npy' % cell_name), cell.xmid)
    np.save(join(folder, 'ymid_%s.npy' % cell_name), cell.ymid)
    np.save(join(folder, 'zmid_%s.npy' % cell_name), cell.zmid)
    np.save(join(folder, 'diam_%s.npy' % cell_name), cell.diam)

    cell.imem = cell.imem[:, -keep_tsteps:]
    cell.vmem = cell.vmem[:, -keep_tsteps:]
    cell.tvec = cell.tvec[-keep_tsteps:] - cell.tvec[-keep_tsteps]

    return cell

if __name__ == '__main__':
    cell_name = 'infinite_neurite'
    return_cell(cell_name)