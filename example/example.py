#!/usr/bin/env python
'''
Simple Hay simulation with one synaptic input, and extracellular electrodes
'''
from os.path import join
import numpy as np
import pylab as plt
import neuron
import LFPy
from hay_model.hay_active_declarations import active_declarations
nrn = neuron.h
import scipy.fftpack as ff

def return_freq_and_psd(tvec, sig):
    """ Returns the power and freqency of the input signal"""
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    timestep = (tvec[1] - tvec[0])/1000. if type(tvec) in [list, np.ndarray] else tvec
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig, axis=1)[:, pidxs[0]]
    power = np.abs(Y)**2/Y.shape[1]
    return freqs, power


def example_synapse(synaptic_y_pos=0, conductance_type='active', weight=0.001):
    """
    Runs a NEURON simulation and returns an LFPy cell object for a single synaptic input.
    :param synaptic_y_pos: position along the apical dendrite where the synapse is inserted.
    :param conductance_type: Either 'active' or 'passive'. If 'active' all original ion-channels are included,
           if 'passive' they are all removed, yielding a passive cell model.
    :param weight: Strength of synaptic input.
    :param input_spike_train: Numpy array containing synaptic spike times
    :return: cell object where cell.imem gives transmembrane currents, cell.vmem gives membrane potentials.
             See LFPy documentation for more details and examples.
    """

    #  Making cell
    model_path = join('hay_model')
    neuron.load_mechanisms(join(model_path, 'mod'))

    sim_time = 100

    cell_parameters = {
        'morphology': join(model_path, 'cell1.hoc'),
        'v_init': -65,
        'passive': False,
        'nsegs_method': 'lambda_f',
        'lambda_f': 100,
        'timeres_NEURON': 2**-4,  # Should be a power of 2
        'timeres_python': 2**-4,
        'tstartms': -200,
        'tstopms': sim_time,
        'custom_code': [join(model_path, 'custom_codes.hoc')],
        'custom_fun': [active_declarations],
        'custom_fun_args': [{'conductance_type': conductance_type}],
    }
    cell = LFPy.Cell(**cell_parameters)
    input_idx = cell.get_closest_idx(x=0., y=synaptic_y_pos, z=0.)

    cell, synapse = make_synapse(cell, weight, input_idx)
    cell.simulate(rec_imem=True, rec_vmem=True)

    plot_electrode_signal(cell, input_idx, False)


def example_white_noise(synaptic_y_pos=0, conductance_type='active', weight=0.001):
    """
    Runs a NEURON simulation and returns an LFPy cell object for a single synaptic input.
    :param synaptic_y_pos: position along the apical dendrite where the synapse is inserted.
    :param conductance_type: Either 'active' or 'passive'. If 'active' all original ion-channels are included,
           if 'passive' they are all removed, yielding a passive cell model.
    :param weight: Strength of synaptic input.
    :param input_spike_train: Numpy array containing synaptic spike times
    :return: cell object where cell.imem gives transmembrane currents, cell.vmem gives membrane potentials.
             See LFPy documentation for more details and examples.
    """

    #  Making cell
    model_path = join('hay_model')
    neuron.load_mechanisms(join(model_path, 'mod'))

    # Repeat same stimuli and save only the last
    repeats = 2
    sim_time = 1000

    cell_parameters = {
        'morphology': join(model_path, 'cell1.hoc'),
        'v_init': -65,
        'passive': False,
        'nsegs_method': 'lambda_f',
        'lambda_f': 100,
        'timeres_NEURON': 2**-3,  # Should be a power of 2
        'timeres_python': 2**-3,
        'tstartms': 0,
        'tstopms': sim_time * repeats,
        'custom_code': [join(model_path, 'custom_codes.hoc')],
        'custom_fun': [active_declarations],
        'custom_fun_args': [{'conductance_type': conductance_type}],
    }
    cell = LFPy.Cell(**cell_parameters)
    input_idx = cell.get_closest_idx(x=0., y=synaptic_y_pos, z=0.)

    cell, synapse, noiseVec = make_white_noise(cell, weight, input_idx)
    cell.simulate(rec_imem=True, rec_vmem=True)
    if not repeats is None:
        cut_off_idx = (len(cell.tvec) - 1) / repeats
        cell.tvec = cell.tvec[-cut_off_idx:] - cell.tvec[-cut_off_idx]
        cell.imem = cell.imem[:, -cut_off_idx:]
        cell.vmem = cell.vmem[:, -cut_off_idx:]

    plot_electrode_signal(cell, input_idx, True)


def make_synapse(cell, weight, input_idx):

    input_spike_train = np.array([20.])
    synapse_parameters = {
        'idx': input_idx,
        'e': 0.,
        'syntype': 'ExpSyn',
        'tau': 10.,
        'weight': weight,
        'record_current': True,
    }
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(input_spike_train)
    return cell, synapse



def make_white_noise(cell, weight, input_idx):
    max_freq = 510
    plt.seed(1234)
    tot_ntsteps = round((cell.tstopms - cell.tstartms) / cell.timeres_NEURON + 1)
    input_array = np.zeros(tot_ntsteps)
    tvec = np.arange(tot_ntsteps) * cell.timeres_NEURON
    for freq in xrange(1, max_freq + 1):
        input_array += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())
    input_array *= weight
    noiseVec = neuron.h.Vector(input_array)

    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print "Input inserted in ", sec.name()
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if syn is None:
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noiseVec.play(syn._ref_amp, cell.timeres_NEURON)
    return cell, syn, noiseVec


def plot_electrode_signal(cell, input_idx, psd=False):
    #  Making extracellular electrode
    elec_x = np.array([25.])
    elec_y = np.zeros(len(elec_x))
    elec_z = np.zeros(len(elec_x))

    electrode_parameters = {
        'sigma': 0.3,              # extracellular conductivity
        'x': elec_x,        # x,y,z-coordinates of contact points
        'y': elec_y,
        'z': elec_z,
    }
    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
    electrode.calc_lfp()

    cell_plot_idx = 0
    plt.subplots_adjust(hspace=0.3)  # Adjusts the vertical distance between panels.
    plt.subplot(132, aspect='equal')
    plt.axis('off')
    [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], c='k') for idx in xrange(cell.totnsegs)]
    [plt.plot(electrode.x[idx], electrode.y[idx], 'bD') for idx in range(len(electrode.x))]
    plt.plot(cell.xmid[input_idx], cell.ymid[input_idx], 'y*', markersize=10)

    if psd:
        x, y_v = return_freq_and_psd(cell.tvec, cell.vmem)
        x, y_i = return_freq_and_psd(cell.tvec, cell.imem)
        x, y_lfp = return_freq_and_psd(cell.tvec, 1000*electrode.LFP)
        scale = 'log'

        plot_dict = {'xlabel': 'Hz',
                     'xscale': 'log',
                     'yscale': 'log',
                     'xlim': [1, 500],
                     }
    else:
        x = cell.tvec
        y_v = cell.vmem
        y_i = cell.imem
        y_lfp = 1000*electrode.LFP
        plot_dict = {'xlabel': 'Time [ms]',
                     'xscale': 'linear',
                     'yscale': 'linear',
                     'xlim': [0, cell.tvec[-1]],
        }

    plt.subplot(231, title='Membrane potential', ylabel='mV', **plot_dict)
    plt.plot(x, y_v[cell_plot_idx, :], color='k', lw=2)
    plt.subplot(234, title='Transmembrane currents', ylabel='nA', **plot_dict)
    plt.plot(x, y_i[cell_plot_idx, :], color='k', lw=2)
    plt.subplot(133, title='Extracellular potential', ylabel='$\mu V$', **plot_dict)
    [plt.plot(x, y_lfp[idx, :], c='b', lw=2) for idx in range(len(electrode.x))]
    plt.savefig('example_fig.png')

if __name__ == '__main__':
    example_synapse(synaptic_y_pos=1000, conductance_type='active', weight=0.01)
    # example_white_noise(synaptic_y_pos=0, conductance_type='passive', weight=0.01)
