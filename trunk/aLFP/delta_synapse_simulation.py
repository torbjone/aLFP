#!/usr/bin/env python
import os
import LFPy
import numpy as np
import neuron
import sys
try:
    from ipdb import set_trace
except:
    pass
if not os.environ.has_key('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
    at_stallo = True
else:
    at_stallo = False

import pylab as plt
from os.path import join
import cPickle
import pickle
import aLFP
import scipy.fftpack as ff
import scipy.signal
import tables

plt.rcParams.update({'font.size' : 8,
    'figure.facecolor' : '1',
    'wspace' : 0.5, 'hspace' : 0.5})
#np.random.seed(1234)

def run_delta_synapse_simulation(cell_params, conductance_list, ofolder, model_path,
                                 ntsteps, synaptic_params, simulation_idx):
   # Excitatory synapse parameters:
    synapseParameters = {
        'e' : 0,   
        'syntype' : 'ExpSyn',      #conductance based exponential synapse
        'tau' : .1,                #Time constant, rise           #Time constant, decay
        'weight' : 0.001,           #Synaptic weight
        'color' : 'r',              #for pl.plot
        'marker' : '.',             #for pl.plot
        'record_current' : True,    #record synaptic currents
        }
    vss = -77
    cell = LFPy.Cell(**cell_params)

    spiketimes_dict = make_input_spiketrain(cell, **synaptic_params)
    for conductance_type in conductance_list:
        neuron.h('forall delete_section()')
        neuron.h('secondorder=2')
        del cell
        cell_params['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                      join(model_path, 'biophys3_%s.hoc' % conductance_type)]        

        cell_params['v_init'] = -77
        cell = LFPy.Cell(**cell_params)
        if len(synaptic_params['section']) == 3:
            sim_name = '%s_homogeneous_sim_%d' %(conductance_type, simulation_idx)
        elif len(synaptic_params['section']) == 1:
            sim_name = '%s_%s_sim_%d' %(conductance_type, synaptic_params['section'][0], simulation_idx)
        else:
            raise RuntimeError, "Wrong synaptic_params"
        print sim_name
        if conductance_type in ['passive_vss', 'Ih_linearized']:
            for comp_idx, sec in enumerate(cell.allseclist):
                for seg in sec:
                    exec('seg.vss_%s = %g'% (conductance_type, vss))
        else:
            pass

        set_input_spiketrain(cell, synapseParameters, spiketimes_dict)
        cell.simulate(rec_imem=True, rec_isyn=True, rec_vmem=True)

        mapping = np.load(join(ofolder, 'mapping.npy'))
        sig = 1000 * np.dot(mapping, cell.imem)
        sig_psd, freqs = aLFP.find_LFP_PSD(sig, (cell.tvec[1] - cell.tvec[0])/1000.)
        np.save(join(ofolder, 'signal_psd_%s.npy' %sim_name), sig_psd)
        np.save(join(ofolder, 'signal_%s.npy' %sim_name), sig)
        np.save(join(ofolder, 'somav_%s.npy' %sim_name), cell.somav)
        np.save(join(ofolder, 'imem_%s.npy' %sim_name), cell.imem)
        np.save(join(ofolder, 'vmem_%s.npy' %sim_name), cell.vmem)

    np.save(join(ofolder, 'freqs.npy'), freqs)
    np.save(join(ofolder, 'tvec.npy'), cell.tvec)
        


def delta_synapse_PSD(ofolder, conductance_type):

    plt.close('all')
    sim_name = conductance_type
    
    f = tables.openFile(join(ofolder, 'signal_%s.h5' %sim_name), mode = "r")
    tvec = f.root.tvec[:]
    y = f.root.somav[:]
    f.close()
    
    sample_freq = ff.fftfreq(len(tvec), d=(tvec[1] - tvec[0])/1000.)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(y)[pidxs]
    amp = np.abs(Y)/len(Y)
    angle = np.angle(Y)
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.plot(tvec, y, lw=1, color='k')

    ax2 = fig.add_subplot(122, ylim=[1e-6, 1e0], xlim=[1e0, 1e5])
    ax2.loglog(freqs, amp, lw=1, color='k')
    
    plt.savefig('testing_%s.png' %sim_name)


    
def make_input_spiketrain(cell, section, n, spTimesFun, args):
    """ Make and return spiketimes for each compartment that receives input """
    cell_idxs = cell.get_rand_idx_area_norm(section=section, nidx=n)
    spiketimes_dict = {}
    for idx in cell_idxs:
        spiketimes_dict[str(idx)] = spTimesFun(args[0], args[1], args[2], args[3])[0]
    return spiketimes_dict

def set_input_spiketrain(cell, synparams, spiketimes_dict):
    """ Find spiketimes """
    for idx, spiketrain in spiketimes_dict.items():
        synparams.update({'idx' : int(idx)})
        s = LFPy.Synapse(cell, **synparams)
        s.set_spike_times(spiketrain)

