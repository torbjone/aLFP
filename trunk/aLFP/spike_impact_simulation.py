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

def delta_synapse_spike_simulation(cell_params, conductance_list, ofolder, model_path,
                                 ntsteps, synaptic_params, simulation_idx):
   # Excitatory synapse parameters:

    syn_strength = 0.001
    synapseParameters = {
        'e' : 0,   
        'syntype' : 'ExpSyn',      #conductance based exponential synapse
        'tau' : .1,                #Time constant, rise           #Time constant, decay
        'weight' : syn_strength,           #Synaptic weight
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
        if len(synaptic_params['section']) == 2:
            sim_name = '%s_homogeneous_sim_%d_%g' %(conductance_type, simulation_idx, syn_strength)
        elif len(synaptic_params['section']) == 1:
            sim_name = '%s_%s_sim_%d_%g' %(conductance_type, synaptic_params['section'][0],
                                           simulation_idx, syn_strength)
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

        plt.close('all')
        plt.plot(cell.tvec, cell.vmem[0,:])
        plt.savefig('spike_%s.png' % conductance_type)

        #set_trace()
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

def plot_PSD_spike_time_window(ifolder, conductance_list, elec_x, elec_y, elec_z):

    freqs = np.load(join(ifolder, 'freqs.npy'))    
    tvec = np.load(join(ifolder, 'tvec.npy'))
    xmid = np.load(join(ifolder, 'xmid.npy' ))
    ymid = np.load(join(ifolder, 'ymid.npy' ))
    zmid = np.load(join(ifolder, 'zmid.npy' ))
    xstart = np.load(join(ifolder, 'xstart.npy' ))
    ystart = np.load(join(ifolder, 'ystart.npy' ))
    zstart = np.load(join(ifolder, 'zstart.npy' ))
    xend = np.load(join(ifolder, 'xend.npy' ))
    yend = np.load(join(ifolder, 'yend.npy' ))
    zend = np.load(join(ifolder, 'zend.npy' ))    
    diam = np.load(join(ifolder, 'diam.npy'))
    imem_dict = {}
    imem_psd_dict = {}
    vmem_dict = {}
    vmem_psd_dict = {}
    signal_dict = {}
    signal_psd_dict = {}

    imem_filt_dict = {}
    vmem_filt_dict = {}
    vmem_filt_psd_dict = {}
    signal_filt_dict = {}


    time_window = [0, 1000]
    time_idx_1 = np.argmin(np.abs(tvec - time_window[0]))
    time_idx_2 = np.argmin(np.abs(tvec - time_window[1]))
    tvec = tvec[time_idx_1:time_idx_2]
    timestep = (tvec[1] - tvec[0])/1000.
    srate = 1./timestep
    
    b, a = scipy.signal.butter(N=6, Wn=np.array([2000., 8000.])/srate/2., btype='low')
    

    for conductance_type in conductance_list:
        sim_name = '%s_homogeneous_sim_%d' %(conductance_type, 0)
        imem_dict[conductance_type] = np.load(join(ifolder, 'imem_%s.npy' 
                                                   %(sim_name)))[:,time_idx_1:time_idx_2]
        imem_psd_dict[conductance_type], freqs = aLFP.find_LFP_PSD(imem_dict[conductance_type], timestep)
        vmem_dict[conductance_type] = np.load(join(ifolder, 'vmem_%s.npy' 
                                                   %(sim_name)))[:,time_idx_1:time_idx_2]
        vmem_psd_dict[conductance_type], freqs = aLFP.find_LFP_PSD(vmem_dict[conductance_type], timestep)
        #signal_dict[conductance_type] = np.load(join(ifolder, 'signal_%s.npy' 
        #                                           %(sim_name)))[:,time_idx_1:time_idx_2]
        #signal_psd_dict[conductance_type], freqs = aLFP.find_LFP_PSD(signal_dict[conductance_type], timestep)

        vmem_filt_dict[conductance_type] = scipy.signal.filtfilt(b, a, vmem_dict[conductance_type], axis=1)
        vmem_filt_psd_dict[conductance_type], freqs = aLFP.find_LFP_PSD(vmem_filt_dict[conductance_type], timestep)
        imem_filt_dict[conductance_type] = scipy.signal.filtfilt(b, a, imem_dict[conductance_type], axis=1)
        #signal_filt_dict[conductance_type] = scipy.signal.filtfilt(b, a, signal_dict[conductance_type], axis=1)
        
    plt.subplot(221, ylabel='Vm', xlabel='ms', ylim=[-10, 20])
    plt.plot(tvec, vmem_dict['active'][0,:] - vmem_dict['active'][0,0], color='r')
    #plt.plot(tvec, vmem_dict['Ih_linearized'][0,:] - vmem_dict['Ih_linearized'][0,0], color='g')
    plt.plot(tvec, vmem_dict['passive_vss'][0,:] - vmem_dict['passive_vss'][0,0], color='k')

    plt.subplot(222, ylabel='Vm', xlim=[0,600], yscale='log', xlabel='Hz', ylim=[1e-4, 1e1])
    plt.plot(freqs, vmem_psd_dict['active'][0,:], color='r')
    #plt.plot(freqs, vmem_psd_dict['Ih_linearized'][0,:], color='g')
    plt.plot(freqs, vmem_psd_dict['passive_vss'][0,:], color='k')


    ## plt.subplot(243, ylabel='Vm', xlim=[0,1000], ylim=[-10, 20], xlabel='ms')
    ## plt.plot(tvec, vmem_filt_dict['active'][0,:] - vmem_filt_dict['active'][0,0], color='r')
    ## plt.plot(tvec, vmem_filt_dict['Ih_linearized'][0,:] - vmem_filt_dict['Ih_linearized'][0,0], color='g')
    ## plt.plot(tvec, vmem_filt_dict['passive_vss'][0,:] - vmem_filt_dict['passive_vss'][0,0], color='k')

    ## plt.subplot(244, ylabel='Vm', xlim=[0,600], yscale='log', xlabel='Hz', ylim=[1e-4, 1e1])
    ## plt.plot(freqs, vmem_filt_psd_dict['active'][0,:], color='r')
    ## plt.plot(freqs, vmem_filt_psd_dict['Ih_linearized'][0,:], color='g')
    ## plt.plot(freqs, vmem_filt_psd_dict['passive_vss'][0,:], color='k')

    
    plt.subplot(223, ylabel='pA', xlabel='ms')
    plt.plot(tvec, imem_dict['active'][0,:] - imem_dict['active'][0,0], color='r')
    #plt.plot(tvec, imem_dict['Ih_linearized'][0,:] - imem_dict['Ih_linearized'][0,0], color='g')
    plt.plot(tvec, imem_dict['passive_vss'][0,:] - imem_dict['passive_vss'][0,0], color='k')

    plt.subplot(224, ylabel='pA', xlim=[0,2000], yscale='log', xlabel='Hz')
    plt.plot(freqs, imem_psd_dict['active'][0,:], color='r', label='Active')
    #plt.plot(freqs, imem_psd_dict['Ih_linearized'][0,:], color='g', label='Ih_linearized')
    plt.plot(freqs, imem_psd_dict['passive_vss'][0,:], color='k', label='passive')


    ## plt.subplot(247, ylabel='pA', xlim=[0,1000], xlabel='ms')
    ## plt.plot(tvec, imem_filt_dict['active'][0,:] - imem_filt_dict['active'][0,0], color='r', label='Active')
    ## plt.plot(tvec, imem_filt_dict['Ih_linearized'][0,:] - imem_filt_dict['Ih_linearized'][0,0], color='g', label='Ih_linearized')
    ## plt.plot(tvec, imem_filt_dict['passive_vss'][0,:] - imem_filt_dict['passive_vss'][0,0], color='k', label='passive')

    
    plt.legend()
    plt.savefig('imem_vmem.png')

    ## lines = []
    ## line_names = []

    ## l, = plt.plot([0,0], color='r')
    ## lines.append(l)
    ## line_names.append('active')

    ## l, = plt.plot([0,0], color='k')
    ## lines.append(l)
    ## line_names.append('passive')

    
    ## plt.close('all')
    ## fig = plt.figure(figsize=[15, 8])
    ## fig.suptitle('r: Radial distance in $\mu m$. h: Height in $\mu m$')
    ## fig.subplots_adjust(hspace=0.6, wspace=0.6)
    ## radiuses = np.array([10., 25, 50., 100., 250., 500., 1000., 2500., 5000.])
    ## heights = np.array([0., 500., 1000.])
    ## idx = 0
    ## for hidx, height in enumerate(heights):
    ##     for ridx, radius in enumerate(radiuses):
    ##         plot_number = len(radiuses) * (len(heights) - hidx - 1) + ridx + 1
    ##         ax = fig.add_subplot(len(heights), len(radiuses), plot_number, title='r: %g h: %g' %(radius, height), xlabel='ms', ylabel='Extracellular Amplitude', xticks=[0, 500, 1000])
    ##         ax.plot(tvec, signal_dict['active'][idx], color='r')
    ##         ax.plot(tvec, signal_dict['passive_vss'][idx], color='k')
    ##         for angle_idx in xrange(30):
    ##             idx += 1

    ## fig.legend(lines, line_names)
    ## fig.savefig('signal_test.png', dpi=150)
    
    ## plt.close('all')
    ## fig = plt.figure(figsize=[15, 8])
    ## fig.suptitle('r: Radial distance in $\mu m$. h: Height in $\mu m$')
    ## plt.subplots_adjust(hspace=0.6, wspace=0.6)
    ## radiuses = np.array([10., 25, 50., 100., 250., 500., 1000., 2500., 5000.])
    ## heights = np.array([0., 500., 1000.])
    ## idx = 0
    ## for hidx, height in enumerate(heights):
    ##     for ridx, radius in enumerate(radiuses):
    ##         plot_number = len(radiuses) * (len(heights) - hidx - 1) + ridx + 1
    ##         plt.subplot(len(heights), len(radiuses), plot_number, title='r: %g h: %g' %(radius, height),
    ##                     xlim=[0, 1e3], yscale='log', xlabel='Hz', ylabel='Extracellular Amplitude', ylim=[1e-8, 1e0], xticks=[0, 500, 1000])
    ##         plt.plot(freqs, signal_psd_dict['active'][idx], color='r')
    ##         plt.plot(freqs, signal_psd_dict['passive_vss'][idx], color='k')
    ##         for angle_idx in xrange(30):
    ##             idx += 1
    ## fig.legend(lines, line_names)        
    ## plt.savefig('signal_psd_test.png', dpi=150)
