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
import random
import matplotlib.mlab as mlab


plt.rcParams.update({'font.size' : 8,
    'figure.facecolor' : '1',
    'wspace' : 0.5, 'hspace' : 0.5})
#np.random.seed(1234)



def plot_single_sigs(folder):

    conductance_type = 'active'
    input_pos = 'dend'
    correlations = [0.0, 1.0]
    syn_strength = 0.01
    sims = np.arange(0, 15)
    spike_std_factor = 6
    
    for correlation in correlations:
        num_spiking_cells = 0
        for sim in sims:
            name = 'signal_%s_%s_%1.2f_%1.3f_sim_%d.npy'% (conductance_type, input_pos, correlation,
                                                           syn_strength, sim)
            sig = np.load(join(folder, name))
            sig_max_idx = np.argmax(np.max(sig, axis=1) - np.min(sig, axis=1))
            

            max_sig = sig[sig_max_idx,:] - np.average(sig[sig_max_idx, :])
            spike_idxs = np.where(np.abs(max_sig) > spike_std_factor * np.std(max_sig))[0]
            plt.close('all')
            if len(spike_idxs) > 0:
                is_spiking = True
                num_spiking_cells += 1
                try:
                    plt.plot(spike_idxs, np.zeros(len(spike_idxs)), 'rD', lw=1)
                except:
                    set_trace()
            else:
                is_spiking = False
            
            plt.title('%s %s %1.2f %1.3f sim:%d, spikes: %s'% (conductance_type, input_pos, correlation,
                                                               syn_strength, sim, is_spiking))
            plt.plot(max_sig, 'k', lw=1)
            plt.plot([0, 1000], [spike_std_factor*np.std(sig[sig_max_idx,:]),
                                 spike_std_factor*np.std(sig[sig_max_idx,:])], 'g')
            plt.plot([0, 1000], [-spike_std_factor*np.std(sig[sig_max_idx,:]),
                                 -spike_std_factor*np.std(sig[sig_max_idx,:])], 'g')
            #[plt.plot(sig[idx,:] - sig[idx, 0]) for idx in xrange(sig.shape[0])]
            plt.savefig('signal_debug_2_%s_%s_%1.2f_%1.3f_sim_%d.png'% (conductance_type, input_pos,
                                                                      correlation, syn_strength, sim))
        print "%s with correlation %1.2f has %d spiking out of %d" % (input_pos, correlation,
                                                                      num_spiking_cells, len(sims))

def plot_somavs(folder, syn_strength):
    all_files = os.listdir(folder)
    all_files = [f for f in all_files if ('somav' in f) and ('_%1.3f_' %syn_strength in f)
                                                        and (not 'inhib' in f)]

    correlations = [0., 1.0]
    input_positions = ['dend', 'apic']
    conductance_list = ['active', 'Ih_linearized',  'passive_vss']
    clr = lambda idx, L: plt.cm.spectral(int(256. * idx/(L - 1.)))
    for correlation in correlations:
        for input_pos in input_positions:
            for cond_number, conductance_type in enumerate(conductance_list):
                plt.close('all')
                fig = plt.figure(figsize=[12,6])
                fig.suptitle('Conductance: %s, Input: %s, Correlation: %1.2f, Synapse strenght: %1.3f'
                             % (conductance_type, input_pos, correlation, syn_strength))
                ax1 = fig.add_subplot(211, xlabel='ms', ylabel='Somatic Vm', ylim=[-77, -60])
                ax2 = fig.add_subplot(212, xlabel='ms', ylabel='Shifted Somatic Vm', ylim=[-10, 10])
                current_list = [f for f in all_files if (input_pos in f) 
                                                    and ('%1.2f_' % correlation in f)
                                                    and (conductance_type in f)]
                lines = []
                line_names = []
                for idx, name in enumerate(current_list):
                    sim_number = name.split('_')[-1][:-4]
                    vm = np.load(join(folder, name))
                    if np.max(vm) > -40:
                        label='Spinking:%s' %sim_number
                    else:
                        label='Not spiking:%s' %sim_number
                    ax1.plot(vm, color=clr(idx, len(current_list)))
                    l, = ax2.plot(vm - vm[0], color=clr(idx, len(current_list)))
                    lines.append(l)
                    line_names.append(label)
                fig.legend(lines, line_names, frameon=False)
                fig.savefig('Vm_%s_%s_%1.2f_%1.3f.png' % (conductance_type, input_pos,
                                                          correlation, syn_strength), dpi=150)
    
def compare_psd_of_input():

    conductance_list = ['active', 'Ih_linearized', 'passive_vss']
    pos_list = ['apic', 'dend']
    syn_list = [0.001, 0.015, 0.02]
    conductance_clr = lambda conductance_idx: plt.cm.jet(int(256. * conductance_idx/
                                                             (len(conductance_list) - 1.)))

    stem = 'signal_%s_%s_1.00_sim_0_%1.3f.npy'
    vmem_stem = 'somav_%s_%s_1.00_sim_0_%1.3f.npy'
    plt.close('all')
    numplots = len(syn_list) * len(pos_list)
    fig = plt.figure(figsize=[7,12])
    fig.subplots_adjust(hspace=0.6, wspace=0.5)
    plot_idx = 0
    divide_into_welch = 6
    line_names = []
    lines = []
    for input_pos in pos_list:
        for syn_strength in syn_list:
            title_ax = fig.add_subplot(numplots, 1, plot_idx + 1, xticks=[], yticks=[], frameon=False,
                                       title='Input pos: %s, Synaptic strength: %1.3f'
                                       %(input_pos, syn_strength))
            
            ax1 = fig.add_subplot(numplots, 3, 3*plot_idx + 1, ylabel='Shifted signal', xlabel='ms')
            ax2 = fig.add_subplot(numplots, 3, 3*plot_idx + 2, ylabel='Power', xlabel='Hz',
                                  ylim=[1e-10, 1e-6])
            ax3 = fig.add_subplot(numplots, 3, 3*plot_idx + 3, ylabel='Memb. pot.', xlabel='ms')            
            ax2.grid(True)

            for cond_number, conductance_type in enumerate(conductance_list):
                filename = stem %(conductance_type, input_pos, syn_strength)
                sig = np.load('hay/%s' % filename)
                vmem = np.load('hay/%s' % vmem_stem %(conductance_type, input_pos, syn_strength))
                ax3.plot(vmem[:], color=conductance_clr(cond_number))
                psd, freqs = mlab.psd(sig[1,:], Fs=1000., NFFT=int(1001./divide_into_welch),
                                            noverlap=int(1001./divide_into_welch/2),
                                            window=plt.window_hanning, detrend=plt.detrend_mean)

                l, = ax1.plot(sig[1,:] - sig[1,0], color=conductance_clr(cond_number))
                ax2.loglog(freqs, psd, color=conductance_clr(cond_number))
                if conductance_type not in line_names:
                    line_names.append(conductance_type)
                    lines.append(l)    
            plot_idx += 1

    fig.legend(lines, line_names, frameon=False, fontsize=12)
    fig.savefig('input_study.png', dpi=150)
                
def population_size_summary(folder, conductance_list, elec_x, elec_y, elec_z, syn_strength,
                            center_idxs, population_radius):

    xmid = np.load(join(folder, 'xmid.npy'))
    ymid = np.load(join(folder, 'ymid.npy'))
    xstart = np.load(join(folder, 'xstart.npy'))
    ystart = np.load(join(folder, 'ystart.npy'))
    zstart = np.load(join(folder, 'zstart.npy'))
    xend = np.load(join(folder, 'xend.npy'))
    yend = np.load(join(folder, 'yend.npy'))
    zend = np.load(join(folder, 'zend.npy'))    
    diam = np.load(join(folder, 'diam.npy'))
    n_elecs = len(elec_z[center_idxs])
    elec_clr = lambda elec_idx: plt.cm.rainbow(int(256. * elec_idx/(n_elecs - 1.)))
    conductance_clr = lambda conductance_idx: plt.cm.jet(int(256. * conductance_idx/
                                                             (len(conductance_list) - 1.)))
    divide_into_welch = 8

    population_radii = np.arange(50, population_radius +1, 25)
    for input_pos in ['dend', 'apic']:
        for radius in population_radii:

            plt.close('all')
            fig = plt.figure(figsize=[7,10])
            fig.suptitle('Stimulation: %s, Population size: %d $\mu m$'
                         % (input_pos, int(radius)))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)
            ax_ = fig.add_subplot(1, 3, 1, frameon=False, xticks=[], yticks=[], ylim=[-300, 1300])
            
            for comp in xrange(len(xstart)):
                if comp == 0:
                    ax_.scatter(xmid[comp] - 100, ymid[comp], s=diam[comp], color='k')
                else:
                    ax_.plot([xstart[comp] - 100, xend[comp] - 100], [ystart[comp], yend[comp]],
                             lw=diam[comp], color='k')

            for elec in center_idxs:
                ax_.plot(elec_x[elec], elec_z[elec], 'o', color=elec_clr(elec))

            for cor_idx, correlation in enumerate([0, 1.0]):
                signal_dict = {}
                signal_psd_dict = {}
                signal_psd_welch_dict = {}

                for conductance_type in conductance_list:
                    stem='signal_%s_%s_%1.2f_%1.3f_total_pop_size_%04d' %(conductance_type, input_pos,
                                                                         correlation, syn_strength, int(radius))

                    signal_dict[conductance_type] = np.load(join(folder, '%s.npy' % stem))
                    signal_psd_dict[conductance_type], freqs = aLFP.find_LFP_PSD(
                        signal_dict[conductance_type][:,:], (1.0)/1000.)
                    
                    foo, freqs_welch = mlab.psd(signal_dict[conductance_type][0,:], Fs=1000.,
                                                NFFT=int(1001./divide_into_welch),
                                                noverlap=int(1001./divide_into_welch/2),
                                                window=plt.window_hanning)
                    signal_psd_welch_dict[conductance_type] = np.zeros((n_elecs, len(foo)))
                    for elec in center_idxs:
                        signal_psd_welch_dict[conductance_type][elec, :], freqs_welch = \
                          mlab.psd(signal_dict[conductance_type][elec,:], Fs=1000.,
                                NFFT=int(1001./divide_into_welch), noverlap=int(1001./divide_into_welch/2),
                                window=plt.window_hanning, detrend=plt.detrend_mean)
                for plotnumb, elec in enumerate(center_idxs):
                    ax_psd = fig.add_subplot(n_elecs, 3, 3*(n_elecs - plotnumb - 1) + 2 + cor_idx,
                                             ylim=[1e-4, 1e1], xlim=[1e0, 1e3])

                    ax_psd.tick_params(color=elec_clr(plotnumb))
                    for spine in ax_psd.spines.values():
                        spine.set_edgecolor(elec_clr(plotnumb))
                    ax_psd.spines['top'].set_visible(False)
                    ax_psd.spines['right'].set_visible(False)
                    ax_psd.get_xaxis().tick_bottom()
                    ax_psd.get_yaxis().tick_left()
                    ax_psd.grid(True)

                    if elec == n_elecs - 1:
                        ax_psd.set_title('Correlation = %1.2f' % correlation)

                    for cond_number, conductance_type in enumerate(conductance_list):
                        #set_trace()
                        #print conductance_type
                        #set_trace()
                        #ax_psd.loglog(freqs, signal_psd_dict[conductance_type][elec], lw=1,
                        #              color=conductance_clr(cond_number))
                        ax_psd.loglog(freqs_welch, np.sqrt(signal_psd_welch_dict[conductance_type][elec]),
                                      lw=1, color=conductance_clr(cond_number))
                    if elec == 0:
                        ax_psd.set_ylabel('$\mu V$')
                        ax_psd.set_xlabel('Hz')
            lines = []
            line_names = []
            for cond_number, conductance_type in enumerate(conductance_list):
                l, = ax_psd.loglog(0, 0, color=conductance_clr(cond_number))
                lines.append(l)
                line_names.append(conductance_type)
            fig.legend(lines, line_names, frameon=False)
            fig.savefig('population_size_summary_%s_%1.3f_%04d.png' %(input_pos, syn_strength,
                                                                      int(radius)), dpi=150)
            #plt.show()

def population_size_frequency_dependence(folder, conductance_list, input_pos, correlations, syn_strength,
                                         population_radius):

    pop_sizes = np.array([100, 200, population_radius])

    plt.close('all')
    fig = plt.figure(figsize=[5,7])
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    size_clrs = ['b', 'r', 'g']

    line_names = []
    lines = []
    divide_into_welch = 8
    for cond_number, conductance_type in enumerate(conductance_list):
        for cor_number, correlation in enumerate(correlations):
            plot_number = len(correlations) * cond_number + cor_number + 1
            ax = fig.add_subplot(3, 2, plot_number, title='%s $c_{in}$=%g'
                                 %(conductance_type, correlation), xlabel='Hz', ylabel='Amp', ylim=[1e-3, 1e0])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.grid(True)

            for size_idx, size in enumerate(pop_sizes):
            
                stem = 'signal_%s_%s_%1.2f_%1.3f_total_pop_size_%04d' %(conductance_type, input_pos,
                                                                        correlation, syn_strength, size)
                signal = np.load(join(folder, '%s.npy' % stem))
                #signal_psd, freqs = aLFP.find_LFP_PSD(np.array([signal[1,:]]), (1.0)/1000.)
                welch_psd, freqs_welch = mlab.psd(signal[1,:], Fs=1000., NFFT=int(1001/divide_into_welch),
                                                  noverlap=int(1001/divide_into_welch/2),
                                                  detrend=plt.detrend_mean, window=plt.window_hanning)
                #l, = ax.loglog(freqs[:], signal_psd[0,:], color=size_clrs[size_idx], alpha=0.5)
                l, = ax.loglog(freqs_welch[:], np.sqrt(welch_psd[:]), color=size_clrs[size_idx])
                if cond_number == 0 and cor_number == 1.:
                    lines.append(l)
                    line_names.append('%d $\mu m$' %size)
    fig.legend(lines, line_names, frameon=False, title='Population sizes', ncol=3)
    fig.savefig('population_size_freq_%s_%1.3f.png' %(input_pos, syn_strength))


def population_size_amp_dependence(folder, conductance_list, input_pos, correlations, syn_strength,
                                   population_radius):


    plot_freqs = np.array([8., 32., 400.])
    population_radii = np.arange(50, population_radius +1, 25)
    
    plt.close('all')
    fig = plt.figure(figsize=[5,7])
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    freq_clrs = ['r', 'y', 'orange', 'g']
    line_names = []
    lines = []
    divide_into_welch = 8
    for cond_number, conductance_type in enumerate(conductance_list):
        for cor_number, correlation in enumerate(correlations):
            plot_number = len(correlations) * cond_number + cor_number + 1
            ax = fig.add_subplot(3, 2, plot_number, title='%s $c_{in}$=%g'
                                 %(conductance_type, correlation), xlabel='Pop. size',
                                 ylabel='Amp', ylim=[1e-3, 1e0], yscale='log')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.grid(True)
            
            amp_for_size_at_freq = np.zeros((len(plot_freqs), len(population_radii)))
            amp_for_size_at_freq_welch = np.zeros((len(plot_freqs), len(population_radii)))
            for size_idx, size in enumerate(population_radii):
                stem = 'signal_%s_%s_%1.2f_%1.3f_total_pop_size_%04d' %(conductance_type, input_pos,
                                                                        correlation, syn_strength, size)
                signal = np.load(join(folder, '%s.npy' % stem))
                signal_psd, freqs = aLFP.find_LFP_power(np.array([signal[1,:]]), (1.0)/1000.)
                welch_psd, freqs_welch = mlab.psd(signal[1,:], Fs=1000., NFFT=int(1001./divide_into_welch),
                                                  noverlap=int(1001./divide_into_welch/2),
                                                  window=plt.window_hanning, detrend=plt.detrend_mean)
                freq_idxs = np.array([np.argmin(np.abs(freqs - plot_freq)) for plot_freq in plot_freqs])
                welch_freq_idxs = np.array([np.argmin(np.abs(freqs_welch - plot_freq))
                                            for plot_freq in plot_freqs])
                print freqs_welch[welch_freq_idxs]       
                amp_for_size_at_freq[:, size_idx] = signal_psd[0,freq_idxs]
                amp_for_size_at_freq_welch[:, size_idx] = np.sqrt(welch_psd[welch_freq_idxs])
                
            for freq_idx, freq in enumerate(plot_freqs):
                #l, = ax.plot(population_radii, amp_for_size_at_freq[freq_idx], '--', color=freq_clrs[freq_idx])
                l, = ax.plot(population_radii, amp_for_size_at_freq_welch[freq_idx], color=freq_clrs[freq_idx])
                if cond_number == 0 and cor_number == 1.:
                    lines.append(l)
                    line_names.append('%d Hz' %freqs_welch[welch_freq_idxs[freq_idx]])
    fig.legend(lines, line_names, frameon=False, ncol=3, title='Frequency')
    fig.savefig('population_size_amp_%s_%1.3f.png' %(input_pos, syn_strength))


def plot_decay_with_dist_from_pop(folder, elec_x, elec_y, elec_z, correlation, syn_strength,
                                  input_pos, lateral_idxs, conductance_list, population_radius):

    conductance_clr = lambda cond_number: plt.cm.jet(int(256. * cond_number/(len(conductance_list) - 1.)))
    divide_into_welch = 8

    numcols = 7
    numrows = 3
    elecnum = 0

    sig_dict = {}
    sig_psd_dict = {}
    lines = []
    line_names = []
    for cond_number, conductance_type in enumerate(conductance_list):
        stem = 'signal_%s_%s_%1.2f_%1.3f_total_pop_size_%04d' %(conductance_type, input_pos,
                                                                correlation, syn_strength, population_radius)
        sig_dict[conductance_type] = np.load(join(folder, '%s.npy' %stem))
        sig_psd_dict[conductance_type], freqs = aLFP.find_LFP_PSD(sig_dict[conductance_type], timestep=1./1000)
        l, = plt.plot(0,0, color=conductance_clr(cond_number))
        lines.append(l)
        line_names.append(conductance_type)
        plt.close('all')
    fig = plt.figure(figsize=[12,8])
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
        
    plotnum = 1
    for col in xrange(numcols):
        for row in xrange(numrows):
            try:
                elecnum = lateral_idxs[plotnum - 1]
            except:
                pass
            ax = fig.add_subplot(numrows, numcols, plotnum, ylim=[1e-11, 1e0], xlabel='Hz', ylabel='Power',
                                 title='X pos: %d $\mu m$' %(elec_x[elecnum]), xlim=[1e0, 1e3])
            ax.grid(True)
            for cond_number, conductance_type in enumerate(conductance_list):
                welch_psd, freqs_welch = mlab.psd(sig_dict[conductance_type][elecnum,:], Fs=1000.,
                                                NFFT=int(1001./divide_into_welch),
                                                noverlap=int(1001./divide_into_welch/2),
                                                window=plt.window_hanning, detrend=plt.detrend_mean)
                ax.loglog(freqs, sig_psd_dict[conductance_type][elecnum]**2, color=conductance_clr(cond_number),
                          alpha=1)
                #ax.loglog(freqs_welch, welch_psd, color=conductance_clr(cond_number))
            plotnum += 1

    fig.legend(lines, line_names, frameon=False)
    fig.savefig('population_from_distance_%s_%1.2f_%1.3f_duplicate.png' %(input_pos, correlation, syn_strength), dpi=150)
    
def plot_correlated_population_signals(ofolder, signal_dict, signal_psd_dict, freqs, num_cells, 
                                       elec_x, elec_y, elec_z, input_pos, correlation, conductance_list):
    conductance_clr = lambda cond_number: plt.cm.jet(int(256. * cond_number/(len(conductance_list) - 1.)))
    n_elecs = len(elec_z)
    fig = plt.figure(figsize=[7,10])
    fig.suptitle('Cells: %d, Stimulation: %s, Correlation: %1.2f'
                 % (num_cells, input_pos, correlation))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    lines = []
    line_names = []

    xmid = np.load(join(ofolder, 'xmid.npy' ))
    ymid = np.load(join(ofolder, 'ymid.npy' ))
    
    xstart = np.load(join(ofolder, 'xstart.npy' ))
    ystart = np.load(join(ofolder, 'ystart.npy' ))
    zstart = np.load(join(ofolder, 'zstart.npy' ))
    xend = np.load(join(ofolder, 'xend.npy' ))
    yend = np.load(join(ofolder, 'yend.npy' ))
    zend = np.load(join(ofolder, 'zend.npy' ))    
    diam = np.load(join(ofolder, 'diam.npy'))
    
    ax = fig.add_subplot(1, 3, 1, aspect='equal', frameon=False, xticks=[], yticks=[])
    for comp in xrange(len(xstart)):
        if comp == 0:
            ax.scatter(xmid[comp], ymid[comp], s=diam[comp], color='k')
        else:
            ax.plot([xstart[comp], xend[comp]], [ystart[comp], yend[comp]], lw=diam[comp], color='k')
    ax.scatter(elec_x, elec_z, c='r', s=10)

    for elec in xrange(n_elecs):
        
        ax = fig.add_subplot(n_elecs, 3, 3*(n_elecs - elec - 1) + 2, title=elec_z[elec])
        ax_psd = fig.add_subplot(n_elecs, 3, 3*(n_elecs - elec - 1) + 3, title=elec_z[elec])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax_psd.spines['top'].set_visible(False)
        ax_psd.spines['right'].set_visible(False)
        ax_psd.get_xaxis().tick_bottom()
        ax_psd.get_yaxis().tick_left()


        ax_psd.grid(True)
        for cond_numb, name in enumerate(conductance_list):
            ax.plot(np.arange(1001), signal_dict[name][elec], color=conductance_clr(cond_numb))
            ax_psd.loglog(freqs, signal_psd_dict[name][elec], color=conductance_clr(cond_numb))
            
            if elec == 0:
                ax.set_xlabel('ms')
                ax.set_ylabel('$\mu V$')
                ax_psd.set_xlabel('Hz')
                ax_psd.set_ylabel('$\mu V$')
                
                l, = ax.plot(0,0, color=conductance_clr(cond_numb))
                lines.append(l)
                line_names.append(name)
            
    fig.legend(lines, line_names)
    fig.savefig('population_%d_%s_%1.2f.png' %(num_cells, input_pos, correlation))

def plot_correlated_population(ofolder, conductance_list, num_cells, 
                               elec_x, elec_y, elec_z, input_pos, correlation):

    signal_dict = {}
    signal_psd_dict = {}
    for conductance_type in conductance_list:
        session_name = join(ofolder, 'signal_%s_%s_%1.2f' 
                            %(conductance_type, input_pos, correlation))
        signal_dict[conductance_type] = np.load('%s_total.npy' %session_name)
        signal_psd_dict[conductance_type], freqs = aLFP.find_LFP_PSD(
            signal_dict[conductance_type][:,:1000], (1.0)/1000.)
        
    plot_correlated_population_signals(ofolder, signal_dict, signal_psd_dict, freqs, num_cells, 
                                       elec_x, elec_y, elec_z, input_pos, correlation, conductance_list)

def sum_signals(ofolder, conductance_type, num_cells, num_elecs, input_pos, correlation):

    total_signal = np.zeros((num_elecs, 1001))
    session_name = join(ofolder, 'signal_%s_%s_%1.2f' %(conductance_type, input_pos, correlation))
    for simulation_idx in xrange(num_cells):
        total_signal += np.load('%s_sim_%d.npy' %(session_name, simulation_idx))
    np.save('%s_total.npy' %session_name, total_signal)

    
def sum_signals_population_sizes(ofolder, conductance_list, num_cells,
                                 num_elecs, input_positions, correlations, population_radius, syn_strength):

    x, y, rotation = np.load('x_y_rotation_%d_%d.npy' %(num_cells, population_radius))
    population_radii = np.arange(50, population_radius +1, 25)

    for conductance_type in conductance_list:
        for correlation in correlations:
            for input_pos in input_positions:
                session_name = join(ofolder, 'signal_%s_%s_%1.2f_%1.3f' %(conductance_type, input_pos,
                                                                          correlation, syn_strength))
                print session_name
                for radius in population_radii:
                    use_idxs = np.array([idx for idx in xrange(len(x))
                                         if np.sqrt(x[idx]**2 + y[idx]**2) <= radius])
                    #plt.close('all')
                    #plt.title('Population radius: %04d, Number of cells: %d ' %(int(radius), len(use_idxs)))
                    #plt.scatter(x[use_idxs], y[use_idxs], edgecolor='none', s=4)
                    #plt.xlim(-1000, 1000)
                    #plt.ylim(-1000, 1000)
                    #plt.savefig('population_radius%04d.png' % int(radius))
                    total_signal = np.zeros((num_elecs, 1001))
                    for simulation_idx in use_idxs:
                        total_signal += np.load('%s_sim_%d.npy' %(session_name, simulation_idx))
                    np.save('%s_total_pop_size_%04d.npy' %(session_name, int(radius)), total_signal)
    
def run_correlated_population_simulation(cell_params, conductance_list, ofolder, model_path, 
                                         elec_x, elec_y, elec_z, ntsteps, spiketrain_params, 
                                         correlation, num_cells, population_radius, simulation_idx):
   # Excitatory synapse parameters:
    syn_strength = 0.015
    synapse_params = {
        'e' : 0,   
        'syntype' : 'ExpSyn',      #conductance based exponential synapse
        'tau' : .1,                #Time constant, rise           #Time constant, decay
        'weight' : syn_strength,           #Synaptic weight
        'color' : 'r',              #for pl.plot
        'marker' : '.',             #for pl.plot
        'record_current' : False,    #record synaptic currents
        }
    vss = -77
    cell = LFPy.Cell(**cell_params)
    cell_input_idxs = cell.get_rand_idx_area_norm(section=spiketrain_params['section'], 
                                                  nidx=spiketrain_params['n'])

    x, y, rotation = np.load('x_y_rotation_%d_%d.npy' %(num_cells, population_radius))[:, simulation_idx]
    rot_params = {'x': 0, 
                  'y': 0, 
                  'z': rotation
                  }
    pos_params = {'xpos': x, 
                  'ypos': y,
                  'zpos': 0,
                  }        
    electrode_parameters = {
        'sigma' : 0.3,      # extracellular conductivity
        'x' : elec_x,  # electrode requires 1d vector of positions
        'y' : elec_y,
        'z' : elec_z
        }
    
    if np.abs(correlation) > 1e-6:
        # The spiketrain indexes are drawn from a common pool without replacement. 
        # The size of the common pool descides the average correlation
        all_spike_trains = np.load(join(ofolder, 'all_spike_trains.npy')).item()
        spike_train_idxs = np.array(random.sample(np.arange(int(spiketrain_params['n']/correlation)), 
                                            spiketrain_params['n']))
    else:
        num_trains = spiketrain_params['n']
        all_spike_trains = {}
        for idx in xrange(num_trains):
            all_spike_trains[idx] = LFPy.inputgenerators.stationary_poisson(
                1, 5, cell_params['tstartms'], cell_params['tstopms'])[0]
        spike_train_idxs = np.arange(spiketrain_params['n'])

    for conductance_type in conductance_list:
        neuron.h('forall delete_section()')
        neuron.h('secondorder=2')
        del cell
        cell_params['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                      join(model_path, 'biophys3_%s.hoc' % conductance_type)]        
        cell_params['v_init'] = -77
        cell = LFPy.Cell(**cell_params)

        cell.set_rotation(**rot_params)
        cell.set_pos(**pos_params)       
        
        if len(spiketrain_params['section']) == 2:
            input_name = 'homogeneous'
            
        elif len(spiketrain_params['section']) == 1:
            input_name = spiketrain_params['section'][0]
        else:
            raise RuntimeError, "Wrong synaptic_params"
        sim_name = '%s_%s_%1.2f_%1.3f_sim_%d' %(conductance_type, input_name,
                                                correlation, syn_strength, simulation_idx)
        
        if conductance_type in ['passive_vss', 'Ih_linearized']:
            for comp_idx, sec in enumerate(cell.allseclist):
                for seg in sec:
                    exec('seg.vss_%s = %g'% (conductance_type, vss))
        else:
            pass
        
        if os.path.isfile(join(ofolder, 'signal_%s.npy' %sim_name))
            print "Skipping %s" %name
            continue
        else:
            pass
        set_input_spiketrain(cell, all_spike_trains, cell_input_idxs, spike_train_idxs, synapse_params)
        cell.simulate(rec_imem=True, rec_vmem=True)

        if np.max(cell.somav) > -40:
            is_spiking = True
            plt.close('all')
            plt.plot(cell.tvec, cell.somav)
            plt.savefig('%s_is_spiking.png' % sim_name)
        else:
            is_spiking = False

        if conductance_type == 'active':
            print '%s is spiking: %s' %(sim_name, is_spiking)
        
        electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
        electrode.calc_lfp()
        if not at_stallo:
            np.save(join(ofolder, 'imem_%s.npy' %sim_name), cell.imem)
            np.save(join(ofolder, 'somav_%s.npy' %sim_name), cell.somav)
            np.save(join(ofolder, 'vmem_%s.npy' %sim_name), cell.vmem)
            #np.save(join(ofolder, 'vmem_%s.npy' %sim_name), cell.vmem)
            if conductance_type == 'active':
                plt.close('all')
                plt.subplot(121, aspect='equal', xlim=[-1000, 1000], ylim=[-1000, 1000])
                plt.scatter(cell.xmid, cell.ymid, edgecolor='none', s=2, c='r')
                plt.scatter(elec_x, elec_y, s=4, c='b')
            
                plt.subplot(122, aspect='equal', xlim=[-1000, 1000], ylim=[-400, 1400])
                plt.scatter(cell.xmid, cell.zmid, edgecolor='none', s=2, c='r')
                plt.scatter(elec_x, elec_z, s=4, c='b')
                plt.savefig('cell_%05d.png' %simulation_idx)

        elif simulation_idx % 500 == 0:
            np.save(join(ofolder, 'imem_%s.npy' %sim_name), cell.imem)
            np.save(join(ofolder, 'somav_%s.npy' %sim_name), cell.somav)
            np.save(join(ofolder, 'vmem_%s.npy' %sim_name), cell.vmem)
            if conductance_type == 'active':
                plt.close('all')
                plt.subplot(121, aspect='equal', xlim=[-1000, 1000], ylim=[-1000, 1000])
                plt.scatter(cell.xmid, cell.ymid, edgecolor='none', s=2, c='r')
                plt.scatter(elec_x, elec_y, s=4, c='b')
                plt.subplot(122, aspect='equal', xlim=[-1000, 1000], ylim=[-400, 1400])
                plt.scatter(cell.xmid, cell.zmid, edgecolor='none', s=2, c='r')
                plt.scatter(elec_x, elec_z, s=4, c='b')
                plt.savefig('cell_%05d.png' %simulation_idx)
        np.save(join(ofolder, 'signal_%s.npy' %sim_name), 1000*electrode.LFP)



## def run_correlated_population_simulation_debug(cell_params, conductance_list, ofolder, model_path, 
##                                          elec_x, elec_y, elec_z, ntsteps, spiketrain_params, 
##                                          correlation, num_cells, population_radius, simulation_idx):
##    # Excitatory synapse parameters:
##     syn_strength = 0.015
##     synapse_params = {
##         'e' : 0,   
##         'syntype' : 'ExpSyn',      #conductance based exponential synapse
##         'tau' : .1,                #Time constant, rise           #Time constant, decay
##         'weight' : syn_strength,           #Synaptic weight
##         'color' : 'r',              #for pl.plot
##         'marker' : '.',             #for pl.plot
##         'record_current' : False,    #record synaptic currents
##         }
##     vss = -77
##     cell = LFPy.Cell(**cell_params)
##     cell_input_idxs = cell.get_rand_idx_area_norm(section=spiketrain_params['section'], 
##                                                   nidx=spiketrain_params['n'])

##     x, y, rotation = np.load('x_y_rotation_%d_%d.npy' %(num_cells, population_radius))[:, simulation_idx]
##     rot_params = {'x': 0, 
##                   'y': 0, 
##                   'z': rotation
##                   }
##     pos_params = {'xpos': x, 
##                   'ypos': y,
##                   'zpos': 0,
##                   }        
##     electrode_parameters = {
##         'sigma' : 0.3,      # extracellular conductivity
##         'x' : elec_x,  # electrode requires 1d vector of positions
##         'y' : elec_y,
##         'z' : elec_z
##         }
    
##     if np.abs(correlation) > 1e-6:
##         # The spiketrain indexes are drawn from a common pool without replacement. 
##         # The size of the common pool descides the average correlation
##         all_spike_trains = np.load(join(ofolder, 'all_spike_trains.npy')).item()
##         spike_train_idxs = np.array(random.sample(np.arange(int(spiketrain_params['n']/correlation)), 
##                                             spiketrain_params['n']))        
##     else:
##         num_trains = spiketrain_params['n']
##         all_spike_trains = {}
##         for idx in xrange(num_trains):
##             all_spike_trains[idx] = LFPy.inputgenerators.stationary_poisson(
##                 1, 5, cell_params['tstartms'], cell_params['tstopms'])[0]
##         spike_train_idxs = np.arange(spiketrain_params['n'])

##     for idx in spike_train_idxs:
##         plt.plot(all_spike_trains[idx], idx*np.ones(len(all_spike_trains[idx])), 'k.')
##     plt.savefig('a_debug_%s_%1.2f.png' % (spiketrain_params['section'], correlation)) 
        
##     for conductance_type in conductance_list:
##         neuron.h('forall delete_section()')
##         neuron.h('secondorder=2')
##         del cell
##         cell_params['custom_code'] = [join(model_path, 'custom_codes.hoc'),
##                                       join(model_path, 'biophys3_%s.hoc' % conductance_type)]        
##         cell_params['v_init'] = -77
##         cell = LFPy.Cell(**cell_params)

##         cell.set_rotation(**rot_params)
##         cell.set_pos(**pos_params)       
        
##         if len(spiketrain_params['section']) == 2:
##             input_name = 'homogeneous'
            
##         elif len(spiketrain_params['section']) == 1:
##             input_name = spiketrain_params['section'][0]
##         else:
##             raise RuntimeError, "Wrong synaptic_params"
##         sim_name = '%s_%s_%1.2f_%1.3f_sim_%d' %(conductance_type, input_name,
##                                                 correlation, syn_strength, simulation_idx)
##         print sim_name
##         if conductance_type in ['passive_vss', 'Ih_linearized']:
##             for comp_idx, sec in enumerate(cell.allseclist):
##                 for seg in sec:
##                     exec('seg.vss_%s = %g'% (conductance_type, vss))
##         else:
##             pass
            
##         set_input_spiketrain(cell, all_spike_trains, cell_input_idxs, spike_train_idxs, synapse_params)
##         cell.simulate(rec_imem=True, rec_vmem=True)
##         electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
##         electrode.calc_lfp()
##         if not at_stallo:
##             np.save(join(ofolder, 'imem_%s.npy' %sim_name), cell.imem)
##             np.save(join(ofolder, 'somav_%s.npy' %sim_name), cell.somav)
##             np.save(join(ofolder, 'vmem_%s.npy' %sim_name), cell.vmem)
##             #np.save(join(ofolder, 'vmem_%s.npy' %sim_name), cell.vmem)
##             if conductance_type == 'active':
##                 plt.close('all')
##                 plt.subplot(121, aspect='equal', xlim=[-1000, 1000], ylim=[-1000, 1000])
##                 plt.scatter(cell.xmid, cell.ymid, edgecolor='none', s=2, c='r')
##                 plt.scatter(elec_x, elec_y, s=4, c='b')
            
##                 plt.subplot(122, aspect='equal', xlim=[-1000, 1000], ylim=[-400, 1400])
##                 plt.scatter(cell.xmid, cell.zmid, edgecolor='none', s=2, c='r')
##                 plt.scatter(elec_x, elec_z, s=4, c='b')
##                 plt.savefig('cell_%05d.png' %simulation_idx)

##         elif simulation_idx % 500 == 0:
##             np.save(join(ofolder, 'imem_%s.npy' %sim_name), cell.imem)
##             np.save(join(ofolder, 'somav_%s.npy' %sim_name), cell.somav)
##             np.save(join(ofolder, 'vmem_%s.npy' %sim_name), cell.vmem)
##             if conductance_type == 'active':
##                 plt.close('all')
##                 plt.subplot(121, aspect='equal', xlim=[-1000, 1000], ylim=[-1000, 1000])
##                 plt.scatter(cell.xmid, cell.ymid, edgecolor='none', s=2, c='r')
##                 plt.scatter(elec_x, elec_y, s=4, c='b')
##                 plt.subplot(122, aspect='equal', xlim=[-1000, 1000], ylim=[-400, 1400])
##                 plt.scatter(cell.xmid, cell.zmid, edgecolor='none', s=2, c='r')
##                 plt.scatter(elec_x, elec_z, s=4, c='b')
##                 plt.savefig('cell_%05d.png' %simulation_idx)
##         np.save(join(ofolder, 'signal_%s.npy' %sim_name), 1000*electrode.LFP)


        
def set_input_spiketrain(cell, all_spike_trains, cell_input_idxs, spike_train_idxs, synapse_params):
    """ Makes synapses and feeds them predetermined spiketimes """
    for number, comp_idx in enumerate(cell_input_idxs):
        synapse_params.update({'idx' : int(comp_idx)})
        s = LFPy.Synapse(cell, **synapse_params)
        spiketrain_idx = spike_train_idxs[number]
        s.set_spike_times(all_spike_trains[spiketrain_idx])

def distribute_cells(num_cells, R):

    x_y_rotation_array = np.zeros((3, num_cells))
    for cell_idx in xrange(num_cells):
        neur = 'cell_%05d' % cell_idx
        x = 2 * R * (np.random.random() - 0.5)
        y = 2 * R * (np.random.random() - 0.5)
        while x**2 + y**2 >= R**2:
            x = 2 * R * (np.random.random() - 0.5)
            y = 2 * R * (np.random.random() - 0.5)
        x_y_rotation_array[:, cell_idx] = x, y, 2*np.pi*np.random.random()

    np.save('x_y_rotation_%d_%d.npy' %(num_cells, R), x_y_rotation_array)
    plt.close('all')
    plt.scatter(x_y_rotation_array[0,:], x_y_rotation_array[1,:], c=x_y_rotation_array[2,:], 
                edgecolor='none', s=4, cmap='hsv')
    plt.axis('equal')
    plt.savefig('cell_distribution_%d_%d.png' %(num_cells, R))
    
