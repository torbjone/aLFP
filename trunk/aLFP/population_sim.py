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
np.random.seed(1234)

def run_population_simulation(cell_params, conductance_list, ofolder, model_path,
                              ntsteps, all_synaptic_params, numsimulations=1):

   # Excitatory synapse parameters:
    synapseParameters_AMPA = {
        'e' : 0,                    #reversal potential
        'syntype' : 'Exp2Syn',      #conductance based exponential synapse
        'tau1' : 1.,                #Time constant, rise
        'tau2' : 3.,                #Time constant, decay
        'weight' : 0.005,           #Synaptic weight
        'color' : 'r',              #for pl.plot
        'marker' : '.',             #for pl.plot
        'record_current' : False,    #record synaptic currents
        }
    # Excitatory synapse parameters
    synapseParameters_NMDA = {         
        'e' : 0,
        'syntype' : 'Exp2Syn',
        'tau1' : 10.,
        'tau2' : 30.,
        'weight' : 0.001, #Scaled down from 0.005 to remove spiking!
        'color' : 'm',
        'marker' : '.',
        'record_current' : False,
        }
    # Inhibitory synapse parameters
    synapseParameters_GABA_A = {         
        'e' : -80,
        'syntype' : 'Exp2Syn',
        'tau1' : 1.,
        'tau2' : 12.,
        'weight' : 0.005,
        'color' : 'b',
        'marker' : '.',
        'record_current' : False
        }
    #static_Vm = np.load(join(ofolder, 'static_Vm_distribution_active_vss.npy'))
    vss = -70
    for part in xrange(numsimulations):
        cell = LFPy.Cell(**cell_params)
    
        AMPA_spiketimes_dict = make_input_spiketrain(cell, **all_synaptic_params['AMPA'])
        GABA_spiketimes_dict = make_input_spiketrain(cell, **all_synaptic_params['GABA_A'])
        NMDA_spiketimes_dict = make_input_spiketrain(cell, **all_synaptic_params['NMDA'])

        for conductance_type in conductance_list:
            neuron.h('forall delete_section()')
            neuron.h('secondorder=2')
            del cell
            cell_params['custom_code'] = [join(model_path, 'custom_codes.hoc'),
                                          join(model_path, 'biophys3_%s.hoc' % conductance_type)]        

            cell_params['v_init'] = -77#np.average(static_Vm)
            cell = LFPy.Cell(**cell_params)
            sim_name = conductance_type + '_%d' %part
            if conductance_type in ['active', 'active_homogeneous_Ih']:
                pass
            elif conductance_type in ['active_vss', 'active_vss_homogeneous_Ih']:
                comp_idx = 0
                for sec in cell.allseclist:
                    for seg in sec:
                        exec('seg.vss_passive_vss = static_Vm[%d]'% (comp_idx))
                        comp_idx += 1
            elif conductance_type in ['Ih_reduced', 'Ih_linearized']:
                comp_idx = 0
                for sec in cell.allseclist:
                    for seg in sec:
                        exec('seg.vss_%s = %g'% (conductance_type, vss))
                        comp_idx += 1
            else:
                raise RuntimeError
                        
            set_input_spiketrain(cell, synapseParameters_AMPA, AMPA_spiketimes_dict)
            set_input_spiketrain(cell, synapseParameters_GABA_A, GABA_spiketimes_dict)
            set_input_spiketrain(cell, synapseParameters_NMDA, NMDA_spiketimes_dict)

            cell.simulate(rec_imem=True)
            f = tables.openFile(join(ofolder, 'signal_%s.h5' %sim_name), mode = "w")
            filters = tables.Filters(complevel=5, complib='zlib')
            somav_array = f.createCArray('/', 'somav', tables.Float32Col(),
                                         [len(cell.somav)], filters=filters)
            somav_array[:] = cell.somav
            t_array = f.createCArray('/', 'tvec', tables.Float32Col(),
                                     [len(cell.tvec)], filters=filters)
            t_array[:] = cell.tvec
            imem_array = f.createCArray('/', 'imem', tables.Float32Col(),
                                        cell.imem.shape, filters=filters)
            imem_array[:,:] = cell.imem
            f.close()
            plot_example(cell, sim_name)


def combine_parts(conductance_list, ofolder, ntsteps, numsimulations=1):
    """ If simulations have been split into parts, they are combined again here"""

    filters = tables.Filters(complevel=5)
    sim_name_temp = conductance_list[0] + '_0'
    f_temp = tables.openFile(join(ofolder, 'signal_%s.h5' %sim_name_temp) , mode = "r")
    ncomps = f_temp.root.imem.shape[0]
    f_temp.close()
    for numb, conductance_type in enumerate(conductance_list):
        f_total = tables.openFile(join(ofolder, 'signal_%s.h5' %conductance_type), mode = "w")
        tot_ntsteps = ntsteps * numsimulations
        sig_array = f_total.createCArray('/', 'imem', tables.Float32Col(), 
                                         [ncomps, tot_ntsteps], filters=filters)
        somav_array = f_total.createCArray('/', 'somav', tables.Float32Col(),
                                         [tot_ntsteps], filters=filters)
        for partnumb in xrange(numsimulations):
            print conductance_type, partnumb
            sim_name = conductance_type + '_%d' %partnumb
            part_filename = join(ofolder, 'signal_%s.h5' %sim_name)
            f_part = tables.openFile(part_filename, mode = "r")
            sig_array[:,partnumb*ntsteps:(partnumb+1)*ntsteps] = f_part.root.imem[:,:ntsteps]
            somav_array[partnumb*ntsteps:(partnumb+1)*ntsteps] = f_part.root.somav[:ntsteps]  
            f_part.close()
        f_total.close()

def calculate_LFP(neuron_dict, conductance_list, ofolder, population_dict, 
                  elec_x, elec_y, elec_z, cell_params):
    for conductance_type in conductance_list:
        f = tables.openFile(join(ofolder, 'signal_%s.h5' %conductance_type), mode = "r")
        ntsteps_window = int(population_dict['window_length_ms'] / population_dict['timeres'])    
        
        somavs = np.zeros((np.min([10, len(neuron_dict)]), ntsteps_window))
        #imems = np.zeros((len(neuron_dict), ntsteps_window))
        total_LFP = np.zeros((len(elec_x), ntsteps_window))
        printlist = [len(neuron_dict)/10 * i for i in xrange(10)]
        for idx, neur in enumerate(neuron_dict):
            if idx in printlist:
                print conductance_type, (idx * 100)/len(neuron_dict) , '%'
            idx1, idx2 = neuron_dict[neur]['time_window_idx']
            if idx < 10:
                somavs[idx,:] = f.root.somav[idx1:idx2]
            #imems[idx,:] = f.root.imem[0,idx1:idx2]
            total_LFP += 1000 * np.dot(neuron_dict[neur]['mapping'], f.root.imem[:,idx1:idx2])
        f.close()    
        timestep = population_dict['timeres']/ 1000.
        total_LFP_psd, freqs = aLFP.find_LFP_PSD(total_LFP, timestep)
        plot_population_LFP(somavs, freqs, total_LFP, total_LFP_psd, neuron_dict, 
                            population_dict, elec_x, elec_y, elec_z, conductance_type, cell_params)



def plot_soma_to_axis(neur_dict, ax, cell_params):
    ax.scatter(neur_dict['position']['xpos'], neur_dict['position']['zpos'], 
               s=4, c=neur_dict['clr'], clip_on=False, edgecolor='none')

def plot_morph_to_axis(neur_dict, ax, cell_params):
    neuron.h('forall delete_section()')
    foo_params = cell_params.copy()
    cell = LFPy.Cell(**foo_params)
    cell.set_rotation(**neur_dict['rotation'])
    cell.set_pos(**neur_dict['position'])   
    for comp in xrange(len(cell.xmid)):
        if comp == 0:
            ax.scatter(cell.xmid[comp], cell.zmid[comp], 
                        s=cell.diam[comp], c=neur_dict['clr'], edgecolor='none')
        else:
            ax.plot([cell.xstart[comp], cell.xend[comp]],
                    [cell.zstart[comp], cell.zend[comp]], 
                    lw=cell.diam[comp]/2, color=neur_dict['clr'])

    
def plot_population_to_axis(neuron_dict, ax, cell_params):

    if len(neuron_dict) > 10:
        func = plot_soma_to_axis
    else:
        func = plot_morph_to_axis
    for idx, neur in enumerate(neuron_dict):
        func(neuron_dict[neur], ax, cell_params)

        
def plot_population_LFP(somavs, freqs, total_LFP, total_LFP_psd, neuron_dict,
                        population_dict, elec_x, elec_y, elec_z, conductance_type, cell_params):

    numplots = np.min([10, len(neuron_dict)])
    fig = plt.figure(figsize=[8,6])
    fig.subplots_adjust(wspace=0.9, hspace=0.7)
    ax1 = fig.add_subplot(141, frameon=False , yticks=[], xticks=[])
    
    ax1.set_ylim(np.min(elec_z) - 100 , np.max(elec_z) + 100)
    ax1.set_title('Electrode and population')
    ntsteps_window = int(population_dict['window_length_ms'] / population_dict['timeres'])
    short_t = np.arange(ntsteps_window)*population_dict['timeres']
    
    major_formatter = plt.FormatStrFormatter('%0.2f')
    ax1.get_xaxis().tick_top()

    plot_population_to_axis(neuron_dict, ax1, cell_params)
    
    for idx, neur in enumerate(neuron_dict):

        if idx < numplots:
            ax = fig.add_subplot(10, 4, (idx*4 + 4), frameon=False, xticks=[], yticks=[])
            
            #ax.spines['top'].set_visible(False)
            #ax.spines['right'].set_visible(False)
            #ax.get_xaxis().tick_bottom()
            #ax.get_yaxis().tick_left()      
            ax.set_ylim(np.min(somavs[idx]) - 5, np.min(somavs[idx]) + 15)
            if idx == 0:
                ax.set_title("Some soma Vm's")    
                ax.plot([1100, 1100], [np.min(somavs[idx]) - 5, np.min(somavs[idx]) + 15], clip_on=False, lw=2, color='k')
                ax.text(1120, np.min(somavs[idx]) + 5, '20 mV')
            ax.plot(short_t, somavs[idx], color=neuron_dict[neur]['clr'])
            #ax.set_xticks([0, int((short_t[-1] + 1)/2),  int(short_t[-1] + 1)])
            #ax.set_yticks([np.min(somavs[idx]), np.max(somavs[idx])])
            #ax.xaxis.set_major_formatter(major_formatter)
            #if idx == numplots - 1:
                #ax.set_xlabel('[ms]')
    elec_clr = lambda idx: plt.cm.jet(int(256./len(elec_x) * idx))  
    
    for elec in xrange(len(elec_x)):
        ax1.scatter(elec_x[elec], elec_z[elec], s=10, c=elec_clr(elec), edgecolor='none')
        ax = fig.add_subplot(len(elec_x), 4, ((len(elec_x) - elec - 1)*4 + 2), frameon=False, xticks=[], yticks=[])
        LFP_sig = total_LFP[elec,:] - np.average(total_LFP[elec,:])
        if elec == len(elec_x) - 1: 
            ax.set_title('LFP [$\mu V$]')      
        
        ax.plot(short_t, LFP_sig, color=elec_clr(elec))
        ax.set_ylim(np.min(LFP_sig), np.max(LFP_sig))
        ax.plot([short_t[-1] + 10, short_t[-1] + 10], [np.min(LFP_sig), np.max(LFP_sig)], lw=2, clip_on=False, color='k')
        ax.text(short_t[-1] + 20, (np.max(LFP_sig) - np.min(LFP_sig))/2 +  np.min(LFP_sig), '%1.3f $\mu V$' %(np.max(LFP_sig) - np.min(LFP_sig)))

        if elec == 0:
            #set_trace()
            ax.plot([short_t[0], short_t[-1]], [np.min(LFP_sig)- (np.max(LFP_sig) - np.min(LFP_sig))/5, np.min(LFP_sig)
                                                - (np.max(LFP_sig) - np.min(LFP_sig))/5], lw=2, clip_on=False, color='k')
            ax.text(short_t[-(len(short_t)/2)], np.min(LFP_sig) - (np.max(LFP_sig) - np.min(LFP_sig))/1.5, '%d ms' % np.ceil(short_t[-1]))
            #ax.set_xlabel('ms')
        
        #ax.set_xticks([0, int((short_t[-1] + 1)/2),  int(short_t[-1] + 1)])
        #ax.set_yticks([np.min(LFP_sig), np.max(LFP_sig)])

        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        #ax.get_xaxis().tick_bottom()
        #ax.get_yaxis().tick_left()      
        #ax.xaxis.set_major_formatter(major_formatter)
        ax_psd = fig.add_subplot(len(elec_x), 4, ((len(elec_x) - elec - 1)*4 + 3))
        if elec == len(elec_x) - 1: 
            ax_psd.set_title('LFP PSD [$\mu V$]')      
        if elec == 0:
            ax_psd.set_xlabel('Hz')
        ax_psd.loglog(freqs, total_LFP_psd[elec,:], color=elec_clr(elec))
        ax_psd.set_xlim(1,110)
        ax_psd.set_ylim(1e-5,1e1)
        ax_psd.set_xticks([1e0, 1e2])
        ax_psd.set_yticks([1e-5, 1e-3, 1e-1, 1e1])
        ax_psd.spines['top'].set_visible(False)
        ax_psd.spines['right'].set_visible(False)
        ax_psd.get_xaxis().tick_bottom()
        ax_psd.get_yaxis().tick_left()
    fig.savefig('population_LFP_%d_%s.png' % (population_dict['numcells'], conductance_type))   
    
def plot_example(cell, sim_name):

    plt.close('all')
    fig = plt.figure(figsize=[15, 6])
    ax1 = fig.add_axes([0.05, 0.1, 0.7, 0.3])
    ax3 = fig.add_axes([0.05, 0.6, 0.7, 0.3])
    ax4 = fig.add_axes([0.75, 0.1, 0.18, 0.80], aspect='equal', frameon=False)
    
    ax1.plot(cell.tvec, cell.somav)
    ax1.set_title('Soma potential [mV]')    
    ax3.plot(cell.tvec, cell.imem[0,:])
    ax3.set_title('Soma membrane current [nA]')
    ax1.set_xlabel('Time [ms]')
    ax3.set_xlabel('Time [ms]')

    for sec in neuron.h.allsec():
        idx = cell.get_idx(sec.name())        
        ax4.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                 np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                 color='k', lw=2)
    for i in xrange(len(cell.synapses)):
        ax4.plot([cell.synapses[i].x], [cell.synapses[i].z],
            color=cell.synapses[i].color, marker=cell.synapses[i].marker, 
            markersize=10)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_title('GABA_A: b, AMPA: r, NMDA: m')
    fig.savefig(join('summary_%s.png'% sim_name))

def make_input_spiketrain(cell, section, n, spTimesFun, args):
    """ Make and return spiketimes for each compartment that receives input """
    cell_idxs = cell.get_rand_idx_area_norm(section=section, nidx=n)
    spiketimes_dict = {}
    for idx in cell_idxs:
        spiketimes_dict[str(idx)] = spTimesFun(args[0], args[1], args[2], args[3])
    return spiketimes_dict

def set_input_spiketrain(cell, synparams, spiketimes_dict):
    """ Find spiketimes """
    for idx, spiketrain in spiketimes_dict.items():
        synparams.update({'idx' : int(idx)})
        s = LFPy.Synapse(cell, **synparams)
        s.set_spike_times(spiketrain)
        
def quickplot_vmem(cell, ofolder):
    plt.plot(cell.tvec, cell.somav)
    plt.savefig(join(ofolder, 'vmem.png'))
    
def plot_population(neuron_dict, elec_x, elec_y, elec_z):

    fig = plt.figure(figsize=[3,6])
    ax1 = fig.add_axes([0.1,0.5, 0.8, 0.4], frameon=False , yticks=[])
    ax2 = fig.add_axes([0.1,0.05,0.8, 0.4], sharex=ax1, frameon=False, yticks=[])
    ax1.scatter(elec_x, elec_z, s=10, c='r', edgecolor='none')
    ax2.scatter(elec_x, elec_y, s=10, c='r', edgecolor='none')

    ax1.set_ylim(np.min(elec_z) - 100 , np.max(elec_z) + 100)
    #ax2.get_xaxis().tick_bottom()
    ax1.get_xaxis().tick_top()
    ax2.spines['top'].set_visible(False)
    for neur in neuron_dict.values():
        #set_trace()
        ax1.scatter(neur['position']['xpos'], neur['position']['zpos'], 
                    s=4, c=neur['clr'], clip_on=False, edgecolor='none')
        ax2.scatter(neur['position']['xpos'], neur['position']['ypos'], 
                    s=4, c=neur['clr'], clip_on=False, edgecolor='none')
    fig.savefig('population.png')   

    
def initialize_dummy_population(population_dict, cell_params,
                    elec_x, elec_y, elec_z, ntsteps, ofolder):
    """ Initializes and saves a dictionary with dummy cells. 
    Each cell entry in the dictionary contains a random x,y,z- position of the soma, as well
    as a random rotation, a color, a mapping, and a time window. The time windows is
    in idxs and is meant to be used to extract imems from the cell simulation and add to the LFP. 
    """

    neur_clr = lambda idx: plt.cm.rainbow(int(256./population_dict['numcells'] * idx))  
    neuron_dict = {}
    tvec = np.arange(population_dict['ntsteps'] * population_dict['numsimulations']) * \
      population_dict['timeres']
    ntsteps_window = int(population_dict['window_length_ms'] / population_dict['timeres'])
    max_idx = 0
    
    if population_dict['numcells'] > 100:
        plot_morphology = False
    else:
        plot_morphology = True
    if plot_morphology:
        fig = plt.figure()
        ax1 = fig.add_subplot(121, aspect='equal', rasterized=True)
        ax2 = fig.add_subplot(122, aspect='equal', rasterized=True)
        
    for cell_id in xrange(population_dict['numcells']):
        neur = 'cell_%04i' % cell_id
        print cell_id, '/', population_dict['numcells']
        xpos = 2 * population_dict['r_limit'] * (np.random.random() - 0.5)
        ypos = 2 * population_dict['r_limit'] * (np.random.random() - 0.5)
        while np.sqrt(xpos*xpos + ypos*ypos) > population_dict['r_limit']:
             xpos = 2 * population_dict['r_limit'] * (np.random.random() - 0.5)
             ypos = 2 * population_dict['r_limit'] * (np.random.random() - 0.5)  
        pos_dict = {'xpos': xpos,
                    'ypos': ypos,
                    'zpos': np.random.normal(population_dict['z_mid'], 100)
                    }
        rot_dict = {'x': 0, 
                    'y': 0, 
                    'z': 2*np.pi*np.random.random()
                    }    
        neuron.h('forall delete_section()')
        try:
            os.mkdir(ofolder)
        except OSError:
            pass
        foo_params = cell_params.copy()
        foo_params['tstartms'] = 0
        foo_params['tstopms'] = 0
        cell = LFPy.Cell(**foo_params)
        cell.set_rotation(**rot_dict)
        cell.set_pos(**pos_dict)       
        if plot_morphology:
            for comp in xrange(len(cell.xmid)):
                if comp == 0:
                    ax1.scatter(cell.xmid[comp], cell.zmid[comp], 
                                s=cell.diam[comp], c=neur_clr(cell_id), edgecolor='none')
                    ax2.scatter(cell.xmid[comp], cell.ymid[comp], 
                                c=neur_clr(cell_id), marker='^', edgecolor='none')
                else:
                    ax1.plot([cell.xstart[comp], cell.xend[comp]],
                             [cell.zstart[comp], cell.zend[comp]], 
                             lw=cell.diam[comp]/2, color=neur_clr(cell_id))
                    
                
        # Define electrode parameters
        electrode_parameters = {
            'sigma' : 0.3,      # extracellular conductivity
            'x' : elec_x,  # electrode requires 1d vector of positions
            'y' : elec_y,
            'z' : elec_z,
            'method': 'pointsource'
            }

        electrode = LFPy.RecExtElectrode(**electrode_parameters)
        cell.simulate(electrode=electrode)
        
        window_start_idx = np.random.randint(0, len(tvec) - ntsteps_window + 1)
        window_end_idx = window_start_idx + ntsteps_window
        neuron_dict[neur] = {'position': pos_dict,
                             'rotation': rot_dict,
                             'clr': neur_clr(cell_id),
                             'mapping': electrode.electrodecoeff,
                             'time_window_idx': [window_start_idx, window_end_idx]
        }
    if plot_morphology:
        ax1.scatter(elec_x, elec_z, c='r', s=20)
        ax2.scatter(elec_x, elec_y, c='r', s=20) 
        fig.savefig('morph_%d.png' %len(neuron_dict))
    plot_population(neuron_dict, elec_x, elec_y, elec_z)
    pickle.dump(neuron_dict, open(join(ofolder, 'neuron_dict.p'), "wb"))
