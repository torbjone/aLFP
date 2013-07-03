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
import pylab as plt
from os.path import join
import cPickle
import aLFP
import scipy.fftpack as ff
import scipy.signal

plt.rcParams.update({'font.size' : 8,
    'figure.facecolor' : '1',
    'wspace' : 0.5, 'hspace' : 0.5})
np.random.seed(1234)




def run_linearized_simulation(cell_params, input_scaling, input_idx, 
                   ofolder, ntsteps, simulation_params, conductance_type, 
                   input_type, downsample=False):

    neuron.h('forall delete_section()')
    neuron.h('secondorder=2')
    static_Vm = np.load(join(ofolder, 'static_Vm_distribution.npy'))
    cell_params['v_init'] = np.average(static_Vm)
    cell = LFPy.Cell(**cell_params)
    sim_name = '%d_%1.3f_%s' %(input_idx, input_scaling, conductance_type)
    if not conductance_type == 'active':
        comp_idx = 0
        for sec in cell.allseclist:
            for seg in sec:
                exec('seg.vss_%s = static_Vm[%d]'% (conductance_type, comp_idx))
                comp_idx += 1

    if input_type == 'synapse':

        # Define synapse parameters
        synapse_parameters = {
            'idx' : input_idx,
            'e' : 0.,                   # reversal potential
            'syntype' : 'ExpSyn',       # synapse type
            'tau' : 10.,                # syn. time constant
            'weight' : input_scaling,            # syn. weight
            'record_current' : True,
            }

        # Create synapse and set time of synaptic input
        synapse = LFPy.Synapse(cell, **synapse_parameters)
        synapse.set_spike_times(np.array([20.]))

    elif input_type == 'ZAP':
        downsample = True
        ZAP_clamp = {
            'idx' : input_idx,
            'record_current' : True,
            'dur' : 20000,
            'delay': 0,
            'freq_start' : 0,
            'freq_end': 15,
            'pkamp' : input_scaling,
            'pptype' : 'ZAPClamp',
            }
        current_clamp = LFPy.StimIntElectrode(cell, **ZAP_clamp)
    else:
        raise FAIL
        
    cell.simulate(**simulation_params)

    if downsample:
        set_trace()
        resample_npts = len(cell.tvec) / 4
        imem, tvec = scipy.signal.resample(cell.imem, resample_npts, t=cell.tvec, axis=1)
        vmem, tvec = scipy.signal.resample(cell.vmem, resample_npts, t=cell.tvec, axis=1)
    else:
        imem = cell.imem
        vmem = cell.vmem
        tvec = cell.tvec
        
    timestep = (tvec[1] - tvec[0])/1000.
    np.save(join(ofolder, 'tvec.npy'), tvec)

    mapping = np.load(join(ofolder, 'mapping.npy'))
    sig = 1000 * np.dot(mapping, imem)
    #sig_psd, freqs = find_LFP_PSD(sig, timestep)
    #np.save(join(ofolder, 'sig_psd_%s.npy' %(sim_name)), sig_psd)
    np.save(join(ofolder, 'sig_%s.npy' %(sim_name)), sig)
    
    linearized_quickplot(cell, sim_name, ofolder, static_Vm)
    #vmem_psd, freqs = find_LFP_PSD(cell.vmem, timestep)
    #np.save(join(ofolder, 'vmem_psd_%s.npy' %(sim_name)), vmem_psd)
    np.save(join(ofolder, 'vmem_%s.npy' %(sim_name)), vmem)
    
    imem_psd, freqs = find_LFP_PSD(imem, timestep)
    #np.save(join(ofolder, 'imem_psd_%s.npy' %(sim_name)), imem_psd)
    np.save(join(ofolder, 'imem_%s.npy' %(sim_name)), imem)
    #np.save(join(ofolder, 'freqs.npy'), freqs)

    
def initialize_dummy_population(cell_params,  cell_name, 
                    elec_x, elec_y, elec_z, ntsteps, ofolder, testing=False, make_WN_input=False):
    """ Initializes and saves a dictionary with dummy cells. 
    Each cell entry in the dictionary contains a random x,y,z- position of the soma, as well
    as a random rotation, a color, a mapping, and a time window. The time windows is
    in idxs and is meant to be used to extract imems from the cell simulation and add to the LFP. """

    pos_params, rot_params,

    population_dict = {'r_limit': 200,
                       'z_mid': 0,
                       'numcells': 1,
                       'window_length_ms': 200, 
                       }

    neuron_dict = {}
    for cell_id in xrange(population_dict['numcells']):
        neur = 'cell_%04i' % idx

        xpos = population_dict['r_limit'] * np.random.random()
        ypos = population_dict['r_limit'] * np.random.random()
        while np.sqrt(xpos*xpos + ypos*ypos) > population_dict['r_limit']:
             xpos = population_dict['r_limit'] * np.random.random()
             ypos = population_dict['r_limit'] * np.random.random()
             
        neuron_dict[neur]['position'] = {'xpos': xpos,
                                         'ypos': ypos,
                                         'zpos': np.random.normal(population_dict['z_mid'], 0.1}
        neuron_dict[neur]['rotation'] = {'x': -np.pi/2, 
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
    foo_params['tstopms'] = 1
    cell = LFPy.Cell(**foo_params)
    cell.set_rotation(**rot_params)
    cell.set_pos(**pos_params)       
    if testing:
        aLFP.plot_comp_numbers(cell)

    # Define electrode parameters
    electrode_parameters = {
        'sigma' : 0.3,      # extracellular conductivity
        'x' : elec_x,  # electrode requires 1d vector of positions
        'y' : elec_y,
        'z' : elec_z
        }
    dist_list = []
    for elec in xrange(len(elec_x)):
        for comp in xrange(len(cell.xmid)):
            dist = np.sqrt((cell.xmid[comp] - elec_x[elec])**2 +
                           (cell.ymid[comp] - elec_y[elec])**2 +
                           (cell.zmid[comp] - elec_z[elec])**2)
            dist_list.append(dist)
    print "Minimum electrode-comp distance: %g" %(np.min(dist_list))
    if np.min(dist_list) <= 1:
        ERR = "Too close"
        raise RuntimeError, ERR
    
    electrode = LFPy.RecExtElectrode(**electrode_parameters)
    cell.simulate(electrode=electrode)
    pos_quickplot(cell, cell_name, elec_x, elec_y, elec_z, ofolder)
