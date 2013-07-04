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

def run_all_WN_simulations(cell_params, model, input_idxs, input_scalings, ntsteps,
                           simulation_params, conductance_type, epas_array=[None]):
    for epas in epas_array:
        for input_idx in input_idxs:
            for input_scaling in input_scalings:
                run_WN_simulation(cell_params, input_scaling, input_idx, model, ntsteps, 
                                  simulation_params, conductance_type, epas)        

def run_all_synaptic_simulations(cell_params, model, input_idxs, input_scalings, ntsteps,
                                 simulation_params, conductance_type, epas_array=[None]):
        for epas in epas_array:
            for input_idx in input_idxs:
                for input_scaling in input_scalings:
                    run_synaptic_simulation(cell_params, input_scaling, input_idx, model, ntsteps,
                                            simulation_params, conductance_type)        
        
def run_all_linearized_simulations(cell_params, model, input_idxs, input_scalings, ntsteps,
                                   simulation_params, conductance_type, input_type='synaptic'):
    for input_idx in input_idxs:
        for input_scaling in input_scalings:
            run_linearized_simulation(cell_params, input_scaling, input_idx, model, ntsteps,
                                      simulation_params, conductance_type, input_type)        
                    
def find_LFP_PSD(sig, timestep):
    """ Returns the power and freqency of the input signal"""
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig, axis=1)[:,pidxs[0]]
    power = np.abs(Y)/Y.shape[1]
    return power, freqs

def return_power(sig, timestep):
    """ Returns the power of the input signal"""
    try:
        sample_freq = ff.fftfreq(len(sig), d=timestep)
    except:
        sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq > 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig)[pidxs]
    power = np.abs(Y)/len(Y)
    return power
            
## def plot_active_currents(cell, sim_name, ofolder):
##     plt.close('all')

##     n_cols = 6
##     n_rows = 6
##     fig = plt.figure(figsize=[15,10])
##     if n_cols * n_rows < cell.imem.shape[0]:
##         n_plots = n_cols * n_rows
##         plots = np.array(np.linspace(0, cell.imem.shape[0] - 1, n_plots), dtype=int)
##     else:
##         n_plots = cell.imem.shape[0]
##         plots = xrange(n_plots)
##     for plot_number, idx in enumerate(plots):
##         ax = fig.add_subplot(n_rows, n_cols, plot_number + 1)
##         for i in cell.rec_variables:
##             ax.plot(cell.tvec, cell.rec_variables[i][idx,:], label=i)
##         ax.plot(cell.tvec, cell.imem[idx,:], label='Imem')
##         if plot_number == n_plots - 1:
##             ax.legend()
##     plt.savefig('active_currents_%s_%s.png' % (ofolder, sim_name))


def test_static_Vm_distribution(cell_params, ofolder, conductance_type):
    
    neuron.h('forall delete_section()')
    neuron.h('secondorder=2')
    cell_params['tstartms'] = 0
    cell_params['tstopms'] = 500
    cell_params['timeres_NEURON'] = 2**-5
    cell_params['timeres_python'] = 2**-5
    cell_params.pop('v_init')
    cell = LFPy.Cell(**cell_params)
    cell.simulate(rec_vmem=True)
    plt.subplot(131, aspect='equal', frameon=False, xticks=[], yticks=[])
    plt.title('Static Vm distribution after totally %d ms of rest' %(cell_params['tstopms'] - cell_params['tstartms']))
    plt.scatter(cell.ymid, cell.zmid, c=cell.vmem[:,-1], s=5, edgecolor='none')
    plt.colorbar()
    plt.subplot(222)
    plt.title('All compartments Vm last %d ms' % cell_params['tstopms'])
    for comp in xrange(len(cell.xmid)):
        plt.plot(cell.tvec, cell.vmem[comp,:], 'k')
    plt.xlim(-5, cell.tvec[-1] + 5)
    plt.ylim(-80, -62)
    plt.subplot(224)
    plt.title('All compartments Vm shifted to 0 at start of last %d ms' % cell_params['tstopms'])
    for comp in xrange(len(cell.xmid)):
        plt.plot(cell.tvec[-100:], cell.vmem[comp,-100:] - cell.vmem[comp,-100], 'k')
    plt.xlim(cell.tvec[-100], cell.tvec[-1] + 1)
    #plt.ylim(-80, -62)
    plt.savefig(join(ofolder, 'Vm_distribution_control.png'))

    plt.close('all')
    del cell
    neuron.h('forall delete_section()')
    neuron.h('secondorder=2')
    cell_params['tstopms'] = 500
    cell = LFPy.Cell(**cell_params)
    static_Vm = np.load(join(ofolder, 'static_Vm_distribution.npy'))
    for idx in xrange(len(static_Vm)):
        v_clamp = {'idx' : idx,
                   'record_current' : True,
                   'pptype' : 'VClamp',
                   'amp[0]' : static_Vm[idx],
                   'dur[0]' : 100,
                   }
        stimulus = LFPy.StimIntElectrode(cell, **v_clamp)
        
    cell.simulate(rec_vmem=True)
    plt.subplot(131, aspect='equal', frameon=False, xticks=[], yticks=[])
    plt.title('Static Vm distribution after totally %d ms of rest' %(cell_params['tstopms'] - cell_params['tstartms']))
    plt.scatter(cell.ymid, cell.zmid, c=cell.vmem[:,-1], s=5, edgecolor='none')
    #plt.subplot(222, aspect='equal', frameon=False, xticks=[], yticks=[])
    #plt.scatter(cell.xmid, cell.zmid, c=cell.vmem[:,-1], s=2, edgecolor='none')
    plt.colorbar()
    plt.subplot(222)
    plt.title('All compartments Vm last %d ms' % cell_params['tstopms'])
    for comp in xrange(len(cell.xmid)):
        plt.plot(cell.tvec, cell.vmem[comp,:], 'k')
    plt.xlim(-5, cell.tvec[-1] + 5)
    plt.ylim(-80, -62)
    plt.subplot(224)
    plt.title('All compartments Vm shifted to 0 at start of last %d ms' % cell_params['tstopms'])
    for comp in xrange(len(cell.xmid)):
        plt.plot(cell.tvec[-100:], cell.vmem[comp,-100:] - cell.vmem[comp,-100], 'k')
    #plt.ylim(-80, -62)
    plt.xlim(cell.tvec[-100], cell.tvec[-1] + 1)
    plt.savefig(join(ofolder, 'Vm_distribution_set_to_static.png'))
    
def find_static_Vm_distribution(cell_params, ofolder, conductance_type, epas=None):

    neuron.h('forall delete_section()')
    neuron.h('secondorder=2')
    cell_params['tstartms'] = -6000
    cell_params['tstopms'] = 200
    cell_params['timeres_NEURON'] = 2**-4
    cell_params['timeres_python'] = 2**-4
    cell = LFPy.Cell(**cell_params)
    #if not epas == None:
    #    for sec in cell.allseclist:
    #        for seg in sec:
    #            seg.pas.e = epas
    cell.simulate(rec_vmem=True)
    plt.subplot(131, aspect='equal', frameon=False, xticks=[], yticks=[])
    plt.title('Static Vm distribution after totally %d ms of rest' %(cell_params['tstopms'] - cell_params['tstartms']))
    plt.scatter(cell.ymid, cell.zmid, c=cell.vmem[:,-1], s=5, edgecolor='none')
    #plt.subplot(222, aspect='equal', frameon=False, xticks=[], yticks=[])
    #plt.scatter(cell.xmid, cell.zmid, c=cell.vmem[:,-1], s=2, edgecolor='none')
    plt.colorbar()
    plt.subplot(222)
    plt.title('All compartments Vm last %d ms' % cell_params['tstopms'])
    for comp in xrange(len(cell.xmid)):
        plt.plot(cell.tvec, cell.vmem[comp,:], 'k')
    plt.xlim(-5, cell.tvec[-1] + 5)
    plt.subplot(224)
    plt.title('All compartments Vm shifted to 0 at start of last %d ms' % cell_params['tstopms'])
    for comp in xrange(len(cell.xmid)):
        plt.plot(cell.tvec, cell.vmem[comp,:] - cell.vmem[comp,0] , 'k')
    plt.xlim(-5, cell.tvec[-1] + 5)
    plt.savefig(join(ofolder, 'static_Vm_distribution.png'))
    np.save(join(ofolder, 'static_Vm_distribution.npy'), cell.vmem[:,-1])

def check_current_sum(cell, noiseVec):
    const = (1E-2 * cell.area)
    active_current = np.zeros(cell.imem.shape)
    for i in cell.rec_variables:
        active_current += cell.rec_variables[i]
    
    active_current = np.array([const[idx] * active_current[idx,:] 
                               for idx in xrange(len(cell.imem))])
    #set_trace()
    summed_current = active_current + cell.ipas + cell.icap
    summed_current[0,:] += np.array(noiseVec)
    idx = 0
    plt.close('all')
    plt.plot(cell.tvec, cell.icap[idx,:], 'k')
    plt.plot(cell.tvec, cell.ipas[idx,:], 'g')
    if idx == 0:
        plt.plot(cell.tvec, noiseVec, 'y')
    #plt.plot(cell.tvec, cell.icap[idx,:] - noiseVec, 'm')
    plt.plot(cell.tvec, cell.imem[idx, :], 'r')
    plt.plot(cell.tvec, summed_current[idx, :], 'r--')
    error = (cell.imem - summed_current)
    print np.sqrt(np.average(error**2)), np.max(np.abs(error))
    plt.show()
    sys.exit()
    
def run_WN_simulation(cell_params, input_scaling, input_idx, 
                   ofolder, ntsteps, simulation_params, conductance_type, epas=None):
    neuron.h('forall delete_section()')
    neuron.h('secondorder=2')
    cell = LFPy.Cell(**cell_params)

    if epas == None:
        sim_name = '%d_%1.3f_%s' %(input_idx, input_scaling, conductance_type)
    else:
        sim_name = '%d_%1.3f_%s_%g' %(input_idx, input_scaling, conductance_type, epas)
    
    input_array = input_scaling * \
                  np.load(join(ofolder, 'input_array.npy'))
    noiseVec = neuron.h.Vector(input_array)
    i = 0

    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if not epas == None:
                seg.pas.e = epas
            if i == input_idx:
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if type(syn) == type(None):
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noiseVec.play(syn._ref_amp, cell.timeres_NEURON)
    #set_trace()    
    #mapping = np.load(join(ofolder, 'mapping.npy'))
    cell.simulate(**simulation_params)
    # Cutting of start of simulations
    
    cut_list = ['cell.imem', 'cell.somav', 'input_array', 'cell.ipas', 'cell.icap']
    if hasattr(cell, 'rec_variables'):
        for cur in cell.rec_variables:
            cut_list.append('cell.rec_variables["%s"]' % cur)
    for cur in cut_list:
        try:
            exec('%s = %s[:,-%d:]' %(cur, cur, ntsteps))
        except IndexError:
            exec('%s = %s[-%d:]' %(cur, cur, ntsteps))
            #cur = cur[-ntsteps:]
    cell.tvec = cell.tvec[-ntsteps:] - cell.tvec[-ntsteps]
    timestep = (cell.tvec[1] - cell.tvec[0])/1000.

    np.save(join(ofolder, 'istim_%s.npy' %(sim_name)), input_array)
    np.save(join(ofolder, 'tvec.npy'), cell.tvec)
    const = (1E-2 * cell.area)
    if hasattr(cell, 'rec_variables'):
        for cur in cell.rec_variables:
            active_current = np.array([const[idx] * cell.rec_variables[cur][idx,:] 
                                       for idx in xrange(len(cell.imem))])
            psd, freqs = find_LFP_PSD(active_current, timestep)
            np.save(join(ofolder, '%s_%s.npy' %(cur, sim_name)), active_current)
            np.save(join(ofolder, '%s_psd_%s.npy' %(cur, sim_name)), psd)
        
    #vmem_quickplot(cell, input_array, sim_name, ofolder)
    #sig = np.dot(mapping, cell.imem)
    #sig_psd, freqs = find_LFP_PSD(sig, timestep)
    #np.save(join(ofolder, 'signal_%s.npy' %(sim_name)), sig)
    #np.save(join(ofolder, 'psd_%s.npy' %(sim_name)), sig_psd)
    
    somav_psd, freqs = find_LFP_PSD(np.array([cell.somav]), timestep)
    np.save(join(ofolder, 'somav_psd_%s.npy' %(sim_name)), somav_psd[0])
    np.save(join(ofolder, 'somav_%s.npy' %(sim_name)), cell.somav)
    
    imem_psd, freqs = find_LFP_PSD(cell.imem, timestep)
    np.save(join(ofolder, 'imem_psd_%s.npy' %(sim_name)), imem_psd)
    np.save(join(ofolder, 'imem_%s.npy' %(sim_name)), cell.imem)

    icap_psd, freqs = find_LFP_PSD(cell.icap, timestep)
    np.save(join(ofolder, 'icap_psd_%s.npy' %(sim_name)), icap_psd)
    np.save(join(ofolder, 'icap_%s.npy' %(sim_name)), cell.icap)

    ipas_psd, freqs = find_LFP_PSD(cell.ipas, timestep)
    np.save(join(ofolder, 'ipas_psd_%s.npy' %(sim_name)), ipas_psd)
    np.save(join(ofolder, 'ipas_%s.npy' %(sim_name)), cell.ipas)

    #ymid = np.load(join(ofolder, 'ymid.npy'))
    #stick = aLFP.return_dipole_stick(cell.imem, ymid)
    #stick_psd, freqs = find_LFP_PSD(stick, timestep)
    #np.save(join(ofolder, 'stick_%s.npy' %(sim_name)), stick)
    #np.save(join(ofolder, 'stick_psd_%s.npy' %(sim_name)), stick_psd)
    
def run_synaptic_simulation(cell_params, input_scaling, input_idx, 
                   ofolder, ntsteps, simulation_params, conductance_type, epas=None):

    neuron.h('forall delete_section()')
    neuron.h('secondorder=2')
    cell = LFPy.Cell(**cell_params)

    if epas == None:
        sim_name = '%d_%1.3f_%s' %(input_idx, input_scaling, conductance_type)
    else:
        sim_name = '%d_%1.3f_%s_%g' %(input_idx, input_scaling, conductance_type, epas)
        for sec in cell.allseclist:
            for seg in sec:
                seg.pas.e = epas
        
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
    
    #mapping = np.load(join(ofolder, 'mapping.npy'))
    cell.simulate(**simulation_params)
    #set_trace()
    
    ###check_current_sum(cell, noiseVec)
    #set_trace()

    # Cutting of start of simulations
    
    #cut_list = ['cell.imem', 'cell.somav', 'input_array', 'cell.ipas', 'cell.icap']
    #if hasattr(cell, 'rec_variables'):
    #    for cur in cell.rec_variables:
    #        cut_list.append('cell.rec_variables["%s"]' % cur)
    #for cur in cut_list:
    #    try:
    #        exec('%s = %s[:,-%d:]' %(cur, cur, ntsteps))
    #    except IndexError:
    #        exec('%s = %s[-%d:]' %(cur, cur, ntsteps))
    #        #cur = cur[-ntsteps:]
    #cell.tvec = cell.tvec[-ntsteps:] - cell.tvec[-ntsteps]
    timestep = (cell.tvec[1] - cell.tvec[0])/1000.
    np.save(join(ofolder, 'tvec.npy'), cell.tvec)
    const = (1E-2 * cell.area)
    if hasattr(cell, 'rec_variables'):
        for cur in cell.rec_variables:
            active_current = np.array([const[idx] * cell.rec_variables[cur][idx,:] 
                                       for idx in xrange(len(cell.imem))])
            psd, freqs = find_LFP_PSD(active_current, timestep)
            np.save(join(ofolder, '%s_%s.npy' %(cur, sim_name)), active_current)
            np.save(join(ofolder, '%s_psd_%s.npy' %(cur, sim_name)), psd)
        
    #vmem_quickplot(cell, input_array, sim_name, ofolder)
    #sig = np.dot(mapping, cell.imem)
    #sig_psd, freqs = find_LFP_PSD(sig, timestep)
    #np.save(join(ofolder, 'signal_%s.npy' %(sim_name)), sig)
    #np.save(join(ofolder, 'psd_%s.npy' %(sim_name)), sig_psd)
    
    somav_psd, freqs = find_LFP_PSD(np.array([cell.somav]), timestep)
    np.save(join(ofolder, 'somav_psd_%s.npy' %(sim_name)), somav_psd[0])
    np.save(join(ofolder, 'somav_%s.npy' %(sim_name)), cell.somav)
    
    imem_psd, freqs = find_LFP_PSD(cell.imem, timestep)
    np.save(join(ofolder, 'imem_psd_%s.npy' %(sim_name)), imem_psd)
    np.save(join(ofolder, 'imem_%s.npy' %(sim_name)), cell.imem)

    icap_psd, freqs = find_LFP_PSD(cell.icap, timestep)
    np.save(join(ofolder, 'icap_psd_%s.npy' %(sim_name)), icap_psd)
    np.save(join(ofolder, 'icap_%s.npy' %(sim_name)), cell.icap)

    ipas_psd, freqs = find_LFP_PSD(cell.ipas, timestep)
    np.save(join(ofolder, 'ipas_psd_%s.npy' %(sim_name)), ipas_psd)
    np.save(join(ofolder, 'ipas_%s.npy' %(sim_name)), cell.ipas)

    ymid = np.load(join(ofolder, 'ymid.npy'))
    stick = aLFP.return_dipole_stick(cell.imem, ymid)
    stick_psd, freqs = find_LFP_PSD(stick, timestep)
    np.save(join(ofolder, 'stick_%s.npy' %(sim_name)), stick)
    np.save(join(ofolder, 'stick_psd_%s.npy' %(sim_name)), stick_psd)
    np.save(join(ofolder, 'freqs.npy'), freqs)



def run_linearized_simulation(cell_params, input_scaling, input_idx, 
                   ofolder, ntsteps, simulation_params, conductance_type, 
                   input_type):

    neuron.h('forall delete_section()')
    neuron.h('secondorder=2')
    static_Vm = np.load(join(ofolder, 'static_Vm_distribution.npy'))
    cell_params['v_init'] = np.average(static_Vm)
    cell = LFPy.Cell(**cell_params)
    sim_name = '%d_%1.3f_%s' %(input_idx, input_scaling, conductance_type)
    print sim_name
    if not conductance_type == 'active':
        comp_idx = 0
        for sec in cell.allseclist:
            for seg in sec:
                exec('seg.vss_%s = static_Vm[%d]'% (conductance_type, comp_idx))
                comp_idx += 1

    if input_type == 'synapse':
        cell, save_method = make_synapse_stimuli(cell, input_idx, input_scaling)
    elif input_type == 'ZAP':
        cell, save_method = make_ZAP_stimuli(cell, input_idx, input_scaling)
    elif input_type == 'WN':
        cell, save_method = make_WN_stimuli(cell, input_idx, input_scaling, ofolder)
    else:
        raise RuntimeError("No known 'input_type'")
    cell.simulate(**simulation_params)
    save_method(cell, sim_name, ofolder, static_Vm, input_idx, ntsteps)


def make_synapse_stimuli(cell, input_idx, input_scaling):
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
    save_method = save_simple
    return cell, save_method

def make_ZAP_stimuli(cell, input_idx, input_scaling):
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
    save_method = save_simple
    return cell, save_method

def make_WN_stimuli(cell, input_idx, input_scaling, ofolder):
    input_array = input_scaling * np.load(join(ofolder, 'input_array.npy'))
    noiseVec = neuron.h.Vector(input_array)
    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if type(syn) == type(None):
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noiseVec.play(syn._ref_amp, cell.timeres_NEURON)
    save_method = save_WN_data
    return cell, save_method
    
def save_simple(cell, sim_name, ofolder, static_Vm, input_idx, ntsteps):

    timestep = (cell.tvec[1] - cell.tvec[0])/1000.
    np.save(join(ofolder, 'tvec.npy'), cell.tvec)

    mapping = np.load(join(ofolder, 'mapping.npy'))
    sig = 1000 * np.dot(mapping, cell.imem)
    #sig_psd, freqs = find_LFP_PSD(sig, timestep)
    #np.save(join(ofolder, 'sig_psd_%s.npy' %(sim_name)), sig_psd)
    np.save(join(ofolder, 'sig_%s.npy' %(sim_name)), sig)
    
    linearized_quickplot(cell, sim_name, ofolder, static_Vm, input_idx)
    #vmem_psd, freqs = find_LFP_PSD(cell.vmem, timestep)
    #np.save(join(ofolder, 'vmem_psd_%s.npy' %(sim_name)), vmem_psd)
    np.save(join(ofolder, 'vmem_%s.npy' %(sim_name)), cell.vmem)
    #imem_psd, freqs = find_LFP_PSD(imem, timestep)
    #np.save(join(ofolder, 'imem_psd_%s.npy' %(sim_name)), imem_psd)
    #np.save(join(ofolder, 'imem_%s.npy' %(sim_name)), imem)
    #np.save(join(ofolder, 'freqs.npy'), freqs)

    

def save_WN_data(cell, sim_name, ofolder, static_Vm, input_idx, ntsteps):
    timestep = (cell.tvec[1] - cell.tvec[0])/1000.
    np.save(join(ofolder, 'tvec.npy'), cell.tvec[:ntsteps])


    cell.imem = cell.imem[:,-ntsteps:]
    cell.vmem = cell.vmem[:,-ntsteps:]
    
    mapping = np.load(join(ofolder, 'mapping.npy'))
    sig = 1000 * np.dot(mapping, cell.imem)
    sig_psd, freqs = find_LFP_PSD(sig, timestep)
    np.save(join(ofolder, 'sig_psd_%s.npy' %(sim_name)), sig_psd)
    np.save(join(ofolder, 'sig_%s.npy' %(sim_name)), sig)
    
    vmem_psd, freqs = find_LFP_PSD(cell.vmem, timestep)
    np.save(join(ofolder, 'vmem_psd_%s.npy' %(sim_name)), vmem_psd)
    np.save(join(ofolder, 'vmem_%s.npy' %(sim_name)), cell.vmem)

    imem_psd, freqs = find_LFP_PSD(cell.imem, timestep)
    np.save(join(ofolder, 'imem_psd_%s.npy' %(sim_name)), imem_psd)
    np.save(join(ofolder, 'imem_%s.npy' %(sim_name)), cell.imem)
    np.save(join(ofolder, 'freqs.npy'), freqs)

    linearized_quickplot_frequency(cell, imem_psd, vmem_psd, freqs, sim_name, 
                                   ofolder, static_Vm, input_idx)

def pos_quickplot(cell, cell_name, elec_x, elec_y, elec_z, ofolder):
    plt.close('all')
    plt.subplot(121)
    plt.scatter(cell.xmid, cell.ymid, s=cell.diam)
    plt.scatter(elec_x, elec_y, c='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.subplot(122)
    plt.scatter(cell.zmid, cell.ymid, s=cell.diam)
    plt.scatter(elec_z, elec_y, c='r')
    plt.xlabel('z')
    plt.ylabel('y')    
    plt.axis('equal')
    plt.savefig(join(ofolder, 'pos_%s.png' % cell_name))



def linearized_quickplot_frequency(cell, imem_psd, vmem_psd, freqs, sim_name, 
                                   ofolder, static_Vm, input_idx):
    plt.close('all')
    plt.subplots_adjust(hspace=0.3)
    ax0 = plt.subplot(161, aspect='equal', frameon=False, xticks=[], yticks=[],
                title='Active\nsteady state')
    vmax = np.max([static_Vm, cell.vmem[:,0]])
    vmin = np.min([static_Vm, cell.vmem[:,0]])
    plt.scatter(cell.xmid, cell.zmid, c=static_Vm, s=8, edgecolor='none',
                vmax=vmax, vmin=vmin)
    plt.subplot(162, aspect='equal', frameon=False, xticks=[], yticks=[],
                title='Initial\n%s' %sim_name, sharey=ax0)
    plt.scatter(cell.xmid, cell.zmid, c=cell.vmem[:,0], s=8, edgecolor='none',
                vmax=vmax, vmin=vmin)    
    plt.colorbar()
    
    plt.subplot(243)
    plt.title('Soma imem amp [nA]')
    plt.xlabel('Hz')
    plt.yscale('log')
    plt.xlim(0,110)
    plt.plot(freqs, imem_psd[0,:])
    
    plt.subplot(247)
    plt.title('Soma vmem amp')
    plt.xlabel('Hz')
    plt.yscale('log')
    plt.xlim(0,110)
    plt.plot(freqs, vmem_psd[0,:])
    plt.plot(0, static_Vm[0], 'ro')

    plt.subplot(244)
    plt.title('Input imem amp [nA]')
    plt.xlabel('Hz')
    plt.yscale('log')
    plt.xlim(0,110)
    plt.plot(freqs, imem_psd[input_idx,:])
    
    plt.subplot(248)
    plt.title('Input vmem amp')    
    plt.xlabel('Hz')
    plt.yscale('log')
    plt.xlim(0,110)
    plt.plot(freqs, vmem_psd[input_idx,:])
    plt.savefig(join(ofolder, 'current_psd_%s.png' %sim_name))

def linearized_quickplot(cell, sim_name, ofolder, static_Vm, input_idx):
    plt.close('all')
    plt.subplots_adjust(hspace=0.3)
    ax0 = plt.subplot(161, aspect='equal', frameon=False, xticks=[], yticks=[],
                title='Active\nsteady state')
    vmax = np.max([static_Vm, cell.vmem[:,0]])
    vmin = np.min([static_Vm, cell.vmem[:,0]])
    plt.scatter(cell.xmid, cell.zmid, c=static_Vm, s=8, edgecolor='none',
                vmax=vmax, vmin=vmin)
    plt.subplot(162, aspect='equal', frameon=False, xticks=[], yticks=[],
                title='Initial\n%s' %sim_name, sharey=ax0)
    plt.scatter(cell.xmid, cell.zmid, c=cell.vmem[:,0], s=8, edgecolor='none',
                vmax=vmax, vmin=vmin)    
    plt.colorbar()
    plt.subplot(243)
    plt.title('Soma imem [nA]')
    #plt.ylim(-0.04, 0.04)    
    plt.plot(cell.tvec, cell.imem[0,:])
    #plt.ylim(-0.1, 0.02)
    plt.subplot(247)
    #plt.ylim(-80, -70)
    plt.title('Soma vmem')
    plt.plot(cell.tvec, cell.somav)
    plt.plot(0, static_Vm[0], 'ro')
    plt.subplot(244)
    plt.title('Input imem [nA]')
    #plt.ylim(-0.04, 0.04)    
    plt.plot(cell.tvec, cell.imem[input_idx,:])
    #plt.ylim(-0.1, 0.02)
    try:
        plt.subplot(248)
        #plt.ylim(-80, -70)
        plt.title('Input vmem')
        plt.plot(0, static_Vm[input_idx], 'ro')
        plt.plot(cell.tvec, cell.vmem[input_idx,:])
    except:
        pass    
    plt.savefig(join(ofolder, 'current_%s.png' %sim_name))

def vmem_quickplot(cell, sim_name, ofolder, input_array=None):
    plt.close('all')
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(311)
    plt.title('Soma imem [nA]')
    #plt.ylim(-0.04, 0.04)    
    plt.plot(cell.tvec, cell.imem[0,:])
    plt.ylim(-0.1, 0.02)
    plt.subplot(312)
    plt.ylim(-77, -74)
    plt.title('Soma vmem')
    plt.plot(cell.tvec, cell.somav)
    if not input_array == None:
        plt.subplot(313)
        plt.title('Input current')
        #plt.ylim(-0.4, 0.4)
        plt.plot(cell.tvec, input_array)   
    plt.savefig(join(ofolder, 'current_%s.png' %sim_name))

def initialize_cell(cell_params, pos_params, rot_params, cell_name, 
                    elec_x, elec_y, elec_z, ntsteps, ofolder, testing=False, make_WN_input=False):
    """ Position and plot a cell """
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
    length = np.sqrt((cell.xend - cell.xstart)**2 +
                     (cell.yend - cell.ystart)**2 +
                     (cell.zend - cell.zstart)**2)
    np.save(join(ofolder, 'mapping.npy' ), electrode.electrodecoeff)
    np.save(join(ofolder, 'xstart.npy' ), cell.xstart)
    np.save(join(ofolder, 'ystart.npy' ), cell.ystart)
    np.save(join(ofolder, 'zstart.npy' ), cell.zstart)
    np.save(join(ofolder, 'xend.npy' ), cell.xend)
    np.save(join(ofolder, 'yend.npy' ), cell.yend)
    np.save(join(ofolder, 'zend.npy' ), cell.zend) 
    np.save(join(ofolder, 'xmid.npy' ), cell.xmid)
    np.save(join(ofolder, 'ymid.npy' ), cell.ymid)
    np.save(join(ofolder, 'zmid.npy' ), cell.zmid)
    np.save(join(ofolder, 'length.npy' ), length)
    np.save(join(ofolder, 'diam.npy' ), cell.diam)
    
    if make_WN_input:
        timestep = cell_params['timeres_NEURON']/1000
        # Making unscaled white noise input array
        input_array = aLFP.make_WN_input(cell_params)
        sample_freq = ff.fftfreq(len(input_array[-ntsteps:]), d=timestep)
        pidxs = np.where(sample_freq >= 0)
        freqs = sample_freq[pidxs]
        Y = ff.fft(input_array[-ntsteps:])[pidxs]
        power = np.abs(Y)/len(Y)
        np.save(join(ofolder, 'input_array.npy' ), input_array)
        np.save(join(ofolder, 'input_array_psd.npy' ), power)
        np.save(join(ofolder, 'freqs.npy' ), freqs)
