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
import pylab as pl
from os.path import join
import cPickle
import aLFP
import scipy.fftpack as ff



pl.rcParams.update({'font.size' : 8,
    'figure.facecolor' : '1',
    'wspace' : 0.5, 'hspace' : 0.5})
np.random.seed(1234)

def run_all_simulations(cell_params, is_active, model, input_idxs, input_scalings, ntsteps):

    for input_idx in input_idxs:
        for input_scaling in input_scalings:
            run_simulation(cell_params, input_scaling, is_active, input_idx, model, ntsteps)        

def find_LFP_PSD(sig, timestep):
    """ Returns the power and freqency of the input signal"""
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq > 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig, axis=1)[:,pidxs[0]]
    power = np.abs(Y)/Y.shape[1]
    return power, freqs

def return_power(sig, timestep):
    """ Returns the power of the input signal"""
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq > 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig)[pidxs]
    power = np.abs(Y)/len(Y)
    return power
            
            
def run_simulation(cell_params, input_scaling, is_active, input_idx, ofolder, ntsteps):

    neuron.h('forall delete_section()')
    neuron.h('secondorder=2')
    cell = LFPy.Cell(**cell_params)

    sim_name = '%d_%1.3f_%s' %(input_idx, input_scaling, bool(is_active))
    input_array = input_scaling * \
                  np.load(join(ofolder, 'input_array.npy'))
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
    mapping = np.load(join(ofolder, 'mapping.npy'))
    simulation_params = {'rec_imem': True,
                         'rec_istim': True,
                         }
    cell.simulate(**simulation_params)

    cell.tvec = cell.tvec[-ntsteps:] - cell.tvec[-ntsteps]
    cell.imem = cell.imem[:,-ntsteps:]
    cell.somav = cell.somav[-ntsteps:]
    input_array = input_array[-ntsteps:]
    
    vmem_quickplot(cell, input_array, sim_name, ofolder)
    sig = np.dot(mapping, cell.imem)

    sig_psd, freqs = find_LFP_PSD(sig, cell.tvec[1] - cell.tvec[0])
    somav_psd, freqs = find_LFP_PSD(np.array([cell.somav]), cell.tvec[1] - cell.tvec[0])
    imem_psd, freqs = find_LFP_PSD(cell.imem, cell.tvec[1] - cell.tvec[0])

    ymid = np.load(join(ofolder, 'ymid.npy'))
    stick = aLFP.return_dipole_stick(cell.imem, ymid)
    stick_psd, freqs = find_LFP_PSD(stick, cell.tvec[1] - cell.tvec[0])
    
    np.save(join(ofolder, 'stick_%s.npy' %(sim_name)), stick)
    np.save(join(ofolder, 'stick_psd_%s.npy' %(sim_name)), stick_psd)
    np.save(join(ofolder, 'somav_psd_%s.npy' %(sim_name)), somav_psd[0])
    np.save(join(ofolder, 'imem_psd_%s.npy' %(sim_name)), imem_psd)
    np.save(join(ofolder, 'imem_%s.npy' %(sim_name)), cell.imem)
    np.save(join(ofolder, 'somav_%s.npy' %(sim_name)), cell.somav)
    np.save(join(ofolder, 'istim_%s.npy' %(sim_name)), input_array)
    np.save(join(ofolder, 'tvec.npy'), cell.tvec)
    np.save(join(ofolder, 'signal_%s.npy' %(sim_name)), sig)
    np.save(join(ofolder, 'psd_%s.npy' %(sim_name)), sig_psd)
    
def pos_quickplot(cell, cell_name, elec_x, elec_y, elec_z, ofolder):
    pl.close('all')
    pl.subplot(121)
    pl.scatter(cell.xmid, cell.ymid, s=cell.diam)
    pl.scatter(elec_x, elec_y, c='r')
    pl.xlabel('x')
    pl.ylabel('y')
    pl.axis('equal')
    pl.subplot(122)
    pl.scatter(cell.zmid, cell.ymid, s=cell.diam)
    pl.scatter(elec_z, elec_y, c='r')
    
    pl.xlabel('z')
    pl.ylabel('y')    
    pl.axis('equal')
    
    pl.savefig(join(ofolder, 'pos_%s.png' % cell_name))


def vmem_quickplot(cell, input_array, sim_name, ofolder):
    pl.close('all')
    pl.subplots_adjust(hspace=0.3)
    pl.subplot(311)
    pl.title('Soma imem [nA]')
    #pl.ylim(-0.04, 0.04)    
    pl.plot(cell.tvec, cell.imem[0,:])
    pl.subplot(312)
    #pl.ylim(-90, -20)
    pl.title('Soma vmem')
    pl.plot(cell.tvec, cell.somav)
    pl.subplot(313)
    pl.title('Input current')
    #pl.ylim(-0.4, 0.4)
    pl.plot(cell.tvec, input_array)   
    pl.savefig(join(ofolder, 'current_%s.png' %sim_name))

    
def initialize_cell(cell_params, pos_params, rot_params, cell_name, 
                    elec_x, elec_y, elec_z, ntsteps, ofolder, testing=False):
    """ Position and plot a cell """
    neuron.h('forall delete_section()')
    try:
        os.mkdir(ofolder)
    except OSError:
        pass
    foo_params = cell_params.copy()
    foo_params['tstopms'] = 1
    cell = LFPy.Cell(**foo_params)
    cell.set_rotation(**rot_params)
    cell.set_pos(**pos_params)       
    if testing:
        aLFP.plot_comp_numbers(cell)
    
    timestep = cell_params['timeres_NEURON']/1000
    # Making unscaled white noise input array
    input_array = aLFP.make_WN_input(cell_params)

    sample_freq = ff.fftfreq(len(input_array[-ntsteps:]), d=timestep)
    pidxs = np.where(sample_freq > 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(input_array[-ntsteps:])[pidxs]
    power = np.abs(Y)/len(Y)

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
    np.save(join(ofolder, 'input_array.npy' ), input_array)
    np.save(join(ofolder, 'input_array_psd.npy' ), power)
    np.save(join(ofolder, 'freqs.npy' ), freqs)
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
