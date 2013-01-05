#!/usr/bin/env python

import LFPy
import numpy as np
import neuron
import os
import sys
from ipdb import set_trace
import pylab as pl
from os.path import join
import scipy.fftpack as ff

pl.rcParams.update({'font.size' : 12,
    'figure.facecolor' : '1',
    'wspace' : 0.5, 'hspace' : 0.5})

neuron_model = os.path.join('neuron_models', 'mainen')
conductances = sys.argv[1]

r_m = 30000
try:
    r_m = float(sys.argv[2])
except:
    pass

if conductances == 'active':
    is_active = True
elif conductances == 'passive':
    is_active = False
elif conductances == 'renormalized':
    is_active = False
#else:
#    print "active, passive or renormalized?"
#    raise RuntimeError

domain = 'phase'
input_amp = 0.001
input_delay = 600

try:
    os.mkdir(domain)
except OSError:
    pass

np.random.seed(1234)
LFPy.cell.neuron.load_mechanisms(neuron_model)

def find_renormalized_Rm(cell, stim_idx):

    # r_tilde = [mV] / ( [nA] / [um]**2) = 10**-2 [Ohm] [cm]**2
    r_tilde = (cell.somav[-1] - cell.somav[stim_idx-1])/\
              (cell.imem[0,-1] - cell.imem[0, stim_idx-1]) * cell.area[0] * 10**-2
    print "Soma area: ", cell.area[0]
    print "Renormalized rm: ", r_tilde
    #pl.plot(cell.tvec, R)
    #pl.show()
    #set_trace()
    return r_tilde


def active_declarations(is_active = True):
    '''set active conductances and correct for spines,
    see file active_declarations_example3.hoc'''
    spine_dens = 1
    spine_area = 0.83 # // um^2  -- K Harris
        
    cm_myelin = 0.04
    g_pas_node = 0.02
    celsius   = 37.
    
    Ek = -85.
    Ena = 60.
    gna_dend = 20.
    gna_node = 30000.
    gna_soma = gna_dend * 10
    
    gkv_axon = 2000.
    gkv_soma = 200.
    
    gca = .3
    gkm = .1
    gkca = 3
    gca_soma = gca
    gkm_soma = gkm
    gkca_soma = gkca
    
    dendritic = neuron.h.SectionList()
    for sec in neuron.h.allsec():
        if sec.name()[:4] == 'soma':
            dendritic.append(sec)
        if sec.name()[:4] == 'dend':
            dendritic.append(sec)
        if sec.name()[:4] == 'apic':
            dendritic.append(sec)
    
    def add_spines(section):
        '''add spines as a modification of the compartment area'''
        is_spiny = 1
        if section == "dend":
            print "adding spines"
            for sec in neuron.h.dendlist:
                a = 0
                for seg in sec:
                    a += neuron.h.area(seg.x)
                F = (sec.L*spine_area*spine_dens + a)/a
                sec.L = sec.L * F**(2./3.)
                for seg in sec:
                    seg.diam = seg.diam * F**(1./3.)
        neuron.h.define_shape()

    # Insert active channels
    def set_active():
        '''set the channel densities'''
        # exceptions along the axon
        for sec in neuron.h.myelin:
            sec.cm = cm_myelin
        for sec in neuron.h.node: 
            sec.g_pas = g_pas_node

        if is_active:
            # na+ channels
            for sec in neuron.h.allsec():
                sec.insert('na')
            for sec in dendritic:
                sec.gbar_na = gna_dend            
            for sec in neuron.h.myelin:
                sec.gbar_na = gna_dend
            for sec in neuron.h.hill:
                sec.gbar_na = gna_node
            for sec in neuron.h.iseg:
                sec.gbar_na = gna_node
            for sec in neuron.h.node:
                sec.gbar_na = gna_node

            # kv delayed rectifier channels
            neuron.h.iseg.insert('kv')
            neuron.h.iseg.gbar_kv = gkv_axon
            neuron.h.hill.insert('kv')
            neuron.h.hill.gbar_kv = gkv_axon
            for sec in neuron.h.soma:
                sec.insert('kv')
                sec.gbar_kv = gkv_soma

            # dendritic channels
            for sec in dendritic:
                sec.insert('km')
                sec.gbar_km  = gkm
                sec.insert('kca')
                sec.gbar_kca = gkca
                sec.insert('ca')
                sec.gbar_ca = gca
                sec.insert('cad')

            # somatic channels
            for sec in neuron.h.soma:
                sec.gbar_na = gna_soma
                sec.gbar_km = gkm_soma
                sec.gbar_kca = gkca_soma
                sec.gbar_ca = gca_soma

            for sec in neuron.h.allsec():
                if neuron.h.ismembrane('k_ion'):
                    sec.ek = Ek

            for sec in neuron.h.allsec():
                if neuron.h.ismembrane('na_ion'):
                    sec.ena = Ena
                    neuron.h.vshift_na = -5

            for sec in neuron.h.allsec():
                if neuron.h.ismembrane('ca_ion'):
                    sec.eca = 140
                    neuron.h.ion_style("ca_ion", 0, 1, 0, 0, 0)
                    neuron.h.vshift_ca = 0
        
        #set the temperature for the neuron dynamics
        neuron.h.celsius = celsius
        
        print "active ion-channels inserted."
        
        
        #// Insert spines
    add_spines('dend')
    
    #// Insert active channels
    set_active()
    

################################################################################
# Define parameters, using dictionaries
# It is possible to set a few more parameters for each class or functions, but
# we chose to show only the most important ones here.
################################################################################

#define cell parameters used as input to cell-class
cellParameters = {
    'morphology' : os.path.join(neuron_model, 'L5_Mainen96_wAxon_LFPy.hoc'),
    #'morphology' : os.path.join('neuron_models', 'example_morphology.hoc'),
    'rm' : r_m,               # membrane resistance
    'cm' : 1.0,                 # membrane capacitance
    'Ra' : 150,                 # axial resistance
    'v_init' : -65,             # initial crossmembrane potential
    'e_pas' : -65,              # reversal potential passive mechs
    'passive' : True,           # switch on passive mechs
    'nsegs_method' : 'lambda_f',# method for setting number of segments,
    'lambda_f' : 100,           # segments are isopotential at this frequency
    'timeres_NEURON' : 2**-5,   # dt of LFP and NEURON simulation.
    'timeres_python' : 2**-5,
    'tstartms' : -500,          #start time, recorders start at t=0
    'tstopms' : 500,           #stop time of simulation
    'custom_fun'  : [active_declarations], # will execute this function
    'custom_fun_args' : [{'is_active': is_active}],
}


# Define electrode geometry corresponding to a laminar electrode, where contact
# points have a radius r, surface normal vectors N, and LFP calculated as the
# average LFP in n random points on each contact:
N = np.empty((16, 3))
for i in xrange(N.shape[0]): N[i,] = [1, 0, 0] #normal unit vec. to contacts
# put parameters in dictionary
electrodeParameters = {
    'sigma' : 0.3,              # Extracellular potential
    'x' : np.zeros(16) + 25,      # x,y,z-coordinates of electrode contacts
    'y' : np.zeros(16),
    'z' : np.linspace(-500, 1000, 16),
    'n' : 20,
    'r' : 10,
    'N' : N,
}
sin_clamp = {
    'idx' : 0,
    'record_current' : True,
    #'amp' : 1., #[nA]
    'dur' : 10000.,
    'delay' :cellParameters['tstartms'],
    'freq' : 100,
    'phase' : 0,
    'pkamp' : input_amp,
    'pptype' : 'SinIClamp',
}

pulse_clamp = {
    'idx' : 0,
    'record_current' : True,
    'amp' : input_amp, #[nA]
    'dur' : 10000.,
    'delay' : input_delay, 
    'pptype' : 'IClamp',
}



# Parameters for the cell.simulate() call, recording membrane- and syn.-currents
simulationParameters = {
    'rec_imem' : True,  # Record Membrane currents during simulation
    'rec_isyn' : True,  # Record synaptic currents
}

def plot_ex3(cell, phase_vm, phase_im, phase_input):
    '''plotting function used by example3/4'''
    
    fig = pl.figure(figsize=[10,8])
    fig.suptitle('rm = %g' % r_m) 
    fig.subplots_adjust(hspace=0.5, wspace = 0.8)
    #fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    vm = cell.somav
    im = cell.imem[0,:]
    i_input = np.sum(cell.imem, axis=0)
    
    #plot the somatic trace
    ax = fig.add_subplot(411)
    ax.plot(cell.tvec, vm)
    ax.set_title('Soma pot. $\phi$ = %g' % round(phase_vm))
    #ax.set_xlabel('Time [ms]')
    ax.set_ylabel('[mV]')

    #plot the somatic trac
    ax = fig.add_subplot(412)
    ax.set_title('Soma imem. $\phi$ = %g' % round(phase_im))
    ax.plot(cell.tvec, im)
    #ax.set_xlabel('Time [ms]')
    ax.set_ylabel('[nA]')    

    ax = fig.add_subplot(413)
    ax.plot(cell.tvec, i_input)
    ax.set_title('Input $\phi$ = %g' % round(phase_input))
    #ax.set_xlabel('Time [ms]')
    ax.set_ylabel('[nA]')
    
    ax = fig.add_subplot(414)
    ax.plot(cell.tvec, (vm - np.average(vm))/np.max(vm - np.average(vm)), label='Vm')
    ax.plot(cell.tvec, (im - np.average(im))/np.max(im - np.average(im)), label='Im')
    ax.plot(cell.tvec, (i_input - np.average(i_input))/np.max(i_input - np.average(i_input)), label='Input')
    ax.legend()
    ax.set_xlabel('Time [ms]')
      
    return fig


def freq_stuff(sig):

    time_step = cellParameters['timeres_NEURON']/1000
    sample_freq = ff.fftfreq(len(cell.tvec), d=time_step)
    pidxs = np.where(sample_freq > 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig)[pidxs]
    power = np.abs(Y)/len(Y)
    offset = (np.abs(ff.fft(sig))/len(ff.fft(sig)))[0]
    #pl.plot(sig)
    #pl.show()
    #pl.plot(freqs, power)
    #pl.show()
    freqIndex = power[:].argmax()
    freq = freqs[freqIndex]
    amp = power[freqIndex]
    phase = np.angle(Y[freqIndex], deg=1)
    return freq, amp, phase, offset

cell = LFPy.Cell(**cellParameters)
currentClamp = LFPy.StimIntElectrode(cell, **sin_clamp)
cell.simulate(**simulationParameters)

freq_vm, amp_vm, phase_vm, offset_vm = freq_stuff(cell.somav)
freq_im, amp_im, phase_im, offset_im = freq_stuff(cell.imem[0,:])
freq_input, amp_input, phase_input, offset_input = freq_stuff(np.sum(cell.imem, axis=0))



print freq_vm, amp_vm, phase_vm, offset_vm
print freq_im, amp_im, phase_im, offset_im
print freq_input, amp_input, phase_input, offset_input



#stim_idx = np.argmin( np.abs(cell.tvec - input_delay))
#print cell.tvec[stim_idx]

np.save(join(domain, 'somav_%s_%g.npy' % (conductances, r_m)), cell.somav)
np.save(join(domain, 'somai_%s_%g.npy' % (conductances, r_m)), cell.imem[0,:])
#t_tilde = find_renormalized_Rm(cell, stim_idx)

#electrode = LFPy.RecExtElectrode(cell, **electrodeParameters)

#print 'simulating LFPs....'
#electrode.calc_lfp()
fig = plot_ex3(cell, phase_vm, phase_im, phase_input)
fig.savefig(join(domain, '%s_%g_%g.png' % (conductances, input_amp, r_m)))
#pl.show()
#pl.show()
