#!/usr/bin/env python
import LFPy
import numpy as np
import neuron
import os
import sys
from ipdb import set_trace
import pylab as pl
from os.path import join
import cPickle

pl.rcParams.update({'font.size' : 8,
    'figure.facecolor' : '1',
    'wspace' : 0.5, 'hspace' : 0.5})
np.random.seed(1234)

def active_declarations(is_active=True):
    '''set active conductances and correct for spines,
    see file active_declarations_example3.hoc'''
    spine_dens = 1
    spine_area = 0.83 # // um^2  -- K Harris
        
    cm_myelin = 0.04
    g_pas_node = 0.02
    celsius   = 37.
    
    Ek = -85.
    Ena = 60.

    if is_active:
        g_adjust = 1
    else:
        g_adjust = 0

    block = 0
    
    gna_dend = 20. * g_adjust
    gna_node = 30000. * g_adjust
    gna_soma = gna_dend * 10
    
    gkv_axon = 2000. * g_adjust
    gkv_soma = 200. * g_adjust
    
    gca = .3 * g_adjust
    gkm = .1 * g_adjust
    gkca = 3 * g_adjust
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

    # Insert active channels
    def set_active():
        '''set the channel densities'''
        # exceptions along the axon
        for sec in neuron.h.myelin:
            sec.cm = cm_myelin
        for sec in neuron.h.node: 
            sec.g_pas = g_pas_node

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
    #// Insert active channels
    set_active()

def return_r_m_tilde(cell):
    r_tilde = (cell.somav[-1] - cell.somav[0])/\
              (cell.imem[0,-1] - cell.imem[0, 0]) * cell.area[0] * 10**-2
    return r_tilde

def return_time_const(cell):
    start_t_idx = np.argmin(np.abs(cell.tvec - input_delay))
    v = cell.somav[:] - cell.somav[0]
    idx = np.argmin(np.abs(v - 0.63*v[-1]))
    print cell.tvec[idx] - cell.tvec[start_t_idx]
    return cell.tvec[idx] - cell.tvec[start_t_idx]

def norm_it(sig):
    return (sig - sig[0])

def plot_all_currents(cell, clamp):

    print "Max imem sum: ", np.max(np.abs(np.sum(cell.imem, axis =0))) 
    print "totnsegs: ", cell.totnsegs
    compartment = 0
    const = 1E-2 * cell.area[compartment]

    imem = cell.imem[compartment,:]
    ipas = cell.ipas[compartment,:]
    icap = cell.icap[compartment,:]
    iclamp = clamp.i
    ica = cell.rec_variables['ica'][compartment,:] * const
    ina = cell.rec_variables['ina'][compartment,:] * const
    ik = cell.rec_variables['ik'][compartment,:] * const
    tvec = cell.tvec[:]
    current_sum = ipas + icap + ica + ina + ik
    #set_trace()
    pl.figure(figsize=[8,10])
    pl.subplots_adjust(hspace=0.5)
    pl.subplot(411)
    pl.title('All currents [nA]')
    pl.plot(tvec, ipas, label='Ipas')
    pl.plot(tvec, iclamp, label='Iclamp')
    #pl.plot(cell.tvec, cell.rec_variables['i_pas'][compartment,:], label='Ipas2')
    pl.plot(tvec, icap, label='Icap')
    pl.plot(tvec, ina, label='ina')
    pl.plot(tvec, ik, label='ik')
    pl.plot(tvec, ica, label='ica')
    pl.legend()
    pl.subplot(412)
    pl.title('All currents shifted')
    pl.plot(tvec, norm_it(ipas), label='Ipas')
    pl.plot(tvec, norm_it(iclamp), label='Iclamp')
    pl.plot(tvec, norm_it(icap), label='Icap')
    pl.plot(tvec, norm_it(ina), label='ina')
    pl.plot(tvec, norm_it(ik), label='ik')
    pl.plot(tvec, norm_it(ica), label='ica')
    #pl.axis([0, 30, -1, 1])
    
    pl.subplot(425)
    pl.plot(tvec, current_sum, label='Sum of soma currents')
    pl.plot(tvec, imem, label='cell.imem[0,:]')
    #pl.axis([0, 30, -0.5, 0.5])
    pl.legend()
    
    pl.subplot(426)
    pl.plot(tvec, current_sum - imem, label='Diff max: %g' % np.max(np.abs(current_sum - imem)))
    #pl.axis([0, 30, -0.5, 0.5])
    pl.legend()
    print "Diffmax: ", np.max(np.abs(current_sum - imem)), "Timestep: ", tvec[1] - tvec[0]

    pl.subplot(414)
    pl.title('Somav [mV]. tau: %g' %tau_tilde)
    pl.plot(tvec, cell.somav)
    
    #pl.title('Sum')
    #axis = pl.axis()
    #pl.axis(axis)
    
    #pl.subplot(143)
    #pl.plot(cell.tvec, cell.imem[compartment,:], label='Imem')
    
    #pl.title('Imem')
    #pl.legend()
    #pl.axis(axis)
    #pl.subplot(144)
    #pl.plot(cell.tvec,  current_sum - cell.imem[compartment,:])
    #pl.title('Diff')
    #pl.legend()
    #pl.axis(axis)
    pl.savefig('mainen_curr_sum_%s.png' % is_active)
    pl.show()


def run_simulation(cell_params, clamp_params):

    neuron.h('forall delete_section()')
    neuron.h('secondorder=2')
    cell = LFPy.Cell(**cell_params)
    synapse_parameters = {
        'idx' : 0,
        'e' : 0.,                   # reversal potential
        'syntype' : 'ExpSyn',       # synapse type
        'tau' : 10.,                # syn. time constant
        'weight' : .001,            # syn. weight
        'record_current' : True,
        }

    # Create synapse and set time of synaptic input
    #synapse = LFPy.Synapse(cell, **synapse_parameters)
    #synapse.set_spike_times(np.array([10.]))
    
    currentClamp = LFPy.StimIntElectrode(cell, **clamp_params)
    cell.simulate(**simulation_params)
    return cell, currentClamp

domain = 'explore_currents'
neuron_model = join('neuron_models', 'mainen')
LFPy.cell.neuron.load_mechanisms(neuron_model)
is_active = True
is_active = bool(int(sys.argv[1]))
input_amp = .0001
input_delay = 100

cell_params = {
    'morphology' : join(neuron_model, 'L5_Mainen96_wAxon_LFPy.hoc'),
    #'morphology' : os.path.join('neuron_models', 'example_morphology.hoc'),
    'rm' : 30000,               # membrane resistance
    'cm' : 1.0,                 # membrane capacitance
    'Ra' : 150,                 # axial resistance
    'v_init' : -65.,             # initial crossmembrane potential
    'e_pas' : -65,              # reversal potential passive mechs
    'passive' : True,           # switch on passive mechs
    'nsegs_method' : 'lambda_f',# method for setting number of segments,
    'lambda_f' : 100,           # segments are isopotential at this frequency
    'timeres_NEURON' : 2**-4,   # dt of LFP and NEURON simulation.
    'timeres_python' : 2**-4,
    'tstartms' : -1000,          #start time, recorders start at t=0
    'tstopms' : 1500,           #stop time of simulation
    'custom_fun'  : [active_declarations], # will execute this function
    'custom_fun_args' : [{'is_active': is_active}],
}

pulse_clamp = {
    'idx' : 0,
    'record_current' : True,
    'amp' : input_amp, #[nA]
    'dur' : 10000.,
    'delay' : input_delay, 
    'pptype' : 'IClamp',
}

simulation_params = {'rec_isyn': True,
                     'rec_imem': True,
                     'rec_ipas': True,
                     'rec_icap': True,
                     'rec_istim': True,
                     'rec_variables' : ['ina', 'ik', 'ica'],
                     }
try:
    os.mkdir(domain)
except OSError:
    pass

cell, clamp = run_simulation(cell_params, pulse_clamp)
tau_tilde = return_time_const(cell)
plot_all_currents(cell, clamp)
