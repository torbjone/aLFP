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

pl.rcParams.update({'font.size' : 12,
    'figure.facecolor' : '1',
    'wspace' : 0.5, 'hspace' : 0.5})
np.random.seed(1234)

def return_r_m_tilde(cell):
    r_tilde = (cell.somav[-1] - cell.somav[0])/\
              (cell.imem[0,-1] - cell.imem[0, 0]) * cell.area[0] * 10**-2
    return r_tilde

def interpolate_current(current):
    return current
    #return 0.5*(current[:-1] + current[1:])

def plot_all_currents(cell, syn):

    print "Max imem sum: ", np.max(np.abs(np.sum(cell.imem, axis =0))) 
    print "totnsegs: ", cell.totnsegs
    compartment = 0
    const = 1E-2 * cell.area[compartment]

    imem = interpolate_current(cell.imem[compartment,:])
    ipas = interpolate_current(cell.ipas[compartment,:])
    icap = interpolate_current(cell.icap[compartment,:])
    isyn = interpolate_current(syn.i)
    il_hh = interpolate_current(cell.rec_variables['il_hh'][compartment,:] * const)
    ina = interpolate_current(cell.rec_variables['ina'][compartment,:] * const)
    ik = interpolate_current(cell.rec_variables['ik'][compartment,:] * const)
    tvec = interpolate_current(cell.tvec[:])
     
    current_sum = ipas + icap + isyn + il_hh + ina + ik
    #set_trace()
    pl.figure(figsize=[12,4])
    pl.subplots_adjust(wspace=0.5)
    pl.subplot(311)
    pl.plot(tvec, ipas, label='Ipas')
    pl.plot(tvec, isyn, label='synI')
    #pl.plot(cell.tvec, cell.rec_variables['i_pas'][compartment,:], label='Ipas2')
    pl.plot(tvec, icap, label='Icap')
    pl.plot(tvec, ina, label='ina')
    pl.plot(tvec, ik, label='ik')
    pl.plot(tvec, il_hh, label='il_hh')
    
    #pl.axis([0, 30, -1, 1])
    pl.legend()
    pl.subplot(312)
    pl.plot(tvec, current_sum, label='Sum')
    pl.plot(tvec, imem, label='Imem')
    #pl.axis([0, 30, -0.5, 0.5])
    pl.legend()
    
    pl.subplot(313)
    pl.plot(tvec, current_sum - imem, label='Diff max: %g' % np.max(np.abs(current_sum - imem)))

    #pl.axis([0, 30, -0.5, 0.5])
    pl.legend()

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
    pl.savefig('ball_n_stick_curr_sum.png')
    pl.show()
    
def active_ball_n_stick(is_active):

    for sec in neuron.h.allsec():
        if is_active:
            sec.insert('hh')
            sec.gnabar_hh = 0.12
            sec.gkbar_hh = 0.036
            sec.gl_hh = 0.0003
            sec.el_hh = -54.3
        sec.insert('pas')
        sec.cm = cell_params['cm']
        sec.Ra = cell_params['Ra']
        sec.g_pas = 1./ cell_params['rm']
        sec.e_pas = cell_params['e_pas']


def run_simulation(cell_params, clamp_params):
    neuron.h('forall delete_section()')
    neuron.h('secondorder=2')
    cell = LFPy.Cell(**cell_params)
    synapse_parameters = {
        'idx' : cell.get_closest_idx(x=0., y=0., z=900.),
        'e' : 0.,                   # reversal potential
        'syntype' : 'ExpSyn',       # synapse type
        'tau' : 10.,                # syn. time constant
        'weight' : .001,            # syn. weight
        'record_current' : True,
        }

    # Create synapse and set time of synaptic input
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([10.]))
    
    #currentClamp = LFPy.StimIntElectrode(cell, **clamp_params)
    simulation_params = {'rec_isyn': True,
                         'rec_imem': True,
                         'rec_ipas': True,
                         'rec_icap': True,
                         'rec_istim': True,
                         'rec_variables' : ['ina', 'ik', 'il_hh', 'i_pas'],
                         }
    cell.simulate(**simulation_params)
    return cell, synapse

domain = 'explore_currents'
neuron_model = 'neuron_models'
is_active = True
is_active = bool(int(sys.argv[1]))
input_amp = 0.0
input_delay = 1000

cell_params = {
    'morphology' : join(neuron_model, 'ball_n_stick.hoc'),
    'rm' : 30000,               # membrane resistance
    'cm' : 1.0,                 # membrane capacitance
    'Ra' : 150,                 # axial resistance
    #'v_init' : -77,             # initial crossmembrane potential
    'e_pas' : -65,              # reversal potential passive mechs
    'passive' : False,           # switch on passive mechs
    'nsegs_method' : 'lambda_f',# method for setting number of segments,
    'lambda_f' : 100,           # segments are isopotential at this frequency
    'timeres_NEURON' : 2**-10,   # dt of LFP and NEURON simulation.
    'timeres_python' : 2**-10,
    'tstartms' : 0,          #start time, recorders start at t=0
    'tstopms' : 30,           #stop time of simulation
    'custom_fun'  : [active_ball_n_stick], # will execute this function
    'custom_fun_args' : [{'is_active':is_active}],  
}

pulse_clamp = {
    'idx' : 0,
    'record_current' : True,
    'amp' : input_amp, #[nA]
    'dur' : 10000.,
    'delay' : input_delay, 
    'pptype' : 'IClamp',
}

try:
    os.mkdir(domain)
except OSError:
    pass

cell, syn = run_simulation(cell_params, pulse_clamp)
plot_all_currents(cell, syn)
