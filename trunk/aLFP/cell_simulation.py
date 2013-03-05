#!/usr/bin/env python
import os
if not os.environ.has_key('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')

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


pl.rcParams.update({'font.size' : 8,
    'figure.facecolor' : '1',
    'wspace' : 0.5, 'hspace' : 0.5})
np.random.seed(1234)

def active_ball_n_stick(is_active):

    for sec in neuron.h.allsec():
        if is_active:
            sec.insert('hh')
            sec.gnabar_hh = 0.12
            sec.gkbar_hh = 0.036
            sec.gl_hh = 0.0003
            sec.el_hh = -54.3
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 150
        sec.g_pas = 1./ 30000
        sec.e_pas = -65

def active_mainen(is_active):
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


def save_data(cell, electrode, syn, neural_sim_dict, neur_input_params):
    """ Saving all relevant simulation results to disk """
    LFP_psd = []
    ntsteps = neural_sim_dict['ntsteps']
    
    for elec in xrange(len(electrode.x)):
        freqs, power = aLFP.return_psd(electrode.LFP[elec,-ntsteps:], neural_sim_dict)
        LFP_psd.append(power)
    LFP_psd = np.array(LFP_psd)
    #set_trace()
    domain = neural_sim_dict['domain']
    name = "%s_%s_%s" %(neur_input_params['input_scaling'],
                        neur_input_params['input_idx'],
                        neural_sim_dict['is_active'])
    try:
        os.mkdir(join(domain, name))
    except OSError:
        pass

    cell.imem = cell.imem[:,-ntsteps:]
    cell.somav = cell.somav[-ntsteps:]
    cell.noiseVec = cell.noiseVec[-ntsteps:]
    electrode.LFP= electrode.LFP[:,-ntsteps:]
    cell.tvec = cell.tvec[-ntsteps:] - cell.tvec[-ntsteps]
    
    print "Saving simulation data"
    electrode.freqs = freqs
    electrode.LFP_psd = LFP_psd
    np.save(join(domain, name, 'freqs.npy'), freqs)
    np.save(join(domain, name, 'imem.npy'), cell.imem)
    np.save(join(domain, name, 'somav.npy'), cell.somav)
    np.save(join(domain, name, 'input.npy'), cell.noiseVec)

    np.save(join(domain, name, 'xmid.npy'), cell.xmid)
    np.save(join(domain, name, 'ymid.npy'), cell.ymid)
    np.save(join(domain, name, 'zmid.npy'), cell.zmid)

    np.save(join(domain, name, 'elec_x.npy'), electrode.x)
    np.save(join(domain, name, 'elec_y.npy'), electrode.y)
    np.save(join(domain, name, 'elec_z.npy'), electrode.z)
    np.save(join(domain, name, 'lfp.npy'), electrode.LFP)
    np.save(join(domain, name, 'extracellular_psd.npy'), LFP_psd)
    np.save(join(domain, name, 'tvec.npy'), cell.tvec)

def load_data(cell, electrode, syn, neural_sim_dict, neur_input_params):
    """ Loading all relevant simulation results and putting them into their
    respective classes """

    domain = neural_sim_dict['domain']
    name = "%s_%s_%s" %(neur_input_params['input_scaling'],
                        neur_input_params['input_idx'],
                        neural_sim_dict['is_active'])
    print "Loading simulation data"
    cell.imem = np.load(join(domain, name, 'imem.npy'))
    cell.somav = np.load(join(domain, name, 'somav.npy'))
    cell.noiseVec = np.load(join(domain, name, 'input.npy'))
    cell.xmid = np.load(join(domain, name, 'xmid.npy'))
    cell.ymid = np.load(join(domain, name, 'ymid.npy'))
    cell.zmid = np.load(join(domain, name, 'zmid.npy'))

    electrode.x = np.load(join(domain, name, 'elec_x.npy'))
    electrode.y = np.load(join(domain, name, 'elec_y.npy'))
    electrode.z = np.load(join(domain, name, 'elec_z.npy'))
    electrode.LFP = np.load(join(domain, name, 'lfp.npy'))
    electrode.LFP_psd = np.load(join(domain, name, 'extracellular_psd.npy'))
    electrode.freqs = np.load(join(domain, name, 'freqs.npy'))

    cell.tvec = np.load(join(domain, name, 'tvec.npy'))

    return cell, electrode, syn


def run_simulation(neural_sim_dict, neur_input_params):
  
    neuron.h('forall delete_section()')
    neuron.h('secondorder=2')
    model = neural_sim_dict['model']
    model_path = join('neuron_models', model)
    LFPy.cell.neuron.load_mechanisms('neuron_models')
    if model == 'ball_n_stick':
        LFPy.cell.neuron.load_mechanisms(model_path)
        cell_params = {
            'morphology' : join(model_path, 'ball_n_stick.hoc'),
            'passive' : False,           # switch on passive mechs
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,           
            'timeres_NEURON' : neural_sim_dict['timeres'],  
            'timeres_python' :  neural_sim_dict['timeres'],
            'tstartms' : neural_sim_dict['tstartms'],          
            'tstopms' :  neural_sim_dict['tstopms'] + neural_sim_dict['cut_off'],          
            'custom_fun'  : [active_ball_n_stick],
            'custom_fun_args' : [{'is_active':neural_sim_dict['is_active']}],  
            }

        sim_record = {'rec_imem': True,
                      }
        
        pos_dict = {'rot_x': 0,
                    'rot_y': 0,
                    'rot_z': 0,
                    }
    elif model == 'mainen':
        LFPy.cell.neuron.load_mechanisms(mod_path)
        cell_params = {
            'morphology' : join(neuron_model, 'L5_Mainen96_wAxon_LFPy.hoc'),
            'rm' : 30000,               # membrane resistance
            'cm' : 1.0,                 # membrane capacitance
            'Ra' : 150,                 # axial resistance
            'v_init' : -65.,             # initial crossmembrane potential
            'e_pas' : -65,              # reversal potential passive mechs
            'passive' : True,           
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,           
            'timeres_NEURON' : neural_sim_dict['timeres'],  
            'timeres_python' : neural_sim_dict['timeres'],
            'tstartms' : neural_sim_dict['tstartms'],        
            'tstopms' : neural_sim_dict['tstopms'] + neural_sim_dict['cut_off'],           
            'custom_fun'  : [active_mainen], 
            'custom_fun_args' : [{'is_active': is_active}],
            }

        sim_record = {'rec_imem': True,
                      'rec_vmem': True,
                      'rec_istim': True,
                      'rec_synapses': True,
                      'rec_variables' : ['ina', 'ik', 'ica'],
                      }
        pos_dict = {'rot_x': -np.pi/2,
                    'rot_y': 0,
                    'rot_z': 0,
                    }
    elif model == 'hay':
        LFPy.cell.neuron.load_mechanisms(mod_path)
        pos_dict = {'rot_x': -np.pi/2,
                    'rot_y': 0,
                    'rot_z': 0,
                    }

    cell = LFPy.Cell(**cell_params)
    cell.set_rotation(z = pos_dict['rot_z'])
    cell.set_rotation(y = pos_dict['rot_y'])
    cell.set_rotation(x = pos_dict['rot_x'])
    cell.set_pos(xpos = 0, \
                 ypos = 0, \
                 zpos = 0)

    initial_ntsteps = round((neural_sim_dict['tstopms'] + neural_sim_dict['cut_off']) / neural_sim_dict['timeres'] + 1)
    
    input_array = neur_input_params['input_scaling'] * \
                  aLFP.make_WN_input(neural_sim_dict)
                  #np.random.normal(initial_ntsteps)
    noiseVec = neuron.h.Vector(input_array)

    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == neur_input_params['input_idx']:
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if type(syn) == type(None):
        raise RuntimeError("Wrong stimuli index")

    syn.dur = 1E9
    syn.delay = 0
    noiseVec.play(syn._ref_amp, cell.timeres_NEURON)
    cell.noiseVec = np.array(noiseVec)
    # Create a grid of measurement locations, in (mum)

    # Define electrode parameters
    electrode_parameters = {
        'sigma' : 0.3,      # extracellular conductivity
        'x' : neural_sim_dict['elec_x'],  # electrode requires 1d vector of positions
        'y' : neural_sim_dict['elec_y'],
        'z' : neural_sim_dict['elec_z']
        }

    # Create electrode object
    electrode = LFPy.RecExtElectrode(**electrode_parameters)
    if neural_sim_dict['load']:
        load_data(cell, electrode, syn, neural_sim_dict, neur_input_params)
    else:
        cell.simulate(electrode=electrode, **sim_record)
        save_data(cell, electrode, syn, neural_sim_dict, neur_input_params)
    
    return cell, syn, electrode

