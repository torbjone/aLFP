#!/usr/bin/env python

import LFPy
import numpy as np
import neuron
import os
import sys
from ipdb import set_trace
import pylab as pl
from os.path import join

import aLFP
from params import *

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

model = 'mainen' 
domain = 'white_noise_%s' %model

np.random.seed(1234)
neuron_model = join('..', 'neuron_models', model)

LFPy.cell.neuron.load_mechanisms(join(neuron_model))      
LFPy.cell.neuron.load_mechanisms(join('..', 'neuron_models'))      
cut_off = 100
is_active = True
input_idxs = [0, 650]

input_scalings = [0.001, 0.01, 0.1]

rot_params = {'x': -np.pi/2, 
              'y': 0, 
              'z': 0
              }

pos_params = {'xpos': 0, 
              'ypos': 0,
              'zpos': 0,
              }        

cell_params = {
    'morphology' : join(model_path, 'L5_Mainen96_wAxon_LFPy.hoc'),
    'rm' : 30000,               # membrane resistance
    'cm' : 1.0,                 # membrane capacitance
    'Ra' : 150,                 # axial resistance
    'v_init' : -65.,             # initial crossmembrane potential
    'e_pas' : -65,              # reversal potential passive mechs
    'passive' : True,           
    'nsegs_method' : 'lambda_f',
    'lambda_f' : 100,           
    'timeres_NEURON' : timeres,  
    'timeres_python' : timeres,
    'tstartms' : 0,        
    'tstopms' : tstopms + cut_off,           
    'custom_fun'  : [active_mainen], 
#'custom_fun_args' : [{'is_active': is_active}],
    }

#aLFP.initialize_cell(cell_params, pos_params, rot_params, model, elec_x, elec_y, elec_z, model)

#aLFP.run_simulation(cell_params, input_scalings[0], is_active, input_idxs[0], model)


# THIS ONE IS NOT TESTED YET


cell_params['custom_fun_args'] = [{'is_active': True}]  
aLFP.run_all_simulations(cell_params, True,  model, input_idxs, input_scalings)

cell_params['custom_fun_args'] = [{'is_active': False}]    
aLFP.run_all_simulations(cell_params, False,  model, input_idxs, input_scalings)
