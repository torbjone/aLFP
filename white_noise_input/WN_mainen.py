#!/usr/bin/env python
import os
if not os.environ.has_key('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
    at_stallo = True
else:
    at_stallo = False
    
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
import aLFP
from params import *


model = 'mainen' 
domain = 'white_noise_%s' %model
np.random.seed(1234)

input_idxs = [0, 1071, 610, 984, 846, 604, 302, 240, 422]
input_scalings = [0.0, 0.001, 0.01, 0.1]

simulation_params = {'rec_imem': True,
                     'rec_icap': True,
                     'rec_ipas': True,
                     'rec_variables': ['ina', 'ik', 'ica'],
                     }
plot_params = {'ymax': 1000,
               'ymin': -250,
               }

n_plots = 10
plot_compartments = np.array(np.linspace(0, 1071, n_plots), dtype=int)

def simulate():
    def active_mainen(conductance_type):
        '''set active conductances and correct for spines,
        see file active_declarations_example3.hoc'''
        
        spine_dens = 1
        spine_area = 0.83 # // um^2  -- K Harris

        cm_myelin = 0.04
        g_pas_node = 0.02
        celsius   = 37.

        Ek = -85.
        Ena = 60.

        if conductance_type == 'passive':
            g_adjust = 0
            g_reduced = 0
        elif conductance_type == 'active':
            g_adjust = 1
            g_reduced = 1
        elif conductance_type == 'reduced_with_na':
            g_adjust = 1
            g_reduced = 0
        else: 
            raise RuntimeError, "Wrong conductance type"
        
        gna_dend = 20. * g_adjust 
        gna_node = 30000. * g_adjust 
        gna_soma = gna_dend * 10

        gkv_axon = 2000. * g_adjust
        gkv_soma = 200. * g_adjust

        gca = .3 * g_adjust * g_reduced
        gkm = .1 * g_adjust
        gkca = 3 * g_adjust * g_reduced
        gca_soma = gca * g_reduced
        gkm_soma = gkm 
        gkca_soma = gkca * g_reduced

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

    if at_stallo:
        neuron_model = join('/home', 'torbness', 'work', 'aLFP', 'neuron_models', model)
    else:
        neuron_model = join('..', 'neuron_models', model)
    LFPy.cell.neuron.load_mechanisms(join(neuron_model))      
    LFPy.cell.neuron.load_mechanisms(join(neuron_model, '..'))      
    cut_off = 3000
    conductance_type = 'reduced_with_na'

    rot_params = {'x': -np.pi/2, 
                  'y': 0, 
                  'z': 0
                  }

    pos_params = {'xpos': 0, 
                  'ypos': 0,
                  'zpos': 0,
                  }        

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
        'timeres_NEURON' : timeres,  
        'timeres_python' : timeres,
        'tstartms' : 0,        
        'tstopms' : tstopms + cut_off,           
        'custom_fun'  : [active_mainen], 
        'custom_fun_args' : [{'conductance_type': conductance_type}],
        }
    ntsteps = round((tstopms - 0) / timeres)
    aLFP.initialize_cell(cell_params, pos_params, rot_params, model, elec_x, 
                         elec_y, elec_z, ntsteps, model, testing=False)

    #aLFP.run_simulation(cell_params, input_scalings[0], is_active, input_idxs[0], model)

    cell_params['custom_fun_args'] = [{'conductance_type': 'active'}]  
    aLFP.run_all_simulations(cell_params, model, input_idxs, 
                             input_scalings, ntsteps, simulation_params, 'active')

    cell_params['custom_fun_args'] = [{'conductance_type': 'reduced_with_na'}]  
    aLFP.run_all_simulations(cell_params, model, input_idxs, 
                             input_scalings, ntsteps, simulation_params, 'reduced_with_na')

    cell_params['custom_fun_args'] = [{'conductance_type': 'reduced'}]  
    aLFP.run_all_simulations(cell_params, model, input_idxs, 
                             input_scalings, ntsteps, simulation_params, 'reduced')
    
    #cell_params['custom_fun_args'] = [{'conductance_type': 'passive'}]  
    #aLFP.run_all_simulations(cell_params, model, input_idxs, 
    #                         input_scalings, ntsteps, simulation_params, 'passive')    



def plot_active():

    for input_idx in input_idxs:
        for input_scaling in input_scalings:
            print input_idx, input_scaling
            #aLFP.plot_active_currents(model, input_scaling, input_idx, simulation_params, 'reduced_with_na')
            try:
                aLFP.plot_active_currents(model, input_scaling, input_idx, plot_params, 
                                          simulation_params, plot_compartments, 'active')
            except:
                pass
            try:
                aLFP.plot_active_currents(model, input_scaling, input_idx, plot_params, 
                                          simulation_params, plot_compartments, 'passive')
            except:
                pass
            try:
                aLFP.plot_active_currents(model, input_scaling, input_idx, plot_params, 
                                          simulation_params, plot_compartments, 'reduced')
            except:
                pass            
            try:
                aLFP.plot_active_currents(model, input_scaling, input_idx, plot_params, 
                                          simulation_params, plot_compartments, 'reduced_with_na')
            except:
                pass            
    
def plot_compare():

    #aLFP.compare_active_passive(model, input_scalings[0] , input_idxs[1], 
    #                            elec_x, elec_y, elec_z, plot_params)
    #sys.exit()
    for input_idx in input_idxs:
        for input_scaling in input_scalings:
            print input_idx, input_scaling
            aLFP.compare_active_passive(model, input_scaling , input_idx, 
                                        elec_x, elec_y, elec_z, plot_params)


    
if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python %s <function-name> \nfunction-name can be plot of simulate" 
                         % sys.argv[0])
        raise SystemExit(1)
    func = eval('%s' % sys.argv[1])
    func()
