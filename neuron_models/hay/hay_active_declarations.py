__author__ = 'torbjone'
#!/usr/bin/env python

import os
from os.path import join
import sys
import numpy as np
import pylab as plt
import neuron
nrn = neuron.h
import LFPy

def make_cell_uniform(Vrest=-60):
    neuron.h.t = 0
    neuron.h.finitialize(Vrest)
    neuron.h.fcurrent()
    for sec in neuron.h.allsec():
        for seg in sec:
            seg.e_pas = seg.v
            if neuron.h.ismembrane("na_ion"):
                seg.e_pas += seg.ina/seg.g_pas
            if neuron.h.ismembrane("k_ion"):
                seg.e_pas += seg.ik/seg.g_pas
            if neuron.h.ismembrane("Ih"):
                seg.e_pas += seg.ihcn_Ih/seg.g_pas
            if neuron.h.ismembrane("Ih_linearized_mod"):
                seg.e_pas = seg.e_pas + seg.ihcn_Ih_linearized_mod/seg.g_pas
                # seg.e_pas += (seg.gIhbar_Ih_linearized_mod * seg.mInf_Ih_linearized_mod *
                #               (Vrest-seg.ehcn_Ih_linearized_mod)/seg.g_pas)
            if neuron.h.ismembrane("ca_ion"):
                seg.e_pas = seg.e_pas + seg.ica/seg.g_pas


def biophys_passive(**kwargs):
    for sec in neuron.h.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = -90

    for sec in neuron.h.soma:
        sec.g_pas = 0.0000338

    for sec in neuron.h.apic:
        sec.cm = 2
        sec.g_pas = 0.0000589

    for sec in neuron.h.dend:
        sec.cm = 2
        sec.g_pas = 0.0000467

    for sec in neuron.h.axon:
        sec.g_pas = 0.0000325

    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])

    print("Passive dynamics inserted.")


def biophys_Ih_linearized(**kwargs):

    Vrest = kwargs['hold_potential'] if 'hold_potential' in kwargs else -70

    for sec in neuron.h.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = Vrest

    for sec in neuron.h.soma:
        sec.insert("Ih_linearized_mod")
        sec.gIhbar_Ih_linearized_mod = 0.0002
        sec.g_pas = 0.0000338
        sec.vss_Ih_linearized_mod = Vrest

    for sec in neuron.h.apic:
        sec.insert("Ih_linearized_mod")
        sec.cm = 2
        sec.g_pas = 0.0000589
        sec.vss_Ih_linearized_mod = Vrest

    nrn.distribute_channels("apic", "gIhbar_Ih_linearized_mod",
                                 2, -0.8696, 3.6161, 0.0, 2.0870, 0.00020000000)

    for sec in neuron.h.dend:
        sec.insert("Ih_linearized_mod")
        sec.cm = 2
        sec.g_pas = 0.0000467
        sec.gIhbar_Ih_linearized_mod = 0.0002
        sec.vss_Ih_linearized_mod = Vrest

    for sec in neuron.h.axon:
        sec.g_pas = 0.0000325

    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])

    print("Linearized Ih inserted.")


def biophys_active(**kwargs):

    for sec in neuron.h.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = -90.

    for sec in neuron.h.soma:
        sec.insert('Ca_LVAst')
        sec.insert('Ca_HVA')
        sec.insert('SKv3_1')
        sec.insert('SK_E2')
        sec.insert('K_Tst')
        sec.insert('K_Pst')
        sec.insert('Nap_Et2')
        sec.insert('NaTa_t')
        sec.insert('CaDynamics_E2')
        sec.insert('Ih')
        sec.ek = -85
        sec.ena = 50
        sec.gIhbar_Ih = 0.0002
        sec.g_pas = 0.0000338
        sec.decay_CaDynamics_E2 = 460.0
        sec.gamma_CaDynamics_E2 = 0.000501
        sec.gCa_LVAstbar_Ca_LVAst = 0.00343
        sec.gCa_HVAbar_Ca_HVA = 0.000992
        sec.gSKv3_1bar_SKv3_1 = 0.693
        sec.gSK_E2bar_SK_E2 = 0.0441
        sec.gK_Tstbar_K_Tst = 0.0812
        sec.gK_Pstbar_K_Pst = 0.00223
        sec.gNap_Et2bar_Nap_Et2 = 0.00172
        sec.gNaTa_tbar_NaTa_t = 2.04

    for sec in neuron.h.apic:
        sec.cm = 2
        sec.insert('Ih')
        sec.insert('SK_E2')
        sec.insert('Ca_LVAst')
        sec.insert('Ca_HVA')
        sec.insert('SKv3_1')
        sec.insert('NaTa_t')
        sec.insert('Im')
        sec.insert('CaDynamics_E2')
        sec.ek = -85
        sec.ena = 50
        sec.decay_CaDynamics_E2 = 122
        sec.gamma_CaDynamics_E2 = 0.000509
        sec.gSK_E2bar_SK_E2 = 0.0012
        sec.gSKv3_1bar_SKv3_1 = 0.000261
        sec.gNaTa_tbar_NaTa_t = 0.0213
        sec.gImbar_Im = 0.0000675
        sec.g_pas = 0.0000589

    nrn.distribute_channels("apic", "gIhbar_Ih", 2, -0.8696, 3.6161, 0.0, 2.0870, 0.00020000000)
    nrn.distribute_channels("apic", "gCa_LVAstbar_Ca_LVAst", 3, 1.000000, 0.010000, 685.000000, 885.000000, 0.0187000000)
    nrn.distribute_channels("apic", "gCa_HVAbar_Ca_HVA", 3, 1.000000, 0.100000, 685.000000, 885.000000, 0.0005550000)

    for sec in neuron.h.dend:
        sec.cm = 2
        sec.insert('Ih')
        sec.gIhbar_Ih = 0.0002
        sec.g_pas = 0.0000467

    for sec in neuron.h.axon:
        sec.g_pas = 0.0000325

    #for sec in neuron.h.allsec():
    #    if neuron.h.ismembrane('k_ion'):
    #        sec.ek = Ek

    #neuron.h.celsius = celsius

    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])

    print("active ion-channels inserted.")


def active_declarations(**kwargs):
    ''' set active conductances for Hay model 2011 '''

    nrn.delete_axon()
    nrn.geom_nseg()
    nrn.define_shape()

    exec('biophys_%s(**kwargs)' % kwargs['conductance_type'])


if __name__ == '__main__':

    timeres = 2**-4
    cut_off = 0
    tstopms = 100
    tstartms = -cut_off
    model_path = join('lfpy_version')


    cell_params = {
        'morphology': join(model_path, 'morphologies', 'cell1.hoc'),
        #'rm' : 30000,               # membrane resistance
        #'cm' : 1.0,                 # membrane capacitance
        #'Ra' : 100,                 # axial resistance
        'v_init': -80,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': 1,
        'tstartms': tstartms,          # start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_code': [join(model_path, 'custom_codes.hoc')],
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'conductance_type': 'passive',
                             'hold_potential': -80}],
    }

    cell = LFPy.Cell(**cell_params)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plt.plot(cell.tvec, cell.somav)
    plt.show()