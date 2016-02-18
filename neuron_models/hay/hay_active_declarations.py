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

def make_cell_uniform(Vrest=-80):

    # for sec in nrn.allsec():
    #     for seg in sec:
    #         if nrn.ismembrane("ca_ion"):
    #             print "Cai: ", sec.name(), seg.cai, seg.ica
    nrn.t = 0
    nrn.finitialize(Vrest)
    nrn.fcurrent()
    for sec in nrn.allsec():
        for seg in sec:
            seg.e_pas = seg.v
            if nrn.ismembrane("na_ion"):
                seg.e_pas += seg.ina/seg.g_pas
            if nrn.ismembrane("k_ion"):
                seg.e_pas += seg.ik/seg.g_pas
            if nrn.ismembrane("ca_ion"):
                seg.e_pas = seg.e_pas + seg.ica/seg.g_pas
            if nrn.ismembrane("Ih"):
                seg.e_pas += seg.ihcn_Ih/seg.g_pas
            if nrn.ismembrane("Ih_z"):
                seg.e_pas += seg.ih_Ih_z/seg.g_pas
            if nrn.ismembrane("Ih_frozen"):
                seg.e_pas += seg.ihcn_Ih_frozen/seg.g_pas
            if nrn.ismembrane("Ih_linearized_v2"):
                seg.e_pas += seg.ihcn_Ih_linearized_v2/seg.g_pas
            if nrn.ismembrane("Ih_linearized_v2_frozen"):
                seg.e_pas += seg.ihcn_Ih_linearized_v2_frozen/seg.g_pas
    # print
    # e = []
    # for sec in nrn.allsec():
    #     for seg in sec:
    #         e.append(seg.e_pas)
            # if nrn.ismembrane("ca_ion"):
            #     print "Cai: ", sec.name(), seg.cai, seg.ica
    # plt.plot(e)
    # plt.ylim([0, -100])
    # plt.xlim([-10, 700])
    # plt.show()

def _get_longest_distance():

    nrn.distance(0, 0.5)
    max_dist = 0
    for sec in nrn.allsec():
        for seg in sec:
            max_dist = np.max([max_dist, nrn.distance(seg.x)])
    return max_dist

def _get_total_area():

    total_area = 0
    for sec in nrn.allsec():
        for seg in sec:
           total_area += nrn.area(seg.x)
    return total_area

def _get_linear_increase_factor(increase_factor, max_dist, total_conductance):
    normalization = 0
    for sec in nrn.allsec():
        for seg in sec:
            normalization += nrn.area(seg.x) * (1 + (increase_factor - 1) * nrn.distance(seg.x)/max_dist)
    return total_conductance / normalization

def _get_linear_decrease_factor(decrease_factor, max_dist, total_conductance):
    normalization = 0
    for sec in nrn.allsec():
        for seg in sec:
            normalization += nrn.area(seg.x) * (decrease_factor - (decrease_factor - 1) * nrn.distance(seg.x)/max_dist)
    return total_conductance / normalization

def biophys_zuchkova(**kwargs):
    nrn.distance(0, 0.5)
    for sec in nrn.allsec():
        sec.insert('pas')
        sec.e_pas = kwargs['hold_potential']
        sec.Ra = 200
        sec.cm = 1.0
        sec.g_pas = 0.09e-3
        sec.insert("Ih_z")
        for seg in sec:
            seg.gIhbar_Ih_z = 26.71e-6 * np.exp(0.0041*nrn.distance(seg.x))

    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])

def biophys_generic(**kwargs):

    # v = kwargs['hold_potential']
    if 'auto' in kwargs['tau_w']:
        # mAlpha = 0.001 * 6.43 * (v + 154.9)/(np.exp((v + 154.9) / 11.9) - 1.)
        # mBeta = 0.001 * 193. * np.exp(v / 33.1)
        tau = 50 #1/(mAlpha + mBeta)
        if '0.1' in kwargs['tau_w']:
            print "1/10th of auto"
            tau_w = tau * 0.1
        elif '10' in kwargs['tau_w']:
            print "10-fold of auto"
            tau_w = tau * 10
        else:
            tau_w = tau
        #print "Ih calculated tau_w: ", tau_w
    else:
        tau_w = kwargs['tau_w']

    for sec in nrn.allsec():
        sec.insert("QA")
        sec.V_r_QA = kwargs['hold_potential']
        sec.tau_w_QA = tau_w
        sec.Ra = 100
        sec.cm = 1.0
        sec.g_pas_QA = kwargs['g_pas']

    total_area = _get_total_area()
    total_w_conductance = kwargs['avrg_w_bar'] * total_area
    max_dist = _get_longest_distance()
    # print "Max dist: ", max_dist

    if kwargs['distribution'] == 'uniform':
        for sec in nrn.allsec():
            for seg in sec:
                seg.g_w_bar_QA = total_w_conductance / total_area
    elif kwargs['distribution'] == 'linear_increase':
        increase_factor = 60
        conductance_factor = _get_linear_increase_factor(increase_factor, max_dist, total_w_conductance)
        nrn.distance(0, 0.5)
        for sec in nrn.allsec():
            for seg in sec:
                seg.g_w_bar_QA = conductance_factor * (1 + (increase_factor - 1.) * nrn.distance(seg.x) / max_dist)
        # print 'Linear increase:  %1.8f + %1.10f * x' % (conductance_factor, conductance_factor*(increase_factor - 1.) / max_dist)

    elif kwargs['distribution'] == 'linear_decrease':
        decrease_factor = 60
        conductance_factor = _get_linear_decrease_factor(decrease_factor, max_dist, total_w_conductance)
        nrn.distance(0, 0.5)
        for sec in nrn.allsec():
            for seg in sec:
                seg.g_w_bar_QA = conductance_factor*(decrease_factor - (decrease_factor - 1.) * nrn.distance(seg.x) / max_dist)
        # print 'Linear decrease: %1.8f - %1.10f * x' % (conductance_factor * decrease_factor, conductance_factor * (decrease_factor - 1.) / max_dist)
    else:
        raise RuntimeError("Unknown distribution...")
    cond_check = 0
    for sec in nrn.allsec():
        for seg in sec:
            cond_check += seg.g_w_bar_QA * nrn.area(seg.x)
            seg.mu_QA = seg.g_w_bar_QA / seg.g_pas_QA * kwargs['mu_factor']

    if np.abs(cond_check - total_w_conductance) < 1e-6:
        pass
        #print "Works", cond_check, total_w_conductance, cond_check - total_w_conductance
    else:
        raise RuntimeError("Total conductance not as expected")

def biophys_passive(**kwargs):
    Vrest = kwargs['hold_potential'] if 'hold_potential' in kwargs else -70
    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = Vrest

    for sec in nrn.soma:
        sec.g_pas = 0.0000338

    for sec in nrn.apic:
        sec.cm = 2
        sec.g_pas = 0.0000589

    for sec in nrn.dend:
        sec.cm = 2
        sec.g_pas = 0.0000467

    for sec in nrn.axon:
        sec.g_pas = 0.0000325

    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])

    #print("Passive dynamics inserted.")

def biophys_Ih_linearized_frozen(**kwargs):

    Vrest = kwargs['hold_potential'] if 'hold_potential' in kwargs else -70

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = Vrest
    for sec in nrn.soma:
        sec.insert("Ih_linearized_v2_frozen")
        sec.gIhbar_Ih_linearized_v2_frozen = 0.0002
        sec.g_pas = 0.0000338
        sec.V_R_Ih_linearized_v2_frozen = Vrest
    for sec in nrn.apic:
        sec.insert("Ih_linearized_v2_frozen")
        sec.cm = 2
        sec.g_pas = 0.0000589
        sec.V_R_Ih_linearized_v2_frozen = Vrest

    nrn.distribute_channels("apic", "gIhbar_Ih_linearized_v2_frozen",
                            2, -0.8696, 3.6161, 0.0, 2.0870, 0.0002)
    for sec in nrn.dend:
        sec.insert("Ih_linearized_v2_frozen")
        sec.cm = 2
        sec.g_pas = 0.0000467
        sec.gIhbar_Ih_linearized_v2_frozen = 0.0002
        sec.V_R_Ih_linearized_v2_frozen = Vrest
    for sec in nrn.axon:
        sec.g_pas = 0.0000325
    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])

    print("Frozen linearized Ih inserted.")

def biophys_Ih_linearized(**kwargs):

    Vrest = kwargs['hold_potential'] if 'hold_potential' in kwargs else -70

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = Vrest
    for sec in nrn.soma:
        sec.insert("Ih_linearized_v2")
        sec.gIhbar_Ih_linearized_v2 = 0.0002
        sec.g_pas = 0.0000338
        # sec.vss_Ih_linearized_mod = Vrest
        sec.V_R_Ih_linearized_v2 = Vrest
    for sec in nrn.apic:
        sec.insert("Ih_linearized_v2")
        sec.cm = 2
        sec.g_pas = 0.0000589
        # sec.vss_Ih_linearized_mod = Vrest
        sec.V_R_Ih_linearized_v2 = Vrest

    nrn.distribute_channels("apic", "gIhbar_Ih_linearized_v2",
                            2, -0.8696, 3.6161, 0.0, 2.0870, 0.00020000000)
    for sec in nrn.dend:
        sec.insert("Ih_linearized_v2")
        sec.cm = 2
        sec.g_pas = 0.0000467
        sec.gIhbar_Ih_linearized_v2 = 0.0002
        # sec.vss_Ih_linearized_v2 = Vrest
        sec.V_R_Ih_linearized_v2 = Vrest
    for sec in nrn.axon:
        sec.g_pas = 0.0000325
    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])

    print("Linearized Ih inserted.")

def biophys_Ih(**kwargs):

    Vrest = kwargs['hold_potential'] if 'hold_potential' in kwargs else -70

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = Vrest
    for sec in nrn.soma:
        sec.insert("Ih")
        sec.gIhbar_Ih = 0.0002
        sec.g_pas = 0.0000338
    for sec in nrn.apic:
        sec.insert("Ih")
        sec.cm = 2
        sec.g_pas = 0.0000589

    nrn.distribute_channels("apic", "gIhbar_Ih",
                            2, -0.8696, 3.6161, 0.0, 2.0870, 0.00020000000)
    for sec in nrn.dend:
        sec.insert("Ih")
        sec.cm = 2
        sec.g_pas = 0.0000467
        sec.gIhbar_Ih = 0.0002
    for sec in nrn.axon:
        sec.g_pas = 0.0000325
    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])

    print("Single Ih inserted.")

def biophys_Ih_frozen(**kwargs):

    Vrest = kwargs['hold_potential'] if 'hold_potential' in kwargs else -70

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = Vrest
    for sec in nrn.soma:
        sec.insert("Ih_frozen")
        sec.gIhbar_Ih_frozen = 0.0002
        sec.g_pas = 0.0000338
    for sec in nrn.apic:
        sec.insert("Ih_frozen")
        sec.cm = 2
        sec.g_pas = 0.0000589

    nrn.distribute_channels("apic", "gIhbar_Ih_frozen",
                            2, -0.8696, 3.6161, 0.0, 2.0870, 0.00020000000)
    for sec in nrn.dend:
        sec.insert("Ih_frozen")
        sec.cm = 2
        sec.g_pas = 0.0000467
        sec.gIhbar_Ih_frozen = 0.0002
    for sec in nrn.axon:
        sec.g_pas = 0.0000325
    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])

    print("Single frozen Ih inserted.")

def biophys_SKv3_1(**kwargs):

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = -90.

    for sec in nrn.soma:
        sec.insert('SKv3_1')
        sec.ek = -85
        sec.g_pas = 0.0000338
        sec.gSKv3_1bar_SKv3_1 = 0.693

    for sec in nrn.apic:
        sec.cm = 2
        sec.insert('SKv3_1')
        sec.ek = -85
        sec.gSKv3_1bar_SKv3_1 = 0.000261
        sec.g_pas = 0.0000589

    for sec in nrn.dend:
        sec.cm = 2
        sec.g_pas = 0.0000467

    for sec in nrn.axon:
        sec.g_pas = 0.0000325

    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])
    print("SKv3_1 ion-channels inserted.")

def biophys_SKv3_1_Ih(**kwargs):

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = -90.

    for sec in nrn.soma:
        sec.insert('SKv3_1')
        sec.insert('Ih')
        sec.ek = -85
        sec.gIhbar_Ih = 0.0002
        sec.g_pas = 0.0000338
        sec.gSKv3_1bar_SKv3_1 = 0.693

    for sec in nrn.apic:
        sec.cm = 2
        sec.insert('Ih')
        sec.insert('SKv3_1')
        sec.ek = -85
        sec.gSKv3_1bar_SKv3_1 = 0.000261
        sec.g_pas = 0.0000589
    nrn.distribute_channels("apic", "gIhbar_Ih", 2, -0.8696, 3.6161, 0.0, 2.087, 0.0002)

    for sec in nrn.dend:
        sec.cm = 2
        sec.insert('Ih')
        sec.gIhbar_Ih = 0.0002
        sec.g_pas = 0.0000467

    for sec in nrn.axon:
        sec.g_pas = 0.0000325

    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])
    print("Ih and SKv3_1 ion-channels inserted.")

def biophys_active(**kwargs):

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = -90.

    for sec in nrn.soma:
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

    for sec in nrn.apic:
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

    nrn.distribute_channels("apic", "gIhbar_Ih", 2, -0.8696, 3.6161, 0.0, 2.087, 0.0002)
    nrn.distribute_channels("apic", "gCa_LVAstbar_Ca_LVAst", 3, 1.0, 0.010, 685.0, 885.0, 0.0187)
    nrn.distribute_channels("apic", "gCa_HVAbar_Ca_HVA", 3, 1.0, 0.10, 685.00, 885.0, 0.000555)

    for sec in nrn.dend:
        sec.cm = 2
        sec.insert('Ih')
        sec.gIhbar_Ih = 0.0002
        sec.g_pas = 0.0000467

    for sec in nrn.axon:
        sec.g_pas = 0.0000325

    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])
    #print("active ion-channels inserted.")

def biophys_reduced(**kwargs):

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = -90.

    for sec in nrn.soma:
        sec.insert('Ca_LVAst')
        # sec.insert('Ca_HVA')
        # sec.insert('SKv3_1')
        # sec.insert('SK_E2')
        # sec.insert('K_Tst')
        # sec.insert('K_Pst')
        # sec.insert('Nap_Et2')
        # sec.insert('NaTa_t')
        # sec.insert('CaDynamics_E2')
        # sec.insert('Ih')
        # sec.ek = -85
        # sec.ena = 50
        # sec.gIhbar_Ih = 0.0002
        sec.g_pas = 0.0000338
        # sec.decay_CaDynamics_E2 = 460.0
        # sec.gamma_CaDynamics_E2 = 0.000501
        sec.gCa_LVAstbar_Ca_LVAst = 0.00343
        # sec.gCa_HVAbar_Ca_HVA = 0.000992
        # sec.gSKv3_1bar_SKv3_1 = 0.693
        # sec.gSK_E2bar_SK_E2 = 0.0441
        # sec.gK_Tstbar_K_Tst = 0.0812
        # sec.gK_Pstbar_K_Pst = 0.00223
        # sec.gNap_Et2bar_Nap_Et2 = 0.00172
        # sec.gNaTa_tbar_NaTa_t = 2.04

    for sec in nrn.apic:
        sec.cm = 2
        # sec.insert('Ih')
        # sec.insert('SK_E2')
        sec.insert('Ca_LVAst')
        print sec.eca
        # sec.insert('Ca_HVA')
        # sec.insert('SKv3_1')
        # sec.insert('NaTa_t')
        # sec.insert('Im')
        # sec.insert('CaDynamics_E2')
        # sec.ek = -85
        # sec.ena = 50
        # sec.decay_CaDynamics_E2 = 122
        # sec.gamma_CaDynamics_E2 = 0.000509
        # sec.gSK_E2bar_SK_E2 = 0.0012
        # sec.gSKv3_1bar_SKv3_1 = 0.000261
        # sec.gNaTa_tbar_NaTa_t = 0.0213
        # sec.gImbar_Im = 0.0000675
        sec.g_pas = 0.0000589

    # nrn.distribute_channels("apic", "gIhbar_Ih", 2, -0.8696, 3.6161, 0.0, 2.087, 0.0002)
    nrn.distribute_channels("apic", "gCa_LVAstbar_Ca_LVAst", 3, 1.0, 0.010, 685.0, 885.0, 0.0187)
    # nrn.distribute_channels("apic", "gCa_HVAbar_Ca_HVA", 3, 1.0, 0.10, 685.00, 885.0, 0.000555)

    for sec in nrn.dend:
        sec.cm = 2
        # sec.insert('Ih')
        # sec.gIhbar_Ih = 0.0002
        sec.g_pas = 0.0000467

    for sec in nrn.axon:
        sec.g_pas = 0.0000325

    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])
    print("reduced ion-channels inserted.")

def biophys_regenerative(**kwargs):

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = -90.

    for sec in nrn.soma:
        # sec.insert('Ca_LVAst')
        # sec.insert('Ca_HVA')
        sec.insert('SKv3_1')
        # sec.insert('SK_E2')
        # sec.insert('K_Tst')
        # sec.insert('K_Pst')
        sec.insert('Nap_Et2')
        sec.insert('NaTa_t')
        # sec.insert('CaDynamics_E2')
        # sec.insert('Ih')
        # sec.ek = -85
        sec.ena = 50
        # sec.gIhbar_Ih = 0.0002
        sec.g_pas = 0.0000338
        # sec.decay_CaDynamics_E2 = 460.0
        # sec.gamma_CaDynamics_E2 = 0.000501
        # sec.gCa_LVAstbar_Ca_LVAst = 0.00343
        # sec.gCa_HVAbar_Ca_HVA = 0.000992
        sec.gSKv3_1bar_SKv3_1 = 0.693
        # sec.gSK_E2bar_SK_E2 = 0.0441
        # sec.gK_Tstbar_K_Tst = 0.0812
        # sec.gK_Pstbar_K_Pst = 0.00223
        sec.gNap_Et2bar_Nap_Et2 = 0.00172
        sec.gNaTa_tbar_NaTa_t = 2.04

    for sec in nrn.apic:
        sec.cm = 2
        # sec.insert('Ih')
        # sec.insert('SK_E2')
        # sec.insert('Ca_LVAst')
        # print sec.eca
        # sec.insert('Ca_HVA')
        # sec.insert('SKv3_1')
        # sec.insert('NaTa_t')
        # sec.insert('Im')
        # sec.insert('CaDynamics_E2')
        # sec.ek = -85
        # sec.ena = 50
        # sec.decay_CaDynamics_E2 = 122
        # sec.gamma_CaDynamics_E2 = 0.000509
        # sec.gSK_E2bar_SK_E2 = 0.0012
        # sec.gSKv3_1bar_SKv3_1 = 0.000261
        # sec.gNaTa_tbar_NaTa_t = 0.0213
        # sec.gImbar_Im = 0.0000675
        sec.g_pas = 0.0000589

    # nrn.distribute_channels("apic", "gIhbar_Ih", 2, -0.8696, 3.6161, 0.0, 2.087, 0.0002)
    # nrn.distribute_channels("apic", "gCa_LVAstbar_Ca_LVAst", 3, 1.0, 0.010, 685.0, 885.0, 0.0187)
    # nrn.distribute_channels("apic", "gCa_HVAbar_Ca_HVA", 3, 1.0, 0.10, 685.00, 885.0, 0.000555)

    for sec in nrn.dend:
        sec.cm = 2
        # sec.insert('Ih')
        # sec.gIhbar_Ih = 0.0002
        sec.g_pas = 0.0000467

    for sec in nrn.axon:
        sec.g_pas = 0.0000325

    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])
    print("Regenerative ion-channels inserted.")

def biophys_K(**kwargs):

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = -90.

    for sec in nrn.soma:
        sec.insert('SKv3_1')
        # sec.insert('SK_E2')
        # sec.insert('K_Tst')
        # sec.insert('K_Pst')
        sec.ek = -85
        sec.g_pas = 0.0000338
        sec.gSKv3_1bar_SKv3_1 = 0.693
        # sec.gSK_E2bar_SK_E2 = 0.0441
        # sec.gK_Tstbar_K_Tst = 0.0812
        # sec.gK_Pstbar_K_Pst = 0.00223

    for sec in nrn.apic:
        sec.cm = 2
        # sec.insert('SK_E2')
        sec.insert('SKv3_1')
        sec.ek = -85
        # sec.gSK_E2bar_SK_E2 = 0.0012
        sec.gSKv3_1bar_SKv3_1 = 0.000261
        sec.g_pas = 0.0000589

    for sec in nrn.dend:
        sec.cm = 2
        sec.g_pas = 0.0000467

    for sec in nrn.axon:
        sec.g_pas = 0.0000325

    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])
    print("K ion-channels inserted.")

def biophys_NaP(**kwargs):

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = -90.

    for sec in nrn.soma:
        sec.insert('Nap_Et2')
        sec.ena = 50
        sec.g_pas = 0.0000338
        sec.gNap_Et2bar_Nap_Et2 = 0.00172

    for sec in nrn.apic:
        sec.cm = 2
        sec.g_pas = 0.0000589

    for sec in nrn.dend:
        sec.cm = 2
        sec.g_pas = 0.0000467

    for sec in nrn.axon:
        sec.g_pas = 0.0000325

    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])
    print("Persistent Na ion-channels inserted.")

def biophys_NaP_linearized(**kwargs):

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = -90.

    for sec in nrn.soma:
        sec.insert('Nap_Et2_linearized')
        sec.ena = 50
        sec.g_pas = 0.0000338
        sec.gNap_Et2bar_Nap_Et2_linearized = 0.00172
        sec.V_R_Nap_Et2_linearized = kwargs['hold_potential']
    for sec in nrn.apic:
        sec.cm = 2
        sec.g_pas = 0.0000589

    for sec in nrn.dend:
        sec.cm = 2
        sec.g_pas = 0.0000467

    for sec in nrn.axon:
        sec.g_pas = 0.0000325

    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])
    print("Linearized persistent Na ion-channels inserted.")

def biophys_NaP_frozen(**kwargs):

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = -90.
    for sec in nrn.soma:
        sec.insert('Nap_Et2_frozen')
        sec.ena = 50
        sec.g_pas = 0.0000338
        sec.gNap_Et2bar_Nap_Et2_frozen = 0.00172
    for sec in nrn.apic:
        sec.cm = 2
        sec.g_pas = 0.0000589
    for sec in nrn.dend:
        sec.cm = 2
        sec.g_pas = 0.0000467
    for sec in nrn.axon:
        sec.g_pas = 0.0000325
    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])
    print("Frozen persistent Na ion-channels inserted.")

def biophys_active_frozen(**kwargs):

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.cm = 1.0
        sec.Ra = 100.
        sec.e_pas = -90.

    for sec in nrn.soma:
        sec.insert('Ca_LVAst_frozen')
        sec.insert('Ca_HVA_frozen')
        sec.insert('SKv3_1_frozen')
        sec.insert('SK_E2_frozen')
        sec.insert('K_Tst_frozen')
        sec.insert('K_Pst_frozen')
        sec.insert('Nap_Et2_frozen')
        sec.insert('NaTa_t_frozen')
        sec.insert('CaDynamics_E2')
        sec.insert('Ih_frozen')
        sec.ek = -85
        sec.ena = 50
        sec.gIhbar_Ih_frozen = 0.0002
        sec.g_pas = 0.0000338
        sec.decay_CaDynamics_E2 = 460.0
        sec.gamma_CaDynamics_E2 = 0.000501
        sec.gCa_LVAstbar_Ca_LVAst_frozen = 0.00343
        sec.gCa_HVAbar_Ca_HVA_frozen = 0.000992
        sec.gSKv3_1bar_SKv3_1_frozen = 0.693
        sec.gSK_E2bar_SK_E2_frozen = 0.0441
        sec.gK_Tstbar_K_Tst_frozen = 0.0812
        sec.gK_Pstbar_K_Pst_frozen = 0.00223
        sec.gNap_Et2bar_Nap_Et2_frozen = 0.00172
        sec.gNaTa_tbar_NaTa_t_frozen = 2.04

    for sec in nrn.apic:
        sec.cm = 2
        sec.insert('Ih_frozen')
        sec.insert('SK_E2_frozen')
        sec.insert('Ca_LVAst_frozen')
        sec.insert('Ca_HVA_frozen')
        sec.insert('SKv3_1_frozen')
        sec.insert('NaTa_t_frozen')
        sec.insert('Im_frozen')
        sec.insert('CaDynamics_E2')
        sec.ek = -85
        sec.ena = 50
        sec.decay_CaDynamics_E2 = 122
        sec.gamma_CaDynamics_E2 = 0.000509
        sec.gSK_E2bar_SK_E2_frozen = 0.0012
        sec.gSKv3_1bar_SKv3_1_frozen = 0.000261
        sec.gNaTa_tbar_NaTa_t_frozen = 0.0213
        sec.gImbar_Im_frozen = 0.0000675
        sec.g_pas = 0.0000589

    nrn.distribute_channels("apic", "gIhbar_Ih_frozen", 2, -0.8696, 3.6161, 0.0, 2.0870, 0.00020000000)
    nrn.distribute_channels("apic", "gCa_LVAstbar_Ca_LVAst_frozen", 3, 1.000000, 0.010000, 685.000000, 885.000000, 0.0187000000)
    nrn.distribute_channels("apic", "gCa_HVAbar_Ca_HVA_frozen", 3, 1.000000, 0.100000, 685.000000, 885.000000, 0.0005550000)

    for sec in nrn.dend:
        sec.cm = 2
        sec.insert('Ih_frozen')
        sec.gIhbar_Ih_frozen = 0.0002
        sec.g_pas = 0.0000467

    for sec in nrn.axon:
        sec.g_pas = 0.0000325

    if 'hold_potential' in kwargs:
        make_cell_uniform(Vrest=kwargs['hold_potential'])
    print("Frozen active ion-channels inserted.")

def make_syaptic_stimuli(cell, input_idx):
    # Define synapse parameters
    synapse_parameters = {
        'idx': input_idx,
        'e': 0.,                   # reversal potential
        'syntype': 'ExpSyn',       # synapse type
        'tau': 10.,                # syn. time constant
        'weight': 0.00145,            # syn. weight
        'record_current': True,
    }
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([5.]))
    return cell, synapse

def active_declarations(**kwargs):
    ''' set active conductances for Hay model 2011 '''
    nrn.delete_axon()
    nrn.geom_nseg()
    nrn.define_shape()
    exec('biophys_%s(**kwargs)' % kwargs['conductance_type'])

def simulate_synaptic_input(input_idx, holding_potential, conductance_type):

    timeres = 2**-4
    cut_off = 0
    tstopms = 150
    tstartms = -cut_off
    model_path = join('lfpy_version')
    neuron.load_mechanisms('mod')
    neuron.load_mechanisms('..')
    cell_params = {
        'morphology': join(model_path, 'morphologies', 'cell1.hoc'),
        #'rm' : 30000,               # membrane resistance
        #'cm' : 1.0,                 # membrane capacitance
        #'Ra' : 100,                 # axial resistance
        'v_init': holding_potential,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': 1,
        'tstartms': tstartms,          # start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_code': [join(model_path, 'custom_codes.hoc')],
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'conductance_type': conductance_type,
                             'hold_potential': holding_potential}],
    }

    cell = LFPy.Cell(**cell_params)
    make_syaptic_stimuli(cell, input_idx)
    cell.simulate(rec_vmem=True, rec_imem=True)

    plt.subplot(211, title='Soma', ylim=[holding_potential-1.5, holding_potential+2.5])
    plt.plot(cell.tvec, cell.vmem[0, :], label='%d %s %d mV' % (input_idx, conductance_type,
                                                                holding_potential))

    plt.subplot(212, title='Shifted', ylim=[-0.5, 3])
    plt.plot(cell.tvec, cell.vmem[input_idx, :] - cell.vmem[input_idx, 0])

def test_frozen_currents(input_idx, holding_potential):

    # input_idx = 0
    # holding_potential = -70
    plt.close('all')
    simulate_synaptic_input(input_idx, holding_potential, 'active_frozen')
    simulate_synaptic_input(input_idx, holding_potential, 'active')
    simulate_synaptic_input(input_idx, holding_potential, 'passive')
    simulate_synaptic_input(input_idx, holding_potential, 'Ih_linearized')
    simulate_synaptic_input(input_idx, holding_potential, 'Ih_linearized_frozen')

    plt.legend(frameon=False)
    plt.savefig('frozen_test_%d_%d.png' % (input_idx, holding_potential))

def test_steady_state():

    timeres = 2**-4
    cut_off = 6000
    tstopms = 1000
    tstartms = -cut_off
    model_path = join('lfpy_version')
    neuron.load_mechanisms('mod')
    neuron.load_mechanisms('..')


    cell_params = {
        'morphology': join(model_path, 'morphologies', 'cell1.hoc'),
        #'rm' : 30000,               # membrane resistance
        #'cm' : 1.0,                 # membrane capacitance
        #'Ra' : 100,                 # axial resistance
        'v_init': -60,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': 1,
        'tstartms': tstartms,          # start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_code': [join(model_path, 'custom_codes.hoc')],
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'conductance_type': 'Ih_linearized',
                             'hold_potential': -80}],
    }
    cell = LFPy.Cell(**cell_params)
    dist = []
    mu_star = []
    for sec in cell.allseclist:
        for seg in sec:
            if hasattr(seg, 'ehcn_Ih_linearized_v2'):
                dist.append(nrn.distance(seg.x, sec=sec))
                mu_star.append(seg.dwinf_Ih_linearized_v2 * (-80 - seg.ehcn_Ih_linearized_v2) * seg.gIhbar_Ih_linearized_v2 / seg.g_pas)

    plt.plot(dist, mu_star)
    plt.show()
    cell.simulate(rec_vmem=True, rec_imem=True)
    plt.plot(cell.tvec, cell.somav)
    plt.ylim([-60.1, -59.9])
    plt.show()

def return_frozen_gh(holding_potential):
    timeres = 2**-4
    cut_off = 0
    tstopms = 150
    tstartms = -cut_off
    model_path = join('lfpy_version')
    neuron.load_mechanisms('mod')
    neuron.load_mechanisms('..')
    cell_params = {
        'morphology': join(model_path, 'morphologies', 'cell1.hoc'),
        #'rm' : 30000,               # membrane resistance
        #'cm' : 1.0,                 # membrane capacitance
        #'Ra' : 100,                 # axial resistance
        'v_init': holding_potential,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': 1,
        'tstartms': tstartms,          # start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_code': [join(model_path, 'custom_codes.hoc')],
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'conductance_type': 'active_frozen',
                             'hold_potential': holding_potential}],
    }

    cell = LFPy.Cell(**cell_params)
    # make_syaptic_stimuli(cell, input_idx)
    # cell.simulate(rec_vmem=True, rec_imem=True)

    gIh = np.zeros(cell.totnsegs)
    gpas = np.zeros(cell.totnsegs)
    dist = np.zeros(cell.totnsegs)
    nrn.distance()
    comp = 0
    for sec in cell.allseclist:
        for seg in sec:
            gIh[comp] = seg.gIh_Ih_frozen if hasattr(seg, "ihcn_Ih_frozen") else 0
            gpas[comp] = seg.g_pas
            dist[comp] = nrn.distance(seg.x)
            comp += 1

    return dist, gIh, gpas

def plot_frozen_gh():
    dist, gIh_80, gpas = return_frozen_gh(-80)
    dist, gIh_70, gpas = return_frozen_gh(-70)
    dist, gIh_60, gpas = return_frozen_gh(-60)


    plt.plot(dist, gIh_80, 'bo', label='Frozen Ih at -80 mV')
    plt.plot(dist, gIh_70, 'go', label='Frozen Ih at -70 mV')
    plt.plot(dist, gIh_60, 'ro', label='Frozen Ih at -60 mV')
    plt.plot(dist, gpas, 'ko', label='g_pas')

    plt.legend(loc=2)
    plt.ylabel('Conductance [S/cm^2]')
    plt.xlabel('Distance from soma [$\mu m$]')
    plt.savefig('frozen_Ih_conductance.png')


def test_ca_initiation():

    timeres = 2**-2
    cut_off = 0
    tstopms = 6000
    tstartms = -cut_off
    model_path = join('lfpy_version')
    neuron.load_mechanisms('mod')
    cell_params = {
        'morphology': join(model_path, 'morphologies', 'cell1.hoc'),
        #'rm' : 30000,               # membrane resistance
        #'cm' : 1.0,                 # membrane capacitance
        #'Ra' : 100,                 # axial resistance
        'v_init': -60,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': 1,
        'tstartms': tstartms,          # start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_code': [join(model_path, 'custom_codes.hoc')],
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'conductance_type': 'active',
                             'hold_potential': -60}],
    }
    cell = LFPy.Cell(**cell_params)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    plt.subplot(231, title='Membrane potential soma')
    plt.plot(cell.tvec, cell.vmem[0, :])
    plt.ylim([-61.5, -59.8])
    plt.subplot(232, title='Calcium current soma')
    plt.plot(cell.tvec, cell.rec_variables['ica'][0, :])
    plt.ylim([0, -0.000008])
    plt.subplot(233, title='Calcium concentration soma')
    plt.plot(cell.tvec, cell.rec_variables['cai'][0, :])
    plt.ylim([0, 0.00015])

    apic_idx = cell.get_closest_idx(x=0, y=0, z=1000)
    plt.subplot(234, title='Membrane potential apic')
    plt.plot(cell.tvec, cell.vmem[apic_idx, :])
    plt.ylim([-61.5, -59.8])
    plt.subplot(235, title='Calcium current apic')
    plt.plot(cell.tvec, cell.rec_variables['ica'][apic_idx, :])
    plt.ylim([0, -0.000008])
    plt.subplot(236, title='Calcium concentration apic')
    plt.plot(cell.tvec, cell.rec_variables['cai'][apic_idx, :])
    plt.ylim([0, 0.00015])
    # plt.savefig('ca_initiation_init.png')
    plt.show()

if __name__ == '__main__':
    test_steady_state()
    #simulate_synaptic_input(0, -65, 'active')
    # plt.savefig('Ca_initiation_testing.png')
    # test_ca_initiation()
    #plot_frozen_gh()
    # test_frozen_currents(0, -80)
    # test_frozen_currents(0, -70)
    # test_frozen_currents(0, -60)
    # test_frozen_currents(750, -60)
    # test_frozen_currents(750, -70)
    # test_frozen_currents(750, -80)