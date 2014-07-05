__author__ = 'torbjone'

import os
import sys
from os.path import join
import numpy as np
import pylab as plt
import aLFP
import neuron
import LFPy
nrn = neuron.h


def biophys_passive(**kwargs):

    Rm = 28000.
    Cm = 1.
    RaAll = 150.
    RaAx = 50.
    Vrest = kwargs['hold_potential']
    nrn("celsius = 35.0")
    nrn("access soma")
    nrn.distance()
    for sec in nrn.allsec():
        sec.insert("pas")
        sec.e_pas = Vrest
        sec.g_pas = 1/Rm
        sec.Ra = RaAll
        sec.cm = Cm
    for sec in nrn.axon:
        sec.Ra = RaAx
    if 'hold_potential' in kwargs:
        make_uniform(kwargs['hold_potential'])

def biophys_active(**kwargs):

    Rm = 28000.
    Cm = 1.
    RaAll = 150.
    RaAx = 50.
    Vrest = kwargs['hold_potential']

    gna = .045
    AXONM = 3.
    gkdr = 0.06
    nrn("celsius = 35.0")
    ka = 0.04
    ghd = 0.00005
    gkm = 0.04
    gcat = 0.0001
    gahp = 0.00001
    nrn("access soma")

    for sec in nrn.allsec():
        sec.insert("pas")
        sec.e_pas = Vrest
        sec.g_pas = 1/Rm
        sec.Ra = RaAll
        sec.cm = Cm

    for sec in nrn.allsec():
        if not sec.name().startswith('axon'):
            continue
        sec.Ra = RaAx
        sec.insert("km")
        sec.gbar_km = gkm
        sec.insert("nax")
        sec.gbar_nax = gna*AXONM
        sec.insert("kdr")
        sec.gkdrbar_kdr = gkdr*AXONM
        sec.insert("kap")
        sec.gkabar_kap = ka
    for sec in nrn.allsec():
        if not sec.name().startswith('soma'):
            continue
        sec.insert("km")
        sec.gbar_km = gkm
        sec.insert("hd")
        sec.ghdbar_hd = ghd
        sec.insert("na3")
        sec.gbar_na3 = gna
        sec.insert("kdr")
        sec.gkdrbar_kdr = gkdr
        sec.insert("kap")
        sec.gkabar_kap = ka
        sec.insert("cat")
        sec.gcatbar_cat = gcat
        sec.insert("cacum")
        sec.tau_cacum = 100.
        sec.depth_cacum = sec.diam/2.
        sec.insert("KahpM95")
        sec.gbar_KahpM95 = gahp

    for sec in nrn.allsec():
        if not sec.name().startswith('dendrite'):
            continue

        sec.insert("na3")
        sec.gbar_na3 = gna
        sec.insert("cat")
        sec.gcatbar_cat = gcat
        sec.insert("cacum")
        sec.tau_cacum = 100.
        sec.depth_cacum = sec.diam/2.
        sec.insert("KahpM95")
        sec.gbar_KahpM95 = gahp

    nrn.distance(0, 0.5)
    for sec in nrn.allsec():
        if not sec.name().startswith('user5'):
            continue

        sec.insert("hd")
        sec.ghdbar_hd = ghd
        sec.insert("na3")
        sec.gbar_na3 = gna
        sec.insert("kdr")
        sec.gkdrbar_kdr = gkdr
        sec.insert("kap")
        sec.gkabar_kap = 0.
        sec.insert("kad")
        sec.gkabar_kad = 0.
        sec.insert("cacum")
        sec.tau_cacum = 100.
        sec.depth_cacum = sec.diam/2.
        sec.insert("cat")
        sec.gcatbar_cat = gcat
        sec.insert("KahpM95")
        sec.gbar_KahpM95 = gahp

        for seg in sec:
            xdist = nrn.distance(seg.x)
            seg.ghdbar_hd = ghd*(1. + 3. * xdist / 100.)
            if xdist > 100:
                seg.gkabar_kad = ka*(1. + xdist / 100.)
            else:
                seg.gkabar_kap = ka*(1. + xdist / 100.)

    nrn.distance(0, 0.5)
    for sec in nrn.allsec():
        if not sec.name().startswith('apical_dendrite'):
            continue

        sec.insert("hd")
        sec.ghdbar_hd = ghd
        sec.insert("na3")
        sec.gbar_na3 = gna
        sec.insert("kdr")
        sec.gkdrbar_kdr = gkdr
        sec.insert("kap")
        sec.gkabar_kap = 0
        sec.insert("kad")
        sec.gkabar_kad = 0
        sec.insert("cacum")
        sec.tau_cacum = 100.
        sec.depth_cacum = sec.diam/2.
        sec.insert("KahpM95")
        sec.gbar_KahpM95 = gahp
        sec.insert("cat")
        sec.gcatbar_cat = gcat
        for seg in sec:
            xdist = nrn.distance(seg.x)
            print sec.name(), seg.x, xdist
            seg.ghdbar_hd = ghd*(1. + 3. * xdist/100.)
            if xdist > 100:
                seg.gkabar_kad = ka*(1. + xdist/100.)
            else:
                seg.gkabar_kap = ka*(1.+xdist/100.)
    nrn.t = 0
    for sec in nrn.allsec():
        sec.v = Vrest
        if nrn.ismembrane("na_ion"):
            sec.ena = 55
        if nrn.ismembrane("k_ion"):
            sec.ek = -90
        if nrn.ismembrane("hd"):
            sec.ehd_hd = -30

    if 'hold_potential' in kwargs:
        make_uniform(kwargs['hold_potential'])

def active_declarations(**kwargs):
    ''' Set active conductances for modified CA1 cell
    '''
    nrn.define_shape()
    exec('biophys_%s(**kwargs)' % kwargs['conductance_type'])


def make_uniform(Vrest):
    """ Makes the cell uniform. Doesn't really work for INaP yet,
    since it is way to strong it seems
    """
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
                seg.e_pas += seg.ica/seg.g_pas
            if nrn.ismembrane("hd"):
                seg.e_pas += seg.i_hd/seg.g_pas

def make_syaptic_stimuli(cell, input_idx):
    # Define synapse parameters
    synapse_parameters = {
        'idx': input_idx,
        'e': 0.,                   # reversal potential
        'syntype': 'ExpSyn',       # synapse type
        'tau': 10.,                # syn. time constant
        'weight': 0.001,            # syn. weight
        'record_current': True,
    }
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([5.]))
    return cell, synapse

def test_original_steady_state():
    model_path = '.'
    timeres_NEURON = 2**-3
    timeres_python = 2**-3
    cut_off = 0
    tstopms = 100

    cell_params = {
        'morphology': join(model_path, 'geo9068802.hoc'),
        'v_init': -70,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',# method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres_NEURON,   # dt of LFP and NEURON simulation.
        'timeres_python': timeres_python,
        'tstartms': -cut_off,          #start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_code': [join(model_path, 'fig-3c_active.hoc')],
    }
    sim_params = {'rec_imem': True,
                  'rec_vmem': True,
                 }
    cell = LFPy.Cell(**cell_params)
    cell.simulate(**sim_params)
    [plt.plot(cell.tvec, cell.vmem[idx, :]) for idx in xrange(len(cell.xmid))]
    plt.show()
    img = plt.scatter(cell.xmid, cell.zmid, c=cell.vmem[:, -1], edgecolor='none', vmin=-71, vmax=-69)
    plt.axis('equal')
    plt.colorbar(img)
    plt.show()

def test_python_steady_state():
    model_path = '.'
    timeres_NEURON = 2**-3
    timeres_python = 2**-3
    cut_off = 0
    tstopms = 100
    hold_potential = -70

    cell_params = {
        'morphology': join(model_path, 'geo9068802.hoc'),
        'v_init': hold_potential,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',# method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres_NEURON,   # dt of LFP and NEURON simulation.
        'timeres_python': timeres_python,
        'tstartms': -cut_off,          #start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_fun': [active_declarations],
        'custom_fun_args': [{'hold_potential': hold_potential,
                             'conductance_type': 'active'}]
    }
    sim_params = {'rec_imem': True,
                  'rec_vmem': True,
                 }
    cell = LFPy.Cell(**cell_params)
    cell.simulate(**sim_params)
    [plt.plot(cell.tvec, cell.vmem[idx, :]) for idx in xrange(len(cell.xmid))]
    plt.show()
    img = plt.scatter(cell.xmid, cell.zmid, c=cell.vmem[:, -1], edgecolor='none', vmin=-71, vmax=-69)
    plt.axis('equal')
    plt.colorbar(img)
    plt.show()

def test_original_synaptic_input():
    model_path = '.'
    timeres_NEURON = 2**-3
    timeres_python = 2**-3
    cut_off = 0
    tstopms = 100
    input_idx = 10
    holding_potential = -70

    cell_params = {
        'morphology': join(model_path, 'geo9068802.hoc'),
        'v_init': holding_potential,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',# method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres_NEURON,   # dt of LFP and NEURON simulation.
        'timeres_python': timeres_python,
        'tstartms': -cut_off,          #start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_code': [join(model_path, 'fig-3c_active.hoc')],
    }
    sim_params = {'rec_imem': True,
                  'rec_vmem': True,
                 }
    cell = LFPy.Cell(**cell_params)

    make_syaptic_stimuli(cell, input_idx)
    # for sec in cell.allseclist:
    #     for seg in sec:
    #         print sec.name(), seg.g_pas

    cell.simulate(**sim_params)
    plt.close('all')
    plt.subplot(121)
    plt.scatter(cell.xmid, cell.zmid, edgecolor='none')
    plt.plot(cell.xmid[input_idx], cell.zmid[input_idx], 'y*', ms=20)

    plt.subplot(222, title='Soma', ylim=[-71, -69])
    plt.plot(cell.tvec, cell.vmem[0, :],
             label='%d %d mV' % (input_idx, holding_potential))

    plt.subplot(224, title='Input idx %d' % input_idx, ylim=[-71, -66])
    plt.plot(cell.tvec, cell.vmem[input_idx, :],
             label='%d %d mV' % (input_idx, holding_potential))
    plt.savefig(join(model_path, 'syn_%d_original.png' % input_idx))

def test_python_synaptic_input():
    model_path = '.'
    timeres_NEURON = 2**-3
    timeres_python = 2**-3
    cut_off = 0
    tstopms = 100
    input_idx = 10
    holding_potential = -70

    cell_params = {
        'morphology': join(model_path, 'geo9068802.hoc'),
        'v_init': holding_potential,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',# method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres_NEURON,   # dt of LFP and NEURON simulation.
        'timeres_python': timeres_python,
        'tstartms': -cut_off,          #start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_fun': [active_declarations],
        'custom_fun_args': [{'hold_potential': holding_potential,
                             'conductance_type': 'active'}]
    }
    sim_params = {'rec_imem': True,
                  'rec_vmem': True,
                 }
    cell = LFPy.Cell(**cell_params)

    make_syaptic_stimuli(cell, input_idx)
    # for sec in cell.allseclist:
    #     for seg in sec:
    #         print sec.name(), seg.g_pas
    cell.simulate(**sim_params)
    plt.close('all')
    plt.subplot(121)
    plt.scatter(cell.xmid, cell.zmid, edgecolor='none')
    plt.plot(cell.xmid[input_idx], cell.zmid[input_idx], 'y*', ms=20)

    plt.subplot(222, title='Soma', ylim=[-71, -69])
    plt.plot(cell.tvec, cell.vmem[0, :], label='%d %d mV' % (input_idx, holding_potential))

    plt.subplot(224, title='Input idx %d' % input_idx, ylim=[-71, -66])
    plt.plot(cell.tvec, cell.vmem[input_idx, :], label='%d %d mV' % (input_idx, holding_potential))
    plt.savefig(join(model_path, 'syn_%d_python.png' % input_idx))



if __name__ == '__main__':
    test_original_synaptic_input()
    # test_python_steady_state()
    test_python_synaptic_input()