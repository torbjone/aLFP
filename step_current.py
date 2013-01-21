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

domain = 'ball_n_stick'

np.random.seed(1234)
#neuron_model = join('neuron_models', 'mainen')
#neuron_model = join('neuron_models', 'hay_model', 'lfpy_version')
#neuron_model = join('neuron_models', 'salam_retina')
neuron_model = 'neuron_models'
if domain == 'hay':
    LFPy.cell.neuron.load_mechanisms(join(neuron_model, '..', 'mod'))
else:
    LFPy.cell.neuron.load_mechanisms(neuron_model)

def find_renormalized_Rm(cell, stim_idx):

    # r_tilde = [mV] / ( [nA] / [um]**2) = 10**-2 [Ohm] [cm]**2
    r_tilde = (cell.somav[-1] - cell.somav[stim_idx-1])/\
              (cell.imem[0,-1] - cell.imem[0, stim_idx-1]) * cell.area[0] * 10**-2
    #print cell.area[0]
    #print r_tilde
    #pl.plot(cell.tvec, R)
    #pl.show()
    #set_trace()
    return r_tilde

def return_r_m_tilde(cell):

    # r_tilde = [mV] / ( [nA] / [um]**2) = 10**-2 [Ohm] [cm]**2
    r_tilde = (cell.somav[-1] - cell.somav[0])/\
              (cell.imem[0,-1] - cell.imem[0, 0]) * cell.area[0] * 10**-2
    return r_tilde


def plot_all_currents(domain, cell_params, pulse_clamp):
    pass


def active_ball_n_stick(cell_params, is_active=True):
    #proc biophys() {
        #forsec all {
            #Ra = 150
            #cm = 1
            #insert hh
            #gnabar_hh = 0.12
            #gkbar_hh = 0.036
            #gl_hh = 0.0003
            #el_hh = -54.3
            #insert pas
            #g_pas = 0.001
            #e_pas = -70
        #    }
        #}
    for sec in neuron.h.allsec():
        sec.insert('pas')
        sec.cm = cell_params['cm']
        sec.Ra = cell_params['Ra']
        sec.g_pas = 1./ cell_params['rm']
        sec.e_pas = cell_params['e_pas']
        if is_active:
            sec.insert('hh')
            sec.gnabar_hh = 0.12
            sec.gkbar_hh = 0.036
            sec.gl_hh = 0.0003
            sec.el_hh = -54.3
 


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
        #// Insert spines
    add_spines('dend')
    
    #// Insert active channels
    set_active()


def plot_ex3(cell, stim_idx):
    '''plotting function used by example3/4'''
    fig = pl.figure(figsize=[12, 8])
    
    #plot the somatic trace
    ax = fig.add_axes([0.1, 0.7, 0.5, 0.2])
    ax.plot(cell.tvec, cell.somav)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Soma pot. [mV]')

    #plot the somatic trace
    ax = fig.add_axes([0.1, 0.4, 0.5, 0.2])
    ax.plot(cell.tvec, cell.imem[0,:])
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Soma imem [nA]')    

    ax = fig.add_axes([0.1, 0.1, 0.5, 0.2])
    ax.plot(cell.tvec[stim_idx:], cell.somav[stim_idx:] - cell.somav[stim_idx-1])
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Vm change [nA]')  

    #plot the morphology, electrode contacts and synapses
    ax = fig.add_axes([0.65, 0.1, 0.25, 0.8], frameon=False)
    for sec in neuron.h.allsec():
        idx = cell.get_idx(sec.name())
        ax.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                color='k')

    #for i in xrange(electrode.x.size):
    #    ax.plot(electrode.x[i], electrode.z[i], color='g', marker='o')
    pl.axis('equal')
    pl.axis(np.array(pl.axis())*0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig

def run_simulation(cell_params, clamp_params):
    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params)
    currentClamp = LFPy.StimIntElectrode(cell, **clamp_params)
    simulation_params = {'rec_isyn': True,
                         'rec_imem': True,
                         'rec_ipas': True,
                         'rec_istim': True,
                         'rec_variables' : ['ina', 'ik'],

                         }
    #for sec in neuron.h.allsec():
    #    for seg in sec:
    #        set_trace()
    cell.simulate(**simulation_params)
    #del currentClamp
    return cell

def check_when_linear(domain, cell_params, clamp_params):

    input_amps = np.linspace(0.0,.01,2)
    #input_amps = [0.01]
    #input_amps[-1] = 0.01
    #do_sim = 0
    
    if do_sim:
        try:
            os.mkdir(join(domain, 'lin_check'))
        except OSError:
            pass
        if conductances == 'renormalized':
            renormalized_rm = np.load(join(domain, 'lin_check', 'renormalized_rm_%s.npy' % 'active'))
        else:
            renormalized_rm = []
        for number, amp in enumerate(input_amps):
            pulse_clamp['amp'] = amp
            if conductances == 'renormalized':
                cell_params['rm'] = renormalized_rm[number][1]
            cell = run_simulation(cell_params, clamp_params)
            print cell_params['e_pas']
            print cell.somav[0]
            if not conductances == 'renormalized':
                renormalized_rm.append((amp, return_r_m_tilde(cell)))
            np.save(join(domain, 'lin_check', 'somav_%s_%g.npy' % (conductances, amp)), cell.somav)
            np.save(join(domain, 'lin_check', 'somai_%s_%g.npy' % (conductances, amp)), cell.imem[0,:])
            np.save(join(domain, 'lin_check', 'soma_a.npy'), cell.area[0])
            #cell.cellpickler(join(domain, 'lin_check', 'cell_%s_%g.cpickle' % (conductances, amp)))
            del cell
        np.save(join(domain, 'lin_check', 'renormalized_rm_%s.npy' % conductances), renormalized_rm)
    else:
        renormalized_rm = np.load(join(domain, 'lin_check', 'renormalized_rm_%s.npy' % 'active'))
        response = []
        soma_a = np.load(join(domain, 'lin_check', 'soma_a.npy'))
        for number, amp in enumerate(input_amps):    
            #f = file(join(domain, 'lin_check', 'cell_%s_%g.cpickle' % (conductances, amp)))
            #cell = cPickle.load(f)
            #f.close()
            vm = np.load(join(domain, 'lin_check', 'somav_%s_%g.npy' % (conductances, amp)))
            im = np.load(join(domain, 'lin_check', 'somai_%s_%g.npy' % (conductances, amp)))

            tvec = np.linspace(0, cell_params['tstopms'], len(im))
            response.append((amp, vm[-1], im[-1]))
            #pl.suptitle(domain)
            pl.subplot(131)
            pl.plot(tvec, vm, label = 'I: %g, O: %g' %(amp, vm[-1]))
            pl.axis([0, cell_params['tstopms'], -65.1, -64.5])
            pl.subplot(132)
            pl.plot(tvec, im[:], label = 'I: %g, O: %g, $\~Rm$: %g' %(amp, im[-1]- im[0], renormalized_rm[number][1]))
            pl.axis([0, cell_params['tstopms'], -0.00005, 0.0004])
        response = np.array(response)
        pl.legend(loc='upper left', bbox_to_anchor=(-0.2, 1.1))
        pl.subplot(133)

        response[:,1] -= np.min(response[:,1])
        response[:,1] /= np.max(response[:,1])
        response[:,2] -= np.min(response[:,2])
        response[:,2] /= np.max(response[:,2])
        slope_1 = (response[1,1] - response[0,1])/(response[1,0] - response[0,0])
        slope_2 = (response[1,2] - response[0,2])/(response[1,0] - response[0,0])

        
        pl.plot(response[:,0], slope_1*response[:,0], '-', color='grey', lw=1)
        pl.plot(response[:,0], response[:,1], '-D', lw=2, label = 'Vm')
        pl.plot(response[:,0], slope_2*response[:,0], '-', color='grey', lw=1)
        pl.plot(response[:,0], response[:,2], '-o', label = 'Im')

        pl.xlabel('Input amp [nA]')
        pl.ylabel('Response')
        pl.legend(loc='upper left')
        #pl.savefig(join(domain, 'response_%s.png' % conductances))
        pl.show()


def find_renormalized_Im(domain, cell_params, clamp_params):
    do_sim = 0
    domain = 'renorm_im'
    if do_sim:
        # Do the active cell
        cell_params['custom_fun_args'][0]['is_active'] = True
        cell = run_simulation(cell_params, clamp_params)
        np.save(join('renorm_im', 'active_im_%g_%g_%g.npy' %(cell_params['rm'], cell_params['Ra'], cell_params['cm'])), cell.imem[0,:])
        np.save(join('renorm_im', 'active_vm_%g_%g_%g.npy' %(cell_params['rm'], cell_params['Ra'], cell_params['cm'])), cell.somav)
        del cell

        # Do the passive cell
        r_ms = [3000, 10000, 30000, 100000, 200000]
        Ras = [15, 50, 100, 150, 200, 500, 1500]
        cms = [0.1, 0.5, 1.0, 1.5, 5, 10]
        for r_m in r_ms:
            for Ra in Ras:
                for cm in cms:
                    cell_params['custom_fun_args'][0]['is_active'] = False
                    cell_params['rm'] = r_m
                    cell_params['Ra'] = Ra
                    cell_params['cm'] = cm
                    cell = run_simulation(cell_params, clamp_params)
                    np.save(join('renorm_im', 'passive_im_%g_%g_%g.npy' %(r_m, Ra, cm)), cell.imem[0,:])
                    np.save(join('renorm_im', 'passive_vm_%g_%g_%g.npy' %(r_m, Ra, cm)), cell.somav)
                    del cell
    else:
        a_im = np.load(join('renorm_im', 'active_im_%g_%g_%g.npy' %(cell_params['rm'], cell_params['Ra'], cell_params['cm'])))
        a_vm = np.load(join('renorm_im', 'active_vm_%g_%g_%g.npy' %(cell_params['rm'], cell_params['Ra'], cell_params['cm'])))
        tvec = np.linspace(0, cell_params['tstopms'], len(a_im))
        pl.subplots_adjust(wspace=0.5)
        pl.subplot(121)
        pl.plot(tvec, a_im - np.min(a_im), '--')
        pl.xlabel('Time [ms]')
        pl.ylabel('$I_m$ [nA]')        
        pl.subplot(122)
        pl.plot(tvec, a_vm - np.min(a_vm), '--')
        pl.xlabel('Time [ms]')
        pl.ylabel('$V_m$ [mV]')
        pl.suptitle('Active: Rm = 30000, Ra = 150, cm = 1.0. \nPassive: Combinations from 0.1 to 10 times all of them') 
        r_ms = [3000, 10000, 30000, 100000, 200000]
        Ras = [15, 50, 100, 150, 200, 500, 1500]
        cms = [0.1, 0.5, 1.0, 1.5, 5, 10]
        for r_m in r_ms:
            for Ra in Ras:
                for cm in cms:
                    try:
                        p_im = np.load(join('renorm_im', 'passive_im_%g_%g_%g.npy' %(r_m, Ra, cm)))
                        p_vm = np.load(join('renorm_im', 'passive_vm_%g_%g_%g.npy' %(r_m, Ra, cm)))
                        pl.subplot(121)
                        pl.plot(tvec, p_im- np.min(p_im))
                        pl.subplot(122)
                        pl.plot(tvec, p_vm - np.min(p_vm))
                    except:
                        pass
        pl.show()

def find_renormalized_passive(domain, cell_params, clamp_params):
    do_sim = 0
    if do_sim:
        # Do the active cell
        #cell_params['custom_fun_args'][0]['is_active'] = True
        #cell = run_simulation(cell_params, clamp_params)
        #cell.cellpickler(join(domain, 'compare', 'cell_active_%g.cpickle' % cell_params['rm']))
        #del cell
        # Do the passive cell
        r_ms = [9452]
        for r_m in r_ms:#[110000, 120000]:
            cell_params['custom_fun_args'][0]['is_active'] = False
            cell_params['rm'] = r_m
            cell = run_simulation(cell_params, clamp_params)             
            cell.cellpickler(join(domain, 'compare','cell_passive_%g.cpickle' % (r_m)))
            del cell
    else:
        f = file(join(domain, 'compare', 'cell_active_%g.cpickle' % cell_params['rm']))
        cell = cPickle.load(f)
        f.close() 
        r_tilde = return_r_m_tilde(cell)
        print cell_params['rm'], r_tilde
        pl.figure(figsize=[8,8])
        dv = cell.somav[-1] - cell.somav[0]
        di = cell.imem[0,-1] - cell.imem[0,0]
        soma_a = cell.area[0]
        print dv, di, soma_a
        pl.suptitle('Input %g nA\n $\Delta V_m / (\Delta I_m / (A_{soma}\ \mu m)^2 ) * 10^2 = 9452\ \Omega\ cm^2$' %input_amp)
        #r_tilde = $\Delta$ $V_m$ [mV] / ($\Delta I_m$ [nA] / A [$\mu$ m]**2) = 10**-2 [$\Omega$] [cm]**2

        pl.subplot(221)
        pl.title('Vm')
        pl.plot(cell.tvec, cell.somav)
        pl.subplot(222)
        pl.title('Vm shifted')
        pl.plot(cell.tvec, cell.somav- cell.somav[0])
        pl.subplot(223)
        pl.title('Im')
        pl.plot(cell.tvec, cell.imem[0,:])
        pl.subplot(224)
        pl.title('Im shifted')
        pl.plot(cell.tvec, cell.imem[0,:]- cell.imem[0,0], label='A 30000')

        r_ms = [9452, 100000]
        for r_m in r_ms:        
            f = file(join(domain, 'compare', 'cell_passive_%g.cpickle' %(r_m)))
            cell = cPickle.load(f)

            r_tilde = return_r_m_tilde(cell)
            print r_m, r_tilde
            
            pl.subplot(221)
            pl.plot(cell.tvec, cell.somav)
            pl.subplot(222)
            pl.plot(cell.tvec, cell.somav- cell.somav[0])
            pl.subplot(223)
            pl.plot(cell.tvec, cell.imem[0,:])
            pl.subplot(224)
            pl.plot(cell.tvec, cell.imem[0,:]- cell.imem[0,0], label='P %d'%r_m)
            f.close()
        pl.legend(loc='upper left')
        pl.savefig(join(domain, 'compare', '_%g.png' % input_amp))
        #pl.show()
        
conductances = sys.argv[1]

input_amp = 0.0001
input_delay = 100
is_active = None

## cell_params = {
##     'morphology' : join(neuron_model, 'L5_Mainen96_wAxon_LFPy.hoc'),
##     #'morphology' : os.path.join('neuron_models', 'example_morphology.hoc'),
##     'rm' : 30000,               # membrane resistance
##     'cm' : 1.0,                 # membrane capacitance
##     'Ra' : 150,                 # axial resistance
##     'v_init' : -65,             # initial crossmembrane potential
##     'e_pas' : -65,              # reversal potential passive mechs
##     'passive' : True,           # switch on passive mechs
##     'nsegs_method' : 'lambda_f',# method for setting number of segments,
##     'lambda_f' : 100,           # segments are isopotential at this frequency
##     'timeres_NEURON' : 2**-4,   # dt of LFP and NEURON simulation.
##     'timeres_python' : 2**-4,
##     'tstartms' : -500,          #start time, recorders start at t=0
##     'tstopms' : 1000,           #stop time of simulation
##     'custom_fun'  : [active_declarations], # will execute this function
##     'custom_fun_args' : [{'is_active': is_active}],
## }

## cell_params = {
##     'morphology' : join(neuron_model, 'morphologies', 'cell1.hoc'),
##     #'rm' : 30000,               # membrane resistance
##     #'cm' : 1.0,                 # membrane capacitance
##     #'Ra' : 100,                 # axial resistance
##     'v_init' : -77,             # initial crossmembrane potential
##     #'e_pas' : -90,              # reversal potential passive mechs
##     'passive' : False,           # switch on passive mechs
##     'nsegs_method' : 'lambda_f',# method for setting number of segments,
##     'lambda_f' : 100,           # segments are isopotential at this frequency
##     'timeres_NEURON' : 2**-4,   # dt of LFP and NEURON simulation.
##     'timeres_python' : 2**-4,
##     'tstartms' : -5000,          #start time, recorders start at t=0
##     'tstopms' : 1000,           #stop time of simulation
##     'custom_code'  : [join(neuron_model, 'custom_codes.hoc'), \
##                       join(neuron_model, 'biophys3_passive.hoc')],
## }

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
    'timeres_NEURON' : 2**-4,   # dt of LFP and NEURON simulation.
    'timeres_python' : 2**-4,
    'tstartms' : -100,          #start time, recorders start at t=0
    'tstopms' : 400,           #stop time of simulation
    'custom_fun'  : [active_ball_n_stick], # will execute this function
    #'custom_fun_args' : [{'is_active': is_active}],  
}


## cell_params = {
##     'morphology' : join(neuron_model, 'lws9287aMorph.hoc'),
##     'rm' : 30000,               # membrane resistance
##     'cm' : 1.0,                 # membrane capacitance
##     'Ra' : 150,                 # axial resistance
##     'v_init' : -65,             # initial crossmembrane potential
##     'e_pas' : -65,              # reversal potential passive mechs
##     'passive' : True,           # switch on passive mechs
##     'nsegs_method' : 'lambda_f',# method for setting number of segments,
##     'lambda_f' : 100,           # segments are isopotential at this frequency
##     'timeres_NEURON' : 2**-4,   # dt of LFP and NEURON simulation.
##     'timeres_python' : 2**-4,
##     'tstartms' : -500,          #start time, recorders start at t=0
##     'tstopms' : 1000,           #stop time of simulation
##     'custom_code'  : [join(neuron_model, 'lws9287a_passive.hoc')],
## }


pulse_clamp = {
    'idx' : 0,
    'record_current' : True,
    'amp' : input_amp, #[nA]
    'dur' : 10000.,
    'delay' : input_delay, 
    'pptype' : 'IClamp',
}


try:
    #renormalized_r_m = float(sys.argv[2])
    do_sim = int(sys.argv[2])
except:
    pass

if conductances == 'active':
    is_active = True
    r_m = 30000
elif conductances == 'passive':
    is_active = False
    r_m = 30000
elif conductances == 'renormalized':
    is_active = False
    cell_params['rm'] = 830
    cell_params['e_pas'] = -64.97
elif conductances == 'compare':
    find_renormalized_passive(domain, cell_params, pulse_clamp)
    sys.exit()
elif conductances == 'isearch':
    find_renormalized_Im(domain, cell_params, pulse_clamp)
    sys.exit()
else:
    print "active, passive or renormalized?"
    raise RuntimeError

try:
    os.mkdir(domain)
except OSError:
    pass
if domain == 'mainen':
    cell_params['custom_fun_args'] = [{'is_active': is_active}]
elif domain == 'hay':
    cell_params['custom_code'] = [join(neuron_model, 'custom_codes.hoc'), \
                                  join(neuron_model, 'biophys3_%s.hoc' % conductances)]
elif domain == 'retina':
    cell_params['custom_code'] = [join(neuron_model, 'lws9287a_%s.hoc' % conductances)]
elif domain == 'ball_n_stick':
    cell_params['custom_fun_args'] = [{'is_active': is_active, 'cell_params' : cell_params}]

plot_all_currents(domain, cell_params, pulse_clamp)
#check_when_linear(domain, cell_params, pulse_clamp)
sys.exit()

cell = run_sim(cell_params, pulse_clamp)

np.save(join(domain, 'somav_%s_%g.npy' % (conductances, r_m)), cell.somav)
np.save(join(domain, 'somai_%s_%g.npy' % (conductances, r_m)), cell.imem[0,:])
t_tilde = find_renormalized_Rm(cell, stim_idx)

fig = plot_ex3(cell, stim_idx)
fig.savefig(join(domain, '%s.png' % conductances))

#pl.show()
