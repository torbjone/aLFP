__author__ = 'torbjone'

import os
from os.path import join
import sys
import neuron
from neuron import h as nrn
import LFPy
import numpy as np
import pylab as plt
import scipy.fftpack as ff
from c12861_declarations import active_declarations

def find_LFP_power(sig, timestep):
    """ Returns the power and freqency of the input signal"""
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]
    # import ipdb; ipdb.set_trace()
    Y = ff.fft(sig, axis=1)[:, pidxs[0]]
    power = np.abs(Y)/Y.shape[1]
    return freqs, power


def make_WN_input(cell_params, max_freq):
    """ White Noise input ala Linden 2010 is made """
    tot_ntsteps = round((cell_params['tstopms'] - cell_params['tstartms'])/\
                  cell_params['timeres_NEURON'] + 1)
    I = np.zeros(tot_ntsteps)
    tvec = np.arange(tot_ntsteps) * cell_params['timeres_NEURON']
    #I = np.random.random(tot_ntsteps) - 0.5
    for freq in xrange(1, max_freq + 1):
        I += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())
    return I


def make_WN_stimuli(cell, cell_params, input_idx, input_scaling, max_freq=1000):

    input_array = input_scaling * make_WN_input(cell_params, max_freq)

    noiseVec = neuron.h.Vector(input_array)
    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print "Input i" \
                      "nserted in ", sec.name()
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if type(syn) == type(None):
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noiseVec.play(syn._ref_amp, cell.timeres_NEURON)
    return cell, noiseVec, syn


def insert_synapses(synparams, cell, section, n, spTimesFun, args):
    ''' find n compartments to insert synapses onto '''
    idx = cell.get_rand_idx_area_norm(section=section, nidx=n)
    #Insert synapses in an iterative fashion
    for i in idx:
        synparams.update({'idx' : int(i)})
        # Some input spike train using the function call
        spiketimes = spTimesFun(args[0], args[1], args[2], args[3], args[4])
        # Create synapse(s) and setting times using the Synapse class in LFPy
        s = LFPy.Synapse(cell, **synparams)
        s.set_spike_times(spiketimes)


def plot_dynamics():

    e_rev = 30
    vhalfn = -47.
    z = 6.5
    gamma = 0.5
    tau0 = 1.
    K = 0.0
    R = 8315.
    F = 9.648e4
    celsius = 33
    v = np.linspace(-100, 0, 1000)

    alpn = np.exp(z * gamma * (v-vhalfn) * F / (R * (273.16+celsius)))
    betn = np.exp(-z * (1 - gamma) * (v-vhalfn) * F / (R * (273.16+celsius)))

    ninf = alpn / (alpn + betn)

    plt.plot(v, ninf)

    plt.plot(v, alpn)
    plt.plot(v, betn)
    plt.show()


def plot_cell_dynamics_state(cell):
    plt.subplot(221, title='n')
    plt.plot(cell.tvec, cell.rec_variables['n_debug_BG'][0,:])
    plt.subplot(222, title='n_inf')
    plt.plot(cell.tvec, cell.rec_variables['ninf_debug_BG'][0,:])
    plt.subplot(223, title='alpha and beta')
    plt.plot(cell.tvec, cell.rec_variables['alpn_debug_BG'][0,:])
    plt.plot(cell.tvec, cell.rec_variables['betn_debug_BG'][0,:])
    plt.subplot(224, title='tau_n')
    plt.plot(cell.tvec, cell.rec_variables['taun_debug_BG'][0,:])

    #plt.ylim(-100, 100)
    plt.show()
    #plot_cell(cell)


def plot_cell_steady_state(cell):
    plt.subplot(121)
    plt.scatter(cell.ymid, -cell.xmid, c=cell.vmem[:,-1], edgecolor='none',
                vmax=(np.max(cell.vmem[:,-1]) + 1), vmin=(np.min(cell.vmem[:,-1]) - 1))
    plt.colorbar()
    plt.axis('equal')

    plt.subplot(122)
    [plt.plot(cell.tvec, cell.vmem[idx, :]) for idx in xrange(len(cell.xmid))]

    plt.show()


def plot_resonances(cell, input_idx, input_scaling, idx_list, cell_params, figfolder):

    plt.figure(figsize=[12, 8])
    clr = lambda idx: plt.cm.jet(int(256. * idx/(len(idx_list) - 1)))

    vmem = cell.vmem[idx_list, :]
    imem = cell.imem[idx_list, :]

    plt.subplot(131, title='Star marks white noise input')
    plt.scatter(cell.xmid, cell.ymid, c='grey', edgecolor='none')
    [plt.plot(cell.xmid[idx], cell.ymid[idx], 'D', color=clr(numb)) for numb, idx in enumerate(idx_list)]
    plt.plot(cell.xmid[input_idx], cell.ymid[input_idx], '*', color='y', ms=15)
    plt.axis('equal')

    plt.subplot(232, title='Vmem')
    [plt.plot(cell.tvec, vmem[numb, :], color=clr(numb)) for numb, idx in enumerate(idx_list)]

    plt.subplot(235, title='Imem')
    [plt.plot(cell.tvec, imem[numb, :], color=clr(numb)) for numb, idx in enumerate(idx_list)]

    plt.subplot(233, xlim=[0, 15], ylim=[0, .4], title='Vmem PSD')
    plt.grid()
    freqs, vmem_psd = find_LFP_power(vmem[:, :-1], cell.timeres_python/1000.)
    [plt.plot(freqs, vmem_psd[numb], color=clr(numb)) for numb, idx in enumerate(idx_list)]
    # upper_lim = 10**(int(np.ceil(np.log10(np.max(vmem_psd[:, 1:])))))
    # lims = plt.axis()
    # plt.axis([lims[0], lims[1], upper_lim/1e3, upper_lim])
    # plt.axis([lims[0], lims[1], 1e-3, 1e-1])
    # print freqs

    plt.subplot(236, xlim=[0, 15], title='Imem PSD')
    freqs, imem_psd = find_LFP_power(imem[:, :-1], cell.timeres_python/1000.)
    [plt.semilogy(freqs, imem_psd[numb], color=clr(numb)) for numb, idx in enumerate(idx_list)]
    upper_lim = 10**(1+int(np.ceil(np.log10(np.max(imem_psd[:, 1:])))))
    lims = plt.axis()
    # plt.axis([lims[0], lims[1], 1e-7, 1e-3])
    plt.axis([lims[0], lims[1], upper_lim/1e5, upper_lim])

    plt.grid()

    fig_name = 'resonance_%d_%1.3f' % (input_idx, input_scaling)
    if 'use_channels' in cell_params['custom_fun_args'][0] and \
                    len(cell_params['custom_fun_args'][0]['use_channels']) > 0:
        for ion in cell_params['custom_fun_args'][0]['use_channels']:
            fig_name += '_%s' % ion
    else:
        fig_name += '_passive'

    if 'hold_potential' in cell_params['custom_fun_args'][0]:
        fig_name += '_%+d' % cell_params['custom_fun_args'][0]['hold_potential']
    print "Saving ", fig_name
    plt.savefig(join(figfolder, '%s.png' % fig_name), dpi=150)
    #plt.show()


def plot_resonance_to_ax(ax, input_idx, hold_potential, simfolder):

    control_clr = 'r'
    reduced_clr = 'b'
    numsims = 1

    control_name = 'simple_%d_0.001_Ih_Im_INaP_%+d' %(input_idx, hold_potential)
    freqs = np.load(join(simfolder, '%s_freqs.npy' % control_name))
    if hold_potential == -80:
        reduced_name = 'simple_%d_0.001_Im_INaP_%+d' % (input_idx, hold_potential)
        reduced_label = 'No Ih'
    elif hold_potential == -65:
        reduced_name = 'simple_%d_0.001_Ih_INaP_%+d' % (input_idx, hold_potential)
        reduced_label = 'No Im'
    else:
        raise RuntimeError("Not recognized holding potential")

    # vmem_psd_control = np.zeros(len(freqs))
    # vmem_psd_reduced = np.zeros(len(freqs))
    # for sim_idx in xrange(numsims):
    vmem_psd_control = np.load(join(simfolder, '%s_vmem_psd.npy' % (control_name)))
    vmem_psd_reduced = np.load(join(simfolder, '%s_vmem_psd.npy' % (reduced_name)))

    # vmem_psd_control /= numsims
    # vmem_psd_reduced /= numsims

    lc, = ax.plot(freqs[1:16], vmem_psd_control[1:16], color=control_clr)
    lr, = ax.plot(freqs[1:16], vmem_psd_reduced[1:16], color=reduced_clr)

    lines = [lc, lr]
    line_names = ['Control', reduced_label]
    ax.legend(lines, line_names, frameon=False)


def plot_morph_to_ax(ax, apic_idx, soma_idx, simfolder):
    xstart = np.load(join(simfolder, 'xstart.npy'))
    ystart = np.load(join(simfolder, 'ystart.npy'))
    xend = np.load(join(simfolder, 'xend.npy'))
    yend = np.load(join(simfolder, 'yend.npy'))
    xmid = np.load(join(simfolder, 'xmid.npy'))
    ymid = np.load(join(simfolder, 'ymid.npy'))

    [ax.plot([xstart[idx], xend[idx]], [ystart[idx], yend[idx]], color='gray')
                for idx in xrange(len(xend))]
    ax.plot(xmid[soma_idx], ymid[soma_idx], 'D', color='r', ms=10)
    ax.plot(xmid[apic_idx], ymid[apic_idx], '*', color='y', ms=15)
    ax.axis('equal')


def recreate_Hu_figs(simfolder, figfolder):

    fig = plt.figure(figsize=[12, 8])
    # clr = lambda idx: plt.cm.jet(int(256. * idx/(len(idx_list) - 1)))

    apic_idx = 573
    soma_idx = 0

    hyp_potential = -80
    dep_potential = -65

    morph_ax = fig.add_subplot(131, title='Star marks white noise input')
    soma_hyp_ax = fig.add_subplot(232, ylim=[0, 0.2], xlim=[0, 16], title='Soma input %d mV' % hyp_potential)
    apic_hyp_ax = fig.add_subplot(233, ylim=[0, 0.2], xlim=[0, 16], title='Apic input %d mV' % hyp_potential)
    soma_dep_ax = fig.add_subplot(235, ylim=[0, 0.2], xlim=[0, 16], title='Soma input %d mV' % dep_potential)
    apic_dep_ax = fig.add_subplot(236, ylim=[0, 0.2], xlim=[0, 16], title='Apic input %d mV' % dep_potential)

    plot_morph_to_ax(morph_ax, apic_idx, soma_idx, simfolder)

    plot_resonance_to_ax(apic_dep_ax, apic_idx, dep_potential, simfolder)
    plot_resonance_to_ax(soma_dep_ax, soma_idx, dep_potential, simfolder)
    plot_resonance_to_ax(apic_hyp_ax, apic_idx, hyp_potential, simfolder)
    plot_resonance_to_ax(soma_hyp_ax, soma_idx, hyp_potential, simfolder)

    plt.savefig(join(figfolder, 'Hu_fig_4_c12861.png'), dpi=150)


def insert_bunch_of_synapses(cell):

    synapseParameters_AMPA = {
        'e': 0,                    #reversal potential
        'syntype': 'Exp2Syn',      #conductance based exponential synapse
        'tau1': 1.,                #Time constant, rise
        'tau2': 3.,                #Time constant, decay
        'weight': 0.005 / 10,           #Synaptic weight
        'color': 'r',              #for plt.plot
        'marker': '.',             #for plt.plot
        'record_current': True,    #record synaptic currents
    }
    # Excitatory synapse parameters
    synapseParameters_NMDA = {
        'e': 0,
        'syntype': 'Exp2Syn',
        'tau1': 10.,
        'tau2': 30.,
        'weight': 0.005 / 10,
        'color': 'm',
        'marker': '.',
        'record_current': True,
    }
    # Inhibitory synapse parameters
    synapseParameters_GABA_A = {
        'e': -80,
        'syntype': 'Exp2Syn',
        'tau1': 1.,
        'tau2': 12.,
        'weight': 0.005 / 10,
        'color': 'b',
        'marker': '.',
        'record_current': True
    }

    # where to insert, how many, and which input statistics
    insert_synapses_AMPA_args = {
        'section': 'apic',
        'n': 100,
        'spTimesFun': LFPy.inputgenerators.stationary_gamma,
        'args': [cell.tstartms, cell.tstopms, 0.5, 80, cell.tstartms]
    }
    insert_synapses_NMDA_args = {
        'section': ['dend', 'apic'],
        'n': 15,
        'spTimesFun': LFPy.inputgenerators.stationary_gamma,
        'args': [cell.tstartms, cell.tstopms, 2, 100, cell.tstartms]
    }
    insert_synapses_GABA_A_args = {
        'section': 'dend',
        'n': 100,
        'spTimesFun': LFPy.inputgenerators.stationary_gamma,
        'args': [cell.tstartms, cell.tstopms, 0.5, 80, cell.tstartms]
    }
    insert_synapses(synapseParameters_AMPA, cell, **insert_synapses_AMPA_args)
    insert_synapses(synapseParameters_NMDA, cell, **insert_synapses_NMDA_args)
    insert_synapses(synapseParameters_GABA_A, cell, **insert_synapses_GABA_A_args)


def savedata(cell, simname, input_idx):

    print "Saving ", simname
    # freqs, imem_psd = find_LFP_power(np.array([cell.imem[input_idx]]), cell.timeres_python/1000.)
    freqs, vmem_psd = find_LFP_power(np.array([cell.vmem[input_idx, :-1]]), cell.timeres_python/1000.)

    #np.save('%s_imem_psd.npy' % simname, imem_psd[0])
    np.save('%s_vmem_psd.npy' % simname, vmem_psd[0])
    #np.save('%s_imem.npy' % simname, cell.imem[input_idx])
    #np.save('%s_vmem.npy' % simname, cell.vmem[input_idx])
    #np.save('%s_tvec.npy' % simname, cell.tvec)
    np.save('%s_freqs.npy' % simname, freqs)

    folder = os.path.dirname(simname)
    if not os.path.isfile(join(folder, 'xmid.npy')):
        np.save(join(folder, 'xmid.npy'), cell.xmid)
        np.save(join(folder, 'ymid.npy'), cell.ymid)
        np.save(join(folder, 'zmid.npy'), cell.zmid)
        np.save(join(folder, 'xend.npy'), cell.xend)
        np.save(join(folder, 'yend.npy'), cell.yend)
        np.save(join(folder, 'zend.npy'), cell.zend)
        np.save(join(folder, 'xstart.npy'), cell.xstart)
        np.save(join(folder, 'ystart.npy'), cell.ystart)
        np.save(join(folder, 'zstart.npy'), cell.zstart)
        np.save(join(folder, 'diam.npy'), cell.diam)


def run_single_test(cell_params, input_array, input_idx, hold_potential, use_channels, sim_idx):

    cell_params.update({'v_init': hold_potential,
                        'custom_fun': [active_declarations],  # will execute this function
                        'custom_fun_args': [{'use_channels': use_channels,
                                             'hold_potential': hold_potential}]
                        })

    neuron.h('forall delete_section()')

    cell = LFPy.Cell(**cell_params)
    noiseVec = neuron.h.Vector(input_array)
    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print "Input i" \
                      "nserted in ", sec.name()
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if type(syn) == type(None):
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noiseVec.play(syn._ref_amp, cell.timeres_NEURON)

    figfolder = 'verifications'
    if not os.path.isdir(figfolder): os.mkdir(figfolder)

    sim_params = {'rec_vmem': True,
                  'rec_imem': True,
                  'rec_variables': []}
    cell.simulate(**sim_params)

    simfolder = 'simresults'
    if not os.path.isdir(simfolder): os.mkdir(simfolder)
    simname = join(simfolder, '%d' % input_idx)

    if len(use_channels) == 0:
        simname += '_passive'
    else:
        for ion in use_channels:
            simname += '_%s' % ion
    simname += '_%+d' % hold_potential
    simname += '_sim_%d' % sim_idx
    savedata(cell, simname, input_idx)
    # recreate_Hu_figs(cell, input_idx, idx_list, cell_params, figfolder)
    plot_resonances(cell, input_idx, 0.01, [0, 460, 565, 451, 728, 303], cell_params, figfolder)
    del cell
    del noiseVec
    del syn


def run_all_sims():

    timeres = 2**-5
    cut_off = 0
    tstopms = 1000
    tstartms = -cut_off
    input_scaling = 0.01
    # input_idxs = [0, 460, 565, 451, 728, 303]

    max_freq = 15

    soma_idx = 0
    apic_idx = 460
    model_path = 'c12861'
    cell_params = {
        'morphology': join(model_path, 'c12861.hoc'),
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': timeres,
        'tstartms': tstartms,          # start time, recorders start at t=0
        'tstopms': tstopms,
    }

    for sim_idx in xrange(0, 1):
        input_array = input_scaling * make_WN_input(cell_params, max_freq)

        run_single_test(cell_params, input_array, soma_idx, -80, ['Ih', 'Im', 'INaP'], sim_idx)
        run_single_test(cell_params, input_array, soma_idx, -80, ['Im', 'INaP'], sim_idx)

        run_single_test(cell_params, input_array, apic_idx, -80, ['Ih', 'Im', 'INaP'], sim_idx)
        run_single_test(cell_params, input_array, apic_idx, -80, ['Im', 'INaP'], sim_idx)

        run_single_test(cell_params, input_array, soma_idx, -60, ['Ih', 'Im', 'INaP'], sim_idx)
        run_single_test(cell_params, input_array, soma_idx, -60, ['Ih', 'INaP'], sim_idx)

        run_single_test(cell_params, input_array, apic_idx, -60, ['Ih', 'Im', 'INaP'], sim_idx)
        run_single_test(cell_params, input_array, apic_idx, -60, ['Ih', 'INaP'], sim_idx)


def simple_test(input_idx, hold_potential, use_channels):

    timeres = 2**-4
    cut_off = 0
    tstopms = 1000
    tstartms = -cut_off
    model_path = 'c12861'

    cell_params = {
        'morphology': join(model_path, 'c12861.hoc'),
        #'rm' : 30000,               # membrane resistance
        #'cm' : 1.0,                 # membrane capacitance
        #'Ra' : 100,                 # axial resistance
        'v_init': hold_potential,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': timeres,
        'tstartms': tstartms,          # start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'use_channels': use_channels,
                             'hold_potential': hold_potential}],
    }
    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params)
    # for sec in cell.allseclist:
    #     for seg in sec:
    #         print sec.name(), seg.diam

    apic_stim_idx = cell.get_idx('apic[66]')[0]

    figfolder = join(model_path, 'verifications')
    if not os.path.isdir(figfolder): os.mkdir(figfolder)

    plt.seed(1234)
    apic_tuft_idx = cell.get_closest_idx(-100, 500, 0)
    trunk_idx = cell.get_closest_idx(0, 300, 0)
    axon_idx = cell.get_idx('axon_IS')[0]
    basal_idx = cell.get_closest_idx(-50, -100, 0)
    soma_idx = 0

    print input_idx, hold_potential
    idx_list = np.array([soma_idx, apic_stim_idx, apic_tuft_idx,
                         trunk_idx, axon_idx, basal_idx])

    print idx_list
    input_scaling = .001

    cell, vec, syn = make_WN_stimuli(cell, cell_params, input_idx, input_scaling, max_freq=15)
    # freqs, psd_sig = find_LFP_power(np.array([vec]), cell.timeres_NEURON/1000.)
    # plt.semilogy(freqs, psd_sig[0,:])
    # plt.show()
    # plot_dynamics()
    # insert_bunch_of_synapses(cell)
    sim_params = {'rec_vmem': True,
                  'rec_imem': True,
                  'rec_variables': []}
    cell.simulate(**sim_params)

    simfolder = join(model_path, 'simresults')
    if not os.path.isdir(simfolder): os.mkdir(simfolder)

    simname = join(simfolder, 'simple_%d_%1.3f' % (input_idx, input_scaling))
    if 'use_channels' in cell_params['custom_fun_args'][0] and \
                    len(cell_params['custom_fun_args'][0]['use_channels']) > 0:
        for ion in cell_params['custom_fun_args'][0]['use_channels']:
            simname += '_%s' % ion
    else:
        simname += '_passive'

    if 'hold_potential' in cell_params['custom_fun_args'][0]:
        simname += '_%+d' % cell_params['custom_fun_args'][0]['hold_potential']

    savedata(cell, simname, input_idx)
    plot_resonances(cell, input_idx, input_scaling, idx_list, cell_params, figfolder)
    # plt.plot(cell.tvec, vec)
    # plt.show()
    #plot_cell_steady_state(cell)
    del cell, vec, syn

if __name__ == '__main__':
    # run_all_sims()
    # recreate_Hu_figs(join('c12861', 'simresults'), '')
    simple_test(0, -60, ['Ih', 'Im'])
    simple_test(0, -80, ['Ih', 'Im'])
    simple_test(573, -60, ['Ih', 'Im'])
    simple_test(573, -80, ['Ih', 'Im'])
    
    simple_test(0, -60, ['Ih'])
    simple_test(0, -80, ['Im'])
    simple_test(573, -60, ['Ih'])
    simple_test(573, -80, ['Im'])