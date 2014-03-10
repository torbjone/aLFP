__author__ = 'torbjone'

import os
from os.path import join
import sys
import numpy as np
import pylab as plt
import LFPy
import neuron
try:
    from ipdb import set_trace
except:
    pass
import aLFP
from matplotlib.colors import LogNorm

def save_synaptic_data(cell, sim_name, cellname, electrode):
    if not os.path.isdir(cellname): os.mkdir(cellname)
    np.save(join(cellname, '%s_tvec.npy' % sim_name), cell.tvec)
    np.save(join(cellname, '%s_sig.npy' % sim_name), electrode.LFP)
    np.save(join(cellname, '%s_vmem.npy' % sim_name), cell.vmem)
    np.save(join(cellname, '%s_imem.npy' % sim_name), cell.imem)
    np.save(join(cellname, '%s_mapping.npy' % sim_name), electrode.electrodecoeff)
    if not os.path.isfile(join(cellname, 'xstart.npy')):
        np.save(join(cellname, 'xstart.npy'), cell.xstart)
        np.save(join(cellname, 'ystart.npy'), cell.ystart)
        np.save(join(cellname, 'zstart.npy'), cell.zstart)
        np.save(join(cellname, 'xend.npy'), cell.xend)
        np.save(join(cellname, 'yend.npy'), cell.yend)
        np.save(join(cellname, 'zend.npy'), cell.zend)
        np.save(join(cellname, 'xmid.npy'), cell.xmid)
        np.save(join(cellname, 'ymid.npy'), cell.ymid)
        np.save(join(cellname, 'zmid.npy'), cell.zmid)
        np.save(join(cellname, 'diam.npy'), cell.diam)


def quick_plot(cell, electrode, sim_name, cell_name,
               num_elecs_x, num_elecs_z, input_idx):

    if hasattr(cell, 'tvec'):
        tvec = cell.tvec
        vmem = cell.vmem
        LFP = 1e3 * electrode.LFP
    else:
        tvec = np.load(join(cell_name, '%s_tvec.npy' % sim_name))
        vmem = np.load(join(cell_name, '%s_vmem.npy' % sim_name))
        LFP = 1e3 * np.load(join(cell_name, '%s_sig.npy' % sim_name))

    plt.close('all')
    fig = plt.figure(figsize=[15, 8])
    fig.suptitle(sim_name)
    fig.subplots_adjust(wspace=0.5)
    ax1 = plt.subplot(141, aspect='equal')
    ax2 = plt.subplot(142, sharey=ax1, sharex=ax1)
    ax3 = plt.subplot(143, sharey=ax1, sharex=ax1)
    ax4 = plt.subplot(144, title='All membrane potentials', ylim=[vmem[0, 0] - 2, vmem[0, 0] + 6])

    ax1.scatter(electrode.x, electrode.z, edgecolor='none', s=1)
    [ax1.plot([cell.xstart[comp], cell.xend[comp]], [cell.zstart[comp], cell.zend[comp]],
              color='k') for comp in xrange(cell.totnsegs)]

    ax2.scatter(electrode.y, electrode.z, edgecolor='none', c='gray', s=1)
    [ax2.plot([cell.ystart[comp], cell.yend[comp]], [cell.zstart[comp], cell.zend[comp]],
              color='k') for comp in xrange(cell.totnsegs)]

    max_t_idxs = np.argmax(np.abs(LFP), axis=1)
    amps = np.array([LFP[elec_idx, max_t_idxs[elec_idx]] for elec_idx in xrange(len(max_t_idxs))])

    vmin = -0.1
    vmax = 0.1
    vlims = np.linspace(vmin, vmax, 20)

    img = ax3.contourf(electrode.x.reshape(num_elecs_z, num_elecs_x),
                      electrode.z.reshape(num_elecs_z, num_elecs_x),
                      amps.reshape(num_elecs_z, num_elecs_x), vmin=vmin, vmax=vmax, levels=vlims)

    ax3.contour(electrode.x.reshape(num_elecs_z, num_elecs_x),
                       electrode.z.reshape(num_elecs_z, num_elecs_x),
                       amps.reshape(num_elecs_z, num_elecs_x),
                       vmin=vmin, vmax=vmax, colors='k', levels=vlims)
    plt.colorbar(img, ax=ax3)

    [ax4.plot(tvec, vmem[idx, :]) for idx in xrange(cell.totnsegs)]
    ax1.plot(cell.xmid[input_idx], cell.zmid[input_idx], '*', color='y', ms=15)
    ax2.plot(cell.ymid[input_idx], cell.zmid[input_idx], '*', color='y', ms=15)
    ax3.plot(cell.xmid[input_idx], cell.zmid[input_idx], '*', color='y', ms=15)
    plt.savefig('%s.png' %sim_name)


def synaptic_reach_simulation(cell_name, cell_params, input_pos,
                              hold_potential, just_plot, **kwargs):

    num_elecs_x = 20
    num_elecs_z = 60

    elec_x, elec_z = np.meshgrid(np.linspace(-200, 200, num_elecs_x),
                                 np.linspace(-200, 1200, num_elecs_z))
    elec_y = np.ones(elec_x.shape) * 100

    electrode_parameters = {
        'sigma': 0.3,      # extracellular conductivity
        'x': elec_x.flatten(),  # electrode requires 1d vector of positions
        'y': elec_y.flatten(),
        'z': elec_z.flatten()
    }

    electrode = LFPy.RecExtElectrode(**electrode_parameters)

    neuron.h('forall delete_section()')
    cell = LFPy.Cell(**cell_params)

    if 0:
        plt.scatter(cell.xmid, cell.zmid)
        plt.scatter(elec_x.flatten(), elec_z.flatten())
        plt.show()

    if input_pos == 'soma':
        input_idx = 0
    elif input_pos == 'apic':
        if cell_name == 'hay':
            input_idx = cell.get_closest_idx(0, 0, 1000)
        else:
            input_idx = cell.get_closest_idx(0, 0, 400)
    else:
        raise RuntimeError("Unknown input position: %s" % input_pos)

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
    if 'conductance_type' in kwargs:
        sim_name = '%s_%d_%+d_%s' % (cell_name, input_idx, hold_potential, kwargs['conductance_type'])
    elif 'use_channels' in kwargs:
        sim_name = '%s_%d_%+d' % (cell_name, input_idx, hold_potential)
        for ion in kwargs['use_channels']:
            sim_name += '_%s' % ion
    else:
        raise RuntimeError("Can't find proper name!")
    if not just_plot:
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)
        save_synaptic_data(cell, sim_name, cell_name, electrode)

    quick_plot(cell, electrode, sim_name, cell_name, num_elecs_x, num_elecs_z, input_idx)