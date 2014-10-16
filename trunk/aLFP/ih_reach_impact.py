__author__ = 'torbjone'

import os
from os.path import join
import sys
import numpy as np
import pylab as plt
import LFPy
import neuron


def save_synaptic_data(cell, sim_name, cell_name, electrode, amps):
    if not os.path.isdir(cell_name): os.mkdir(cell_name)
    # np.save(join(cellname, '%s_tvec.npy' % sim_name), cell.tvec)
    # np.save(join(cellname, '%s_sig.npy' % sim_name), electrode.LFP)
    # np.save(join(cellname, '%s_vmem.npy' % sim_name), cell.vmem)
    # np.save(join(cellname, '%s_imem.npy' % sim_name), cell.imem)
    np.save(join(cell_name, '%s_amps.npy' % sim_name), amps)
    # np.save(join(cellname, '%s_mapping.npy' % sim_name), electrode.electrodecoeff)

    np.save(join(cell_name, 'elec_x.npy'), electrode.x)
    np.save(join(cell_name, 'elec_y.npy'), electrode.y)
    np.save(join(cell_name, 'elec_z.npy'), electrode.z)

    np.save(join(cell_name, 'xstart.npy'), cell.xstart)
    np.save(join(cell_name, 'ystart.npy'), cell.ystart)
    np.save(join(cell_name, 'zstart.npy'), cell.zstart)
    np.save(join(cell_name, 'xend.npy'), cell.xend)
    np.save(join(cell_name, 'yend.npy'), cell.yend)
    np.save(join(cell_name, 'zend.npy'), cell.zend)
    np.save(join(cell_name, 'xmid.npy'), cell.xmid)
    np.save(join(cell_name, 'ymid.npy'), cell.ymid)
    np.save(join(cell_name, 'zmid.npy'), cell.zmid)
    np.save(join(cell_name, 'diam.npy'), cell.diam)


def plot_detectable_volume(cell, electrode, sim_name, cell_name, input_idx,
                           num_elecs_x, num_elecs_y, num_elecs_z, detection_limit, amps):

    if hasattr(cell, 'tvec'):
        LFP = 1e3 * electrode.LFP
    else:
        LFP = 1e3 * np.load(join(cell_name, '%s_sig.npy' % sim_name))

    if amps is None:
        amps = 1e3 * np.load(join(cell_name, '%s_amps.npy' % sim_name))
    else:
        amps *= 1e3

    detected_idxs = np.where(amps > detection_limit)[0]
    x_plane = electrode.x[np.argmin(np.abs(electrode.x))]
    y_plane = electrode.y[np.argmin(np.abs(electrode.y))]
    z_plane = electrode.z[np.argmin(np.abs(electrode.z - 400))]

    x_plane_idxs = np.where((np.abs(electrode.x - x_plane) < 1e-6))[0]
    y_plane_idxs = np.where((np.abs(electrode.y - y_plane) < 1e-6))[0]
    z_plane_idxs = np.where((np.abs(electrode.z - z_plane) < 1e-6))[0]

    detected_x_plane_idxs = np.array(list(set(detected_idxs) & set(x_plane_idxs)))
    detected_y_plane_idxs = np.array(list(set(detected_idxs) & set(y_plane_idxs)))
    detected_z_plane_idxs = np.array(list(set(detected_idxs) & set(z_plane_idxs)))

    detected_x = np.zeros(electrode.x.shape)
    if len(detected_x_plane_idxs) > 0:
        detected_x[detected_x_plane_idxs] = 1
    detected_x = detected_x[x_plane_idxs].reshape(num_elecs_y, num_elecs_z)
    y_x_plane = electrode.y[x_plane_idxs].reshape(num_elecs_y, num_elecs_z)
    z_x_plane = electrode.z[x_plane_idxs].reshape(num_elecs_y, num_elecs_z)

    detected_y = np.zeros(electrode.x.shape)
    if len(detected_y_plane_idxs) > 0:
        detected_y[detected_y_plane_idxs] = 1
    detected_y = detected_y[y_plane_idxs].reshape(num_elecs_x, num_elecs_z)
    x_y_plane = electrode.x[y_plane_idxs].reshape(num_elecs_x, num_elecs_z)
    z_y_plane = electrode.z[y_plane_idxs].reshape(num_elecs_x, num_elecs_z)

    detected_z = np.zeros(electrode.x.shape)
    if len(detected_z_plane_idxs) > 0:
        detected_z[detected_z_plane_idxs] = 1
    detected_z = detected_z[z_plane_idxs].reshape(num_elecs_x, num_elecs_y)
    x_z_plane = electrode.x[z_plane_idxs].reshape(num_elecs_x, num_elecs_y)
    y_z_plane = electrode.y[z_plane_idxs].reshape(num_elecs_x, num_elecs_y)

    fig = plt.figure(figsize=[12, 8])
    fig.suptitle("Signal detectable ( > %1.4f) at %d electrodes" % (detection_limit, len(detected_idxs)))
    ax1 = fig.add_subplot(131, xlabel='x', ylabel='z', aspect='equal')
    ax2 = fig.add_subplot(132, xlabel='y', ylabel='z', aspect='equal')
    ax3 = fig.add_subplot(133, xlabel='x', ylabel='y', aspect='equal')

    ax1.plot([x_plane, x_plane], [np.min(electrode.z), np.max(electrode.z)], 'g')
    ax1.plot([np.min(electrode.x), np.max(electrode.x)], [z_plane, z_plane], 'g')

    ax2.plot([y_plane, y_plane], [np.min(electrode.z), np.max(electrode.z)], 'g')
    ax2.plot([np.min(electrode.y), np.max(electrode.y)], [z_plane, z_plane], 'g')

    ax3.plot([x_plane, x_plane], [np.min(electrode.y), np.max(electrode.y)], 'g')
    ax3.plot([np.min(electrode.x), np.max(electrode.x)], [y_plane, y_plane], 'g')

    ax1.plot(cell.xmid[input_idx], cell.zmid[input_idx], 'y*', zorder=10, ms=10)
    ax2.plot(cell.ymid[input_idx], cell.zmid[input_idx], 'y*', zorder=10, ms=10)
    ax3.plot(cell.xmid[input_idx], cell.ymid[input_idx], 'y*', zorder=10, ms=10)

    [ax1.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], lw=1, color='gray')
            for idx in xrange(cell.totnsegs)]
    [ax2.plot([cell.ystart[idx], cell.yend[idx]], [cell.zstart[idx], cell.zend[idx]], lw=1, color='gray')
        for idx in xrange(cell.totnsegs)]
    [ax3.plot([cell.xstart[idx], cell.xend[idx]], [cell.ystart[idx], cell.yend[idx]], lw=1, color='gray')
        for idx in xrange(cell.totnsegs)]
    colors = ('w', '0.8')

    ax1.contourf(x_y_plane, z_y_plane, detected_y, levels=[-0.1, 0, 1], colors=colors)
    ax2.contourf(y_x_plane, z_x_plane, detected_x, levels=[-0.1, 0, 1], colors=colors)
    ax3.contourf(x_z_plane, y_z_plane, detected_z, levels=[-0.1, 0, 1], colors=colors)

    # ax1.scatter(electrode.x[y_plane_idxs], electrode.z[y_plane_idxs], marker='o',
    #             c='k', edgecolor='none')
    # if not len(detected_y_plane_idxs) == 0:
    #     ax1.scatter(electrode.x[detected_y_plane_idxs], electrode.z[detected_y_plane_idxs],
    #                 marker='o', c='r', edgecolor='none')
    #
    #
    # ax2.scatter(electrode.y[x_plane_idxs], electrode.z[x_plane_idxs],
    #             marker='o', c='k', edgecolor='none')
    # if not len(detected_x_plane_idxs) == 0:
    #     ax2.scatter(electrode.y[detected_x_plane_idxs], electrode.z[detected_x_plane_idxs],
    #                 marker='o', c='r', edgecolor='none')
    #
    # ax3.scatter(electrode.x[z_plane_idxs], electrode.y[z_plane_idxs],
    #             marker='o', c='k', edgecolor='none')
    # if not len(detected_z_plane_idxs) == 0:
    #     ax3.scatter(electrode.x[detected_z_plane_idxs], electrode.y[detected_z_plane_idxs],
    #                 marker='o', c='r', edgecolor='none')
    plt.savefig('detectable_volume_%s.png' % sim_name)


def plot_morph_to_ax(ax, cellname, input_idxs):

    xstart = np.load(join(cellname, 'xstart.npy'))
    ystart = np.load(join(cellname, 'ystart.npy'))
    zstart = np.load(join(cellname, 'zstart.npy'))
    xend = np.load(join(cellname, 'xend.npy'))
    yend = np.load(join(cellname, 'yend.npy'))
    zend = np.load(join(cellname, 'zend.npy'))
    xmid = np.load(join(cellname, 'xmid.npy'))
    ymid = np.load(join(cellname, 'ymid.npy'))
    zmid = np.load(join(cellname, 'zmid.npy'))

    [ax.plot(xmid[idx], zmid[idx], 'y*', zorder=10, ms=10) for idx in input_idxs]
    [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1, color='gray')
            for idx in xrange(len(xmid))]

def hay_detection_values(hold_potentials, soma_idx, conductance_types, apic_idx,
                         cell_name, detection_limit):
    names = []
    soma_values = []
    apic_values = []

    for hold_potential in hold_potentials:
        soma_pot_list = []
        apic_pot_list = []
        for conductance_type in conductance_types:
            soma_name = '%s_%d_%+d_%s' % (cell_name, soma_idx, hold_potential, conductance_type)
            apic_name = '%s_%d_%+d_%s' % (cell_name, apic_idx, hold_potential, conductance_type)
            soma_amps = 1e3 * np.load(join(cell_name, '%s_amps.npy' % soma_name))
            apic_amps = 1e3 * np.load(join(cell_name, '%s_amps.npy' % apic_name))
            num_detection_pts_soma = len(np.where(soma_amps > detection_limit)[0])
            num_detection_pts_apic = len(np.where(apic_amps > detection_limit)[0])
            soma_pot_list.append(num_detection_pts_soma)
            apic_pot_list.append(num_detection_pts_apic)
        names.append(str(hold_potential))
        soma_values.append(soma_pot_list)
        apic_values.append(apic_pot_list)
    return names, soma_values, apic_values


def ca1_detection_values(hold_potentials, soma_idx, conductance_types, apic_idx,
                         cell_name, detection_limit):
    names = []
    soma_values = []
    apic_values = []

    for hold_potential in hold_potentials:
        soma_pot_list = []
        apic_pot_list = []
        for conductance_type in conductance_types:
            cond_name = ''
            for ion in conductance_type:
                cond_name += '_%s' % ion
            soma_name = '%s_%d_%+d%s' % (cell_name, soma_idx, hold_potential, cond_name)
            apic_name = '%s_%d_%+d%s' % (cell_name, apic_idx, hold_potential, cond_name)
            soma_amps = 1e3 * np.load(join(cell_name, '%s_amps.npy' % soma_name))
            apic_amps = 1e3 * np.load(join(cell_name, '%s_amps.npy' % apic_name))
            num_detection_pts_soma = len(np.where(soma_amps > detection_limit)[0])
            num_detection_pts_apic = len(np.where(apic_amps > detection_limit)[0])
            soma_pot_list.append(num_detection_pts_soma)
            apic_pot_list.append(num_detection_pts_apic)
        names.append(str(hold_potential))
        soma_values.append(soma_pot_list)
        apic_values.append(apic_pot_list)
    return names, soma_values, apic_values

def compare_detectable_volumes(detection_limit=0.005):

    cell_name = 'c12861'
    if cell_name == 'hay':
        conductance_types = ['active', 'Ih_linearized', 'passive']
        max_elecs = 1400
    elif cell_name == 'c12861' or cell_name == 'n120':
        conductance_types = [['Ih', 'Im', 'INaP'], ['Ih', 'INaP'], ['Im', 'INaP']]
        max_elecs = 1400
    else:
        raise RuntimeError("Unrecognized cell name")

    soma_idx = 0
    apic_idx = {'hay': 852, 'c12861': 997, 'n120': 747}[cell_name]

    fig = plt.figure(figsize=[8, 4])
    ax1 = fig.add_axes([0.05, 0.1, 0.2, 0.8], frameon=False, xticks=[], yticks=[])
    ax2 = fig.add_axes([0.3, 0.6, 0.6, 0.3], title='Apic synaptic input',
                       ylabel='Detection volume', ylim=[0, max_elecs])
    ax3 = fig.add_axes([0.3, 0.1, 0.6, 0.3], title='Somatic synaptic input',
                       ylabel='Detection volume', ylim=[0, max_elecs])
    hold_potentials = [-80, -70, -60]
    plot_morph_to_ax(ax1, cell_name, (soma_idx, apic_idx))
    dim = len(hold_potentials)
    w = 0.75
    dimw = w / dim

    if cell_name == 'hay':
        names, soma_values, apic_values = hay_detection_values(hold_potentials, soma_idx, conductance_types,
                                                               apic_idx, cell_name, detection_limit)
    else:
        names, soma_values, apic_values = ca1_detection_values(hold_potentials, soma_idx, conductance_types,
                                                               apic_idx, cell_name, detection_limit)

    x = np.arange(dim)
    for i in range(len(conductance_types)):
        y_soma = [d[i] for d in soma_values]
        y_apic = [d[i] for d in apic_values]
        ax3.bar(x + i * dimw, y_soma, dimw, color='rgb'[i])
        ax2.bar(x + i * dimw, y_apic, dimw, color='rgb'[i])

    for ax in [ax2, ax3]:
        ax.set_xticks(x + 3 * dimw / 2)
        ax.set_xticklabels(['%s mV' % name for name in names])

    bars = []
    labels = []
    for i, conductance_type in enumerate(conductance_types):
        b = plt.bar(0, 0, 0, color='rgb'[i])
        bars.append(b)
        labels.append(conductance_type)
    fig.legend(bars, labels, frameon=False, ncol=3)
    plt.savefig('summary_detected_volume_%s.png' % cell_name)


def quick_plot(cell, electrode, sim_name, cell_name,
               num_elecs_x, num_elecs_y, num_elecs_z, input_idx):

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
    plt.savefig('%s.png' % sim_name)


def synaptic_reach_simulation(cell_name, cell_params, input_pos,
                              hold_potential, just_plot, **kwargs):

    num_elecs_x = 20
    num_elecs_y = 20
    num_elecs_z = 30

    grid_x = np.linspace(-500, 500, num_elecs_x)
    grid_y = np.linspace(-500, 500, num_elecs_y)
    grid_z = np.linspace(-500, 1600, num_elecs_z)
    elec_x, elec_y, elec_z = np.meshgrid(grid_x, grid_y, grid_z)

    single_elec_box_volume = ((grid_x[1]-grid_x[0]) * (grid_y[1]-grid_y[0]) *
                              (grid_z[1]-grid_z[0]))

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
        plt.subplot(121, xlabel='x', ylabel='z')
        plt.scatter(cell.xmid, cell.zmid)
        plt.scatter(elec_x.flatten(), elec_z.flatten())
        plt.axis('equal')

        plt.subplot(122, xlabel='y', ylabel='z')
        plt.scatter(cell.ymid, cell.zmid)
        plt.scatter(elec_y.flatten(), elec_z.flatten())
        plt.axis('equal')
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

    detection_limit = 0.005

    amps = None
    if not just_plot:
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)
        max_t_idxs = np.argmax(np.abs(electrode.LFP), axis=1)
        amps = np.array([np.abs(electrode.LFP[elec_idx, max_t_idxs[elec_idx]])
                              for elec_idx in xrange(len(max_t_idxs))])

        save_synaptic_data(cell, sim_name, cell_name, electrode, amps)

    plot_detectable_volume(cell, electrode, sim_name, cell_name, input_idx,
                           num_elecs_x, num_elecs_y, num_elecs_z, detection_limit, amps)