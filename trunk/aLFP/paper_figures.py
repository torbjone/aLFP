__author__ = 'torbjone'

import sys
import os
import pylab as plt
import numpy as np
from os.path import join
from plotting_convention import *

class PaperFigures():

    conductance_clr = {'active': 'r', 'Ih_linearized': 'g', 'passive': 'b'}
    def __init__(self):
        self.figure_folder = join('/home', 'torbjone', 'work', 'aLFP', 'paper_figures')
        self.sim_folder = join('/home', 'torbjone', 'work', 'aLFP', 'paper_simulations')
        self.root_folder = join('/home', 'torbjone', 'work', 'aLFP')

    @staticmethod
    def _draw_morph_to_axis(ax, folder, input_pos):
        xstart = np.load(join(folder, 'xstart.npy'))
        zstart = np.load(join(folder, 'zstart.npy'))
        xend = np.load(join(folder, 'xend.npy'))
        zend = np.load(join(folder, 'zend.npy'))
        xmid = np.load(join(folder, 'xmid.npy'))
        zmid = np.load(join(folder, 'zmid.npy'))
        if type(input_pos) is int:
            ax.plot(xmid[input_pos], zmid[input_pos], 'y*', zorder=1, ms=10)

        [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1, color='gray', zorder=0)
         for idx in xrange(len(xmid))]

    @staticmethod
    def save_synaptic_data(cell, sim_folder, electrode, input_idx,
                           conductance_type, holding_potential):

        if not os.path.isdir(sim_folder): os.mkdir(sim_folder)
        sim_name = '%d_%s_%+d' % (input_idx, conductance_type, holding_potential)
        np.save(join(sim_folder, 'tvec.npy'), cell.tvec)
        np.save(join(sim_folder, 'sig_%s.npy' % sim_name), electrode.LFP)
        np.save(join(sim_folder, 'vmem_%s.npy' % sim_name), cell.vmem)
        np.save(join(sim_folder, 'imem_%s.npy' % sim_name), cell.imem)

        np.save(join(sim_folder, 'elec_x.npy'), electrode.x)
        np.save(join(sim_folder, 'elec_y.npy'), electrode.y)
        np.save(join(sim_folder, 'elec_z.npy'), electrode.z)

        np.save(join(sim_folder, 'xstart.npy'), cell.xstart)
        np.save(join(sim_folder, 'ystart.npy'), cell.ystart)
        np.save(join(sim_folder, 'zstart.npy'), cell.zstart)
        np.save(join(sim_folder, 'xend.npy'), cell.xend)
        np.save(join(sim_folder, 'yend.npy'), cell.yend)
        np.save(join(sim_folder, 'zend.npy'), cell.zend)
        np.save(join(sim_folder, 'xmid.npy'), cell.xmid)
        np.save(join(sim_folder, 'ymid.npy'), cell.ymid)
        np.save(join(sim_folder, 'zmid.npy'), cell.zmid)
        np.save(join(sim_folder, 'diam.npy'), cell.diam)

    def _do_single_fig_1_simulation(self, conductance_type, holding_potential,
                                    input_idx, sim_folder, plot_positions):
        import neuron
        from hay_active_declarations import active_declarations as hay_active
        import LFPy

        np.random.seed(1234)
        electrode_parameters = {
            'sigma': 0.3,
            'x': np.array(plot_positions)[:, 0].flatten(),
            'y': np.array(plot_positions)[:, 1].flatten(),
            'z': np.array(plot_positions)[:, 2].flatten()
        }
        electrode = LFPy.RecExtElectrode(**electrode_parameters)
        neuron.h('forall delete_section()')
        neuron_models = join(self.root_folder, 'neuron_models')
        timeres = 2**-4
        tstopms = 100
        neuron.load_mechanisms(join(neuron_models, 'hay', 'mod'))
        cell_params = {
            'morphology': join(neuron_models, 'hay', 'lfpy_version', 'morphologies', 'cell1.hoc'),
            'v_init': holding_potential,             # initial crossmembrane potential
            'passive': False,           # switch on passive mechs
            'nsegs_method': 'lambda_f',  # method for setting number of segments,
            'lambda_f': 100,           # segments are isopotential at this frequency
            'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
            'timeres_python': timeres,
            'tstartms': 0,          # start time, recorders start at t=0
            'tstopms': tstopms,
            'custom_code': [join(neuron_models, 'hay', 'lfpy_version', 'custom_codes.hoc')],
            'custom_fun': [hay_active],  # will execute this function
            'custom_fun_args': [{'conductance_type': conductance_type,
                                 'hold_potential': holding_potential}]
        }

        cell = LFPy.Cell(**cell_params)

        if 0:
            plt.subplot(121, xlabel='x', ylabel='z')
            plt.scatter(cell.xmid, cell.zmid)
            plt.scatter(electrode.x, electrode.z)
            plt.axis('equal')

            plt.subplot(122, xlabel='y', ylabel='z')
            plt.scatter(cell.ymid, cell.zmid)
            plt.scatter(electrode.y, electrode.z)
            plt.axis('equal')
            plt.show()

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
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)
        self.save_synaptic_data(cell, sim_folder, electrode, input_idx,
                                conductance_type, holding_potential)
        del synapse, cell

    def _do_all_fig_1_simulations(self, holding_potentials, plot_positions, sim_folder,
                                  conductance_types, soma_idx, apic_idx):
        for holding_potential in holding_potentials:
            for input_idx in [apic_idx, soma_idx]:
                for conductance_type in conductance_types:
                    self._do_single_fig_1_simulation(conductance_type, holding_potential,
                                                 input_idx, sim_folder, plot_positions)
    @staticmethod
    def _find_closest_idx_to_pos(pos, elec_x, elec_y, elec_z):
        return np.argmin(np.sum(np.array([(elec_x - pos[0])**2, (elec_y - pos[1])**2, (elec_z - pos[2])**2]), axis=0))

    @classmethod
    def _return_idxs_from_positions(cls, position_list, elec_x, elec_y, elec_z):
        return [cls._find_closest_idx_to_pos(pos, elec_x, elec_y, elec_z) for pos in position_list]

    @classmethod
    def _plot_EC_signal_to_ax(cls, ax, idx, signal, elec_x_pos, elec_z_pos, scale_factor_EP,
                              scale_factor_time, tvec, conductance_type):
        x = elec_x_pos + (tvec * scale_factor_time)
        y = elec_z_pos + (signal[idx] * scale_factor_EP)
        ax.plot(x, y, color=cls.conductance_clr[conductance_type], lw=2)

    def make_fig_1(self, do_simulations=False):

        conductance_types = ['passive', 'active']

        # plot_positions = ((0, 0, -170), (-170, 0, 0), (100, 0, -30),
        #                                 (50, 0, 300), (-100, 0, 400), (70, 0, 600),
        #                                 (-80, 0, 900), (160, 0, 1000))

        elec_x, elec_z = np.meshgrid(np.linspace(-200, 200, 7),
                                     np.linspace(-200, 1200, 15))
        elec_x = elec_x.flatten()
        elec_z = elec_z.flatten()
        elec_y = np.zeros(len(elec_x))
        plot_positions = np.array([elec_x, elec_y, elec_z]).T

        cell_dict = {'holding_potentials': [-80, -70, -60],
                     'soma_idx': 0,
                     'apic_idx': 852,
                     'plot_positions': plot_positions
        }

        sim_folder = join(self.root_folder, 'paper_simulations', 'intro_fig')
        if do_simulations:
            self._do_all_fig_1_simulations(sim_folder=sim_folder, conductance_types=conductance_types,
                                           **cell_dict)
        tvec = np.load(join(sim_folder, 'tvec.npy'))
        elec_x = np.load(join(sim_folder, 'elec_x.npy'))
        elec_y = np.load(join(sim_folder, 'elec_y.npy'))
        elec_z = np.load(join(sim_folder, 'elec_z.npy'))

        start_t = 0
        end_t = 80
        start_idx = np.argmin(np.abs(tvec - start_t))
        end_idx = np.argmin(np.abs(tvec - end_t))

        tvec = tvec[start_idx:end_idx]
        plt.close('all')
        fig = plt.figure(figsize=[5, 10])
        fig.subplots_adjust(hspace=0.0, top=0.95, bottom=0.)
        ax1 = plt.subplot(321, ylim=[-150, 1000], xlim=[-200, 200])
        ax2 = plt.subplot(322, sharey=ax1, sharex=ax1)
        ax3 = plt.subplot(323, sharey=ax1, sharex=ax1)
        ax4 = plt.subplot(324, sharey=ax1, sharex=ax1)
        ax5 = plt.subplot(325, sharey=ax1, sharex=ax1)
        ax6 = plt.subplot(326, sharey=ax1, sharex=ax1)


        idx_list = self._return_idxs_from_positions(cell_dict['plot_positions'], elec_x, elec_y, elec_z)

        scale_factor_length = 500.
        bar_length = 0.1
        scale_factor_EP = scale_factor_length / bar_length  # 0.1 uV should be 200 um on figure
        scale_factor_time = 100. / (end_t - start_t)

        ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]

        ax1.plot([300, 300], [-500, -500 - scale_factor_length], 'k', lw=3)
        ax1.text(310, -500 - scale_factor_length/2, '0.1 $\mu$V',
                 verticalalignment='center',
                 horizontalalignment='left')
        # ax1.text(190, -300, '200 $\mu$m', verticalalignment='center', horizontalalignment='right')
        ax_numb = 0
        for holding_potential in cell_dict['holding_potentials']:
            for input_idx in [cell_dict['soma_idx'], cell_dict['apic_idx']]:
                for conductance_type in conductance_types:
                    name = 'Soma' if input_idx == 0 else 'Apic'
                    ax_list[ax_numb].set_title('%s %d mV' % (name, holding_potential))
                    self.plot_one_signal(input_idx, conductance_type, holding_potential, start_idx, end_idx, idx_list,
                                         sim_folder, ax_list[ax_numb], elec_x, elec_z, scale_factor_EP, scale_factor_time, tvec)
                ax_numb += 1

        [self._draw_morph_to_axis(ax, sim_folder, cell_dict['soma_idx']) for ax in [ax1, ax3, ax5]]
        [self._draw_morph_to_axis(ax, sim_folder, cell_dict['apic_idx']) for ax in [ax2, ax4, ax6]]

        # ax1.scatter(elec_x[idx_list], elec_z[idx_list], edgecolor='none', c='k', s=20)
        # ax2.scatter(elec_x[idx_list], elec_z[idx_list], edgecolor='none', c='k', s=20)
        lines = []
        line_names = []
        for conductance_type in conductance_types:
            l, = plt.plot(0, 0, color=self.conductance_clr[conductance_type], lw=2)
            lines.append(l)
            line_names.append(conductance_type)

        mark_subplots(ax_list, xpos=0.12, ypos=0.9)
        [ax.axis('off') for ax in ax_list]
        fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=2)
        fig.savefig(join(self.figure_folder, 'figure_1.png'), dpi=150)

    def plot_one_signal(self, input_idx, conductance_type, holding_potential, start_idx, end_idx,
                        idx_list, sim_folder, ax, elec_x, elec_z, scale_factor_EP, scale_factor_time, tvec):
        LFP = 1e3*np.load(join(sim_folder, 'sig_%d_%s_%+d.npy' % (input_idx, conductance_type,
                                                                  holding_potential)))[:, start_idx:end_idx]
        for idx in idx_list[::5]:
            self._plot_EC_signal_to_ax(ax, idx, LFP, elec_x[idx], elec_z[idx],
                                       scale_factor_EP, scale_factor_time, tvec, conductance_type)
if __name__ == '__main__':
    # TODO: MAKE FIGURE WITH MORE HOLDING POTENTIALS
    PaperFigures().make_fig_1(do_simulations=False)