__author__ = 'torbjone'

import sys
import os
import numpy as np
from os.path import join
from plotting_convention import *
import scipy.fftpack as ff
import aLFP
from matplotlib.collections import LineCollection

class PaperFigures():

    def __init__(self):
        np.random.seed(1234)
        self.conductance_clr = {'active': 'r', 'active_frozen': 'b', 'Ih_linearized': 'g', 'passive': 'k',
                                'Ih_linearized_frozen': 'c'}
        self.figure_folder = join('/home', 'torbjone', 'work', 'aLFP', 'paper_figures')
        self.root_folder = join('/home', 'torbjone', 'work', 'aLFP')

    def _draw_morph_to_axis(self, ax, input_pos):
        xstart = np.load(join(self.sim_folder, 'xstart_%s.npy' % self.cell_name))
        zstart = np.load(join(self.sim_folder, 'zstart_%s.npy' % self.cell_name))
        xend = np.load(join(self.sim_folder, 'xend_%s.npy' % self.cell_name))
        zend = np.load(join(self.sim_folder, 'zend_%s.npy' % self.cell_name))
        xmid = np.load(join(self.sim_folder, 'xmid_%s.npy' % self.cell_name))
        zmid = np.load(join(self.sim_folder, 'zmid_%s.npy' % self.cell_name))
        if type(input_pos) is int:
            ax.plot(xmid[input_pos], zmid[input_pos], 'y*', zorder=1, ms=10)

        [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1, color='0.5', zorder=0, alpha=1)
         for idx in xrange(len(xmid))]

    def _draw_simplified_morph_to_axis(self, ax, input_pos=None, grading=None):

        xstart = [-150, 150, 0, 0, 0]
        xend = [0, 0, 0, -100, 100]
        zstart = [-150, -150, 0, 1000, 1000]
        zend = [0, 0, 1000, 1100, 1100]
        lineprops = {'lw': 5, 'color': 'gray'}

        color_at_pos = lambda dist: plt.cm.gray(int(256./1500 * dist))
        if grading is None:
            [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], **lineprops) for idx in xrange(len(xstart))]
            ax.plot(0, 0, '^', c=lineprops['color'], ms=20, mec=lineprops['color'])
        else:
            num_interpts = 10
            x = np.array([np.linspace(xstart[i], xend[i], num_interpts)
                              for i in xrange(len(xstart))]).flatten()
            z = np.array([np.linspace(zstart[i], zend[i], num_interpts)
                              for i in xrange(len(zend))]).flatten()
            dist = np.sqrt(x**2 + z**2)
            points = np.array([x, z]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=plt.get_cmap('gray'),
                        norm=plt.Normalize(0, 1500), zorder=0)
            if grading == 'uniform':
                lc.set_array(np.ones(len(z)) * 500)
                soma_clr = color_at_pos(500)
            elif grading == 'linear_increase':
                lc.set_array(dist)
                soma_clr = color_at_pos(0)
            elif grading == 'linear_decrease':
                lc.set_array(np.max(dist) - dist)
                soma_clr = color_at_pos(np.max(dist))
            else:
                raise RuntimeError("Unknown grading!")
            lc.set_linewidth(3)
            ax.plot(0, 0, '^', c=soma_clr, ms=10, mec=soma_clr)
            ax.add_collection(lc)
        if not input_pos is None:
            synapse_clr = 'g'
            if input_pos is 'soma':
                synapse_coor = [-90, 20]
                axon_start = [-175, 50]
                marker = (3, 0, 55)
            else:
                synapse_coor = [-100, 1010]
                axon_start = [-200, 900]
                marker = (3, 0, 15)

            ax.plot([axon_start[0], synapse_coor[0]], [axon_start[1], synapse_coor[1]], lw=5, c=synapse_clr)
            ax.plot(synapse_coor[0], synapse_coor[1], c=synapse_clr, mec=synapse_clr, ms=15, marker=marker)
        ax.axis('equal')

    @staticmethod
    def _find_closest_idx_to_pos(pos, elec_x, elec_y, elec_z):
        return np.argmin(np.sum(np.array([(elec_x - pos[0])**2, (elec_y - pos[1])**2, (elec_z - pos[2])**2]), axis=0))

    @classmethod
    def _return_idxs_from_positions(cls, position_list, elec_x, elec_y, elec_z):
        return [cls._find_closest_idx_to_pos(pos, elec_x, elec_y, elec_z) for pos in position_list]

class IntroFigures():
    np.random.seed(1234)
    conductance_clr = {'active': 'r', 'active_frozen': 'b', 'Ih_linearized': 'g', 'passive': 'k',
                       'Ih_linearized_frozen': 'c'}
    def __init__(self, cell_name, figure_name):
        self.cell_name = cell_name
        self.figure_name = figure_name
        self.figure_folder = join('/home', 'torbjone', 'work', 'aLFP', 'paper_figures')
        self.root_folder = join('/home', 'torbjone', 'work', 'aLFP')
        self.timeres_NEURON = 2**-5
        self.timeres_python = 2**-3
        self.holding_potentials = [-80, -60]
        self._set_cell_specific_properties()
        self._set_figure_specific_properties()

    def _set_figure_specific_properties(self):

        if self.figure_name == 'figure_1':
            self.start_t = 0
            self.end_t = 30
            self.cut_off = 0
            self.sim_folder = join(self.root_folder, 'paper_simulations', 'intro_fig_synapse')
            self.stimuli_function = self._make_syaptic_stimuli
            self.ec_ax_dict = {'frameon': False,
                               'xticks': [],
                               'xticklabels': [],
                               'yticks': [],
                               'yticklabels': [],
                               'ylim': [0, 0.005],
                               'xlim': [0, 30]}
            self.clip_on = False
        elif self.figure_name == 'figure_2':
            self.start_t = 0
            self.end_t = 1000
            self.cut_off = 6000
            self.sim_folder = join(self.root_folder, 'paper_simulations', 'intro_fig_white_noise')
            self.stimuli_function = self._make_white_noise_stimuli
            self.ec_ax_dict = {'frameon': True,
                               'xticks': [1e0, 1e1, 1e2],
                               'xticklabels': [],
                               'yticks': [1e-6, 1e-5, 1e-4, 1e-3],
                               'yticklabels': [], 'xscale': 'log', 'yscale': 'log',
                               'ylim': [1e-5, 1e-3],
                               'xlim': [1, 500]}
            self.clip_on = True
        else:
            raise RuntimeError("Unrecognized figure name: %s" % self.figure_name)

    def _set_cell_specific_properties(self):
        if self.cell_name == 'hay':
            elec_x, elec_z = np.meshgrid(np.linspace(-200, 200, 7),
                                         np.linspace(-200, 1200, 15))
            elec_x = elec_x.flatten()
            elec_z = elec_z.flatten()
            elec_y = np.zeros(len(elec_x))
            self.plot_positions = np.array([elec_x, elec_y, elec_z]).T
            self.conductance_types = ['active', 'passive']
            self.soma_idx = 0
            self.apic_idx = 852
            self.use_elec_idxs = [8, 36, 26, 67, 85]
            self.ax_dict = {'ylim': [-300, 1300], 'xlim': [-400, 400]}

        elif self.cell_name == 'n120':

            elec_x, elec_z = np.meshgrid(np.linspace(-150, 150, 7),
                                         np.linspace(-150, 800, 15))
            elec_x = elec_x.flatten()
            elec_z = elec_z.flatten()
            elec_y = np.zeros(len(elec_x))
            self.plot_positions = np.array([elec_x, elec_y, elec_z]).T
            self.conductance_types = ['active', 'passive']
            self.soma_idx = 0
            self.apic_idx = 685
            self.use_elec_idxs = [33, 2, 78, 61, 22]
            self.ax_dict = {'ylim': [-200, 700], 'xlim': [-250, 250]}

        elif self.cell_name == 'c12861':
            elec_x, elec_z = np.meshgrid(np.linspace(-150, 150, 7),
                                         np.linspace(-150, 800, 15))
            elec_x = elec_x.flatten()
            elec_z = elec_z.flatten()
            elec_y = np.zeros(len(elec_x))
            self.plot_positions = np.array([elec_x, elec_y, elec_z]).T
            self.conductance_types = ['active', 'passive']
            self.soma_idx = 0
            self.apic_idx = 1128
            self.use_elec_idxs = [26, 2, 77, 68, 28]#[19, 8, 23, 71, 52]
            self.ax_dict = {'ylim': [-200, 750], 'xlim': [-300, 300]}
        else:
            raise RuntimeError("Unknown cell name: %s" % self.cell_name)


    def _draw_morph_to_axis(self, ax, input_pos):
        xstart = np.load(join(self.sim_folder, 'xstart_%s.npy' % self.cell_name))
        zstart = np.load(join(self.sim_folder, 'zstart_%s.npy' % self.cell_name))
        xend = np.load(join(self.sim_folder, 'xend_%s.npy' % self.cell_name))
        zend = np.load(join(self.sim_folder, 'zend_%s.npy' % self.cell_name))
        xmid = np.load(join(self.sim_folder, 'xmid_%s.npy' % self.cell_name))
        zmid = np.load(join(self.sim_folder, 'zmid_%s.npy' % self.cell_name))
        if type(input_pos) is int:
            ax.plot(xmid[input_pos], zmid[input_pos], 'y*', zorder=1, ms=10)

        [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1, color='0.8', zorder=0, alpha=1)
         for idx in xrange(len(xmid))]

    def save_neural_sim_data(self, cell, electrode, input_idx,
                             conductance_type, holding_potential):

        if not os.path.isdir(self.sim_folder): os.mkdir(self.sim_folder)
        sim_name = '%s_%d_%s_%+d' % (self.cell_name, input_idx, conductance_type, holding_potential)
        np.save(join(self.sim_folder, 'tvec_%s.npy' % self.cell_name), cell.tvec)
        # np.save(join(self.sim_folder, 'sig_%s.npy' % sim_name), electrode.LFP)
        np.save(join(self.sim_folder, 'sig_%s.npy' % sim_name), np.dot(electrode.electrodecoeff, cell.imem))
        np.save(join(self.sim_folder, 'vmem_%s.npy' % sim_name), cell.vmem)
        np.save(join(self.sim_folder, 'imem_%s.npy' % sim_name), cell.imem)

        np.save(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name), electrode.x)
        np.save(join(self.sim_folder, 'elec_y_%s.npy' % self.cell_name), electrode.y)
        np.save(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name), electrode.z)

        np.save(join(self.sim_folder, 'xstart_%s.npy' % self.cell_name), cell.xstart)
        np.save(join(self.sim_folder, 'ystart_%s.npy' % self.cell_name), cell.ystart)
        np.save(join(self.sim_folder, 'zstart_%s.npy' % self.cell_name), cell.zstart)
        np.save(join(self.sim_folder, 'xend_%s.npy' % self.cell_name), cell.xend)
        np.save(join(self.sim_folder, 'yend_%s.npy' % self.cell_name), cell.yend)
        np.save(join(self.sim_folder, 'zend_%s.npy' % self.cell_name), cell.zend)
        np.save(join(self.sim_folder, 'xmid_%s.npy' % self.cell_name), cell.xmid)
        np.save(join(self.sim_folder, 'ymid_%s.npy' % self.cell_name), cell.ymid)
        np.save(join(self.sim_folder, 'zmid_%s.npy' % self.cell_name), cell.zmid)
        np.save(join(self.sim_folder, 'diam_%s.npy' % self.cell_name), cell.diam)

    def quickplot_exp_setup(self, cell, electrode):
        plt.subplot(121, xlabel='x', ylabel='z')
        plt.scatter(cell.xmid, cell.zmid)
        plt.scatter(electrode.x, electrode.z)
        plt.axis('equal')

        plt.subplot(122, xlabel='y', ylabel='z')
        plt.scatter(cell.ymid, cell.zmid)
        plt.scatter(electrode.y, electrode.z)
        plt.axis('equal')
        plt.show()

    def _return_cell(self, holding_potential, conductance_type):
        import neuron
        from hay_active_declarations import active_declarations as hay_active
        from ca1_sub_declarations import active_declarations as ca1_active
        import LFPy
        neuron_models = join(self.root_folder, 'neuron_models')
        neuron.load_mechanisms(join(neuron_models))

        if self.cell_name == 'hay':
            neuron.load_mechanisms(join(neuron_models, 'hay', 'mod'))
            cell_params = {
                'morphology': join(neuron_models, 'hay', 'lfpy_version', 'morphologies', 'cell1.hoc'),
                'v_init': holding_potential,
                'passive': False,           # switch on passive mechs
                'nsegs_method': 'lambda_f',  # method for setting number of segments,
                'lambda_f': 100,           # segments are isopotential at this frequency
                'timeres_NEURON': self.timeres_NEURON,   # dt of LFP and NEURON simulation.
                'timeres_python': self.timeres_python,
                'tstartms': -self.cut_off,          # start time, recorders start at t=0
                'tstopms': self.end_t,
                'custom_code': [join(neuron_models, 'hay', 'lfpy_version', 'custom_codes.hoc')],
                'custom_fun': [hay_active],  # will execute this function
                'custom_fun_args': [{'conductance_type': conductance_type,
                                     'hold_potential': holding_potential}]
            }
        elif self.cell_name in ['n120', 'c12861']:
            if conductance_type == 'active':
                use_channels = ['Ih', 'Im', 'INaP']
            elif conductance_type == 'passive':
                use_channels = []
            elif conductance_type == 'active_frozen':
                use_channels = ['Ih_frozen', 'Im_frozen', 'INaP_frozen']
            else:
                raise RuntimeError("Unrecognized conductance_type for %s: %s" % (self.cell_name, conductance_type))
            neuron.load_mechanisms(join(neuron_models, 'ca1_sub'))
            cell_params = {
                    'morphology': join(neuron_models, 'ca1_sub', self.cell_name, '%s.hoc' % self.cell_name),
                    'v_init': holding_potential,             # initial crossmembrane potential
                    'passive': False,           # switch on passive mechs
                    'nsegs_method': 'lambda_f',  # method for setting number of segments,
                    'lambda_f': 100,           # segments are isopotential at this frequency
                    'timeres_NEURON': self.timeres_NEURON,   # dt of LFP and NEURON simulation.
                    'timeres_python': self.timeres_python,
                    'tstartms': -self.cut_off,          # start time, recorders start at t=0
                    'tstopms': self.end_t,
                    'custom_fun': [ca1_active],  # will execute this function
                    'custom_fun_args': [{'use_channels': use_channels,
                                         'cellname': self.cell_name,
                                         'hold_potential': holding_potential}],
                    }

        else:
            raise RuntimeError("Unrecognized cell name: %s" % self.cell_name)

        cell = LFPy.Cell(**cell_params)
        return cell

    def _do_single_neural_simulation(self, conductance_type, holding_potential, input_idx):
        import neuron
        import LFPy

        electrode_parameters = {
            'sigma': 0.3,
            'x': np.array(self.plot_positions)[:, 0].flatten(),
            'y': np.array(self.plot_positions)[:, 1].flatten(),
            'z': np.array(self.plot_positions)[:, 2].flatten()
        }
        electrode = LFPy.RecExtElectrode(**electrode_parameters)
        neuron.h('forall delete_section()')
        cell = self._return_cell(holding_potential, conductance_type)
        if 0:
            [plt.plot([cell.xstart[i], cell.xend[i]], [cell.zstart[i], cell.zend[i]], 'k') for i in xrange(cell.totnsegs)]
            [plt.text(cell.xmid[i], cell.zend[i], '%1.2f' % i, color='r') for i in xrange(cell.totnsegs)]

            plt.axis('equal')
            plt.show()

        plt.seed(1234)
        cell, syn, noiseVec = self.stimuli_function(cell, input_idx)
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)

        self.save_neural_sim_data(cell, electrode, input_idx, conductance_type, holding_potential)

    def make_WN_input(self, cell, max_freq):
        """ White Noise input ala Linden 2010 is made """
        tot_ntsteps = round((cell.tstopms - cell.tstartms)/\
                      cell.timeres_NEURON + 1)
        I = np.zeros(tot_ntsteps)
        tvec = np.arange(tot_ntsteps) * cell.timeres_NEURON
        for freq in xrange(1, max_freq + 1):
            I += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())
        return I

    def _make_white_noise_stimuli(self, cell, input_idx):
        input_scaling = 0.001
        max_freq = 500
        import neuron
        input_array = input_scaling * self.make_WN_input(cell, max_freq)
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
        if syn is None:
            raise RuntimeError("Wrong stimuli index")
        syn.dur = 1E9
        syn.delay = 0
        noiseVec.play(syn._ref_amp, cell.timeres_NEURON)
        return cell, syn, noiseVec

    def _make_syaptic_stimuli(self, cell, input_idx):
        import LFPy
        # Define synapse parameters
        synapse_parameters = {
            'idx': input_idx,
            'e': 0.,                   # reversal potential
            'syntype': 'ExpSyn',       # synapse type
            'tau': 2.,                # syn. time constant
            'weight': 0.001,            # syn. weight
            'record_current': True,
        }

        synapse = LFPy.Synapse(cell, **synapse_parameters)
        synapse.set_spike_times(np.array([5.]))
        return cell, synapse, None

    def _do_all_simulations(self):
        for holding_potential in self.holding_potentials:
            for input_idx in [self.apic_idx, self.soma_idx]:
                for conductance_type in self.conductance_types:
                    self._do_single_neural_simulation(conductance_type, holding_potential, input_idx)

    @staticmethod
    def _find_closest_idx_to_pos(pos, elec_x, elec_y, elec_z):
        return np.argmin(np.sum(np.array([(elec_x - pos[0])**2, (elec_y - pos[1])**2, (elec_z - pos[2])**2]), axis=0))

    @classmethod
    def _return_idxs_from_positions(cls, position_list, elec_x, elec_y, elec_z):
        return [cls._find_closest_idx_to_pos(pos, elec_x, elec_y, elec_z) for pos in position_list]

    def _plot_EC_signal_to_ax(self, fig, ax, idx, signal, elec_x_pos, elec_z_pos, x_vec, conductance_type):

        ec_plot_dict = {'color': self.conductance_clr[conductance_type],
                        'lw': 1,
                        'clip_on': self.clip_on}
        ax_ = fig.add_axes(self.return_ax_coors(fig, ax, (elec_x_pos, elec_z_pos)), **self.ec_ax_dict)

        ax_.minorticks_off()
        simplify_axes(ax_)
        ax_.patch.set_alpha(0.0)
        ax_.plot(x_vec, signal[idx, :], **ec_plot_dict)
        if self.figure_name == 'figure_2':
            ax_.grid(True)

    def return_ax_coors(self, fig, mother_ax, pos):
        if self.figure_name == 'figure_1':
            ax_w = 0.1
            ax_h = 0.03
        else:
            ax_w = 0.12
            ax_h = 0.06

        xstart, ystart = fig.transFigure.inverted().transform(mother_ax.transData.transform(pos))
        return [xstart, ystart, ax_w, ax_h]

    def _plot_one_signal(self, input_idx, conductance_type, holding_potential, start_idx, end_idx,
                        ax, fig, elec_x, elec_z, tvec):
        LFP = 1e3*np.load(join(self.sim_folder, 'sig_%s_%d_%s_%+d.npy' % (self.cell_name, input_idx, conductance_type,
                                                                          holding_potential)))[:, start_idx:end_idx]

        ax.scatter(elec_x[self.use_elec_idxs], elec_z[self.use_elec_idxs], c='k', s=10)

        if self.figure_name == 'figure_1':
            x_vec, y_vec = tvec, LFP
        elif self.figure_name == 'figure_2':
            x_vec, y_vec = aLFP.return_freq_and_psd(tvec, LFP)
        else:
            raise RuntimeError("Unknown figure name: %s" % self.figure_name)

        for idx in self.use_elec_idxs:
            self._plot_EC_signal_to_ax(fig, ax, idx, y_vec, elec_x[idx], elec_z[idx], x_vec, conductance_type)

    def make_figure(self, do_simulations=False):

        if do_simulations:
            self._do_all_simulations()
        tvec = np.load(join(self.sim_folder, 'tvec_%s.npy' % self.cell_name))
        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name))

        if not elec_x.shape == self.plot_positions[:, 0].shape:
            raise RuntimeError('Loaded elec_x shape: %s, Current elec_x shape: %s' %
                               (elec_x.shape, self.plot_positions[:, 0].shape))

        start_idx = np.argmin(np.abs(tvec - self.start_t))
        end_idx = np.argmin(np.abs(tvec - self.end_t))

        tvec = tvec[start_idx:end_idx]

        plt.close('all')
        fig = plt.figure(figsize=[5, 7])
        fig.subplots_adjust(hspace=0.15, wspace=0.15, top=0.9, bottom=0.1, left=0.07, right=0.95)

        ax1 = plt.subplot(221, **self.ax_dict)
        ax2 = plt.subplot(222, **self.ax_dict)
        ax3 = plt.subplot(223, **self.ax_dict)
        ax4 = plt.subplot(224, **self.ax_dict)

        ax_list = [ax1, ax2, ax3, ax4]

        [self._draw_morph_to_axis(ax, self.apic_idx) for ax in [ax1, ax2]]
        [self._draw_morph_to_axis(ax, self.soma_idx) for ax in [ax3, ax4]]

        # ax1.text(190, -300, '200 $\mu$m', verticalalignment='center', horizontalalignment='right')

        fig.text(0.01, 0.75, 'Apical input', rotation='vertical', va='center', size=20)
        fig.text(0.01, 0.25, 'Somatic input', rotation='vertical', va='center', size=20)
        fig.text(0.25, .95, '-80 mV', rotation='horizontal', ha='center', size=20)
        fig.text(0.75, .95, '-60 mV', rotation='horizontal', ha='center', size=20)

        ax_numb = 0
        for input_idx in [self.apic_idx, self.soma_idx]:
            for holding_potential in self.holding_potentials:
                for conductance_type in self.conductance_types:
                    # name = 'Soma' if input_idx == 0 else 'Apic'
                    # ax_list[ax_numb].text(0, self.ax_dict['ylim'][1], '%s %d mV' % (name, holding_potential),
                    #                       horizontalalignment='center')
                    self._plot_one_signal(input_idx, conductance_type, holding_potential, start_idx, end_idx,
                                         ax_list[ax_numb], fig, elec_x, elec_z, tvec)
                ax_numb += 1

        lines = []
        line_names = []
        for conductance_type in self.conductance_types:
            l, = plt.plot(0, 0, color=self.conductance_clr[conductance_type], lw=2)
            lines.append(l)
            line_names.append(conductance_type)

        if self.figure_name is 'figure_1':
            bar_ax = fig.add_axes(self.return_ax_coors(fig, ax4, (-500, -300)), **self.ec_ax_dict)
            bar_ax.axis('off')
            bar_ax.plot([0, 0], bar_ax.axis()[2:], lw=3, color='k')
            bar_ax.plot(bar_ax.axis()[:2], [0, 0], lw=3, color='k')

            bar_ax.text(2, bar_ax.axis()[2] + (bar_ax.axis()[3] - bar_ax.axis()[2])/2, '%1.2f $\mu V$'
                                            % (bar_ax.axis()[3] - bar_ax.axis()[2]), verticalalignment='bottom')

            bar_ax.text(12, bar_ax.axis()[2], '%d $ms$' % (bar_ax.axis()[1] - bar_ax.axis()[0]),
                        verticalalignment='top')

        mark_subplots(ax_list, xpos=0.12, ypos=0.9)
        [ax.axis('off') for ax in ax_list]

        fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=5)
        fig.savefig(join(self.figure_folder, '%s_%s.png' % (self.figure_name, self.cell_name)), dpi=200)


class Figure3(PaperFigures):

    def __init__(self):
        PaperFigures.__init__(self)
        self.cell_name = 'hay'
        self.figure_name = 'figure_3'
        self.sim_folder = join(self.root_folder, 'paper_simulations', 'intro_fig_white_noise')
        # self.timeres = 2**-4
        self.conductance_types = ['active', 'Ih_linearized', 'passive', 'Ih_linearized_frozen']
        self.conductance_dict = {'active': 'Active',
                                 'Ih_linearized': 'Linearized Ih',
                                 'passive': 'Passive',
                                 'Ih_linearized_frozen': 'Passive rescaled to Ih'}
        self.holding_potentials = [-80, -60]
        self.elec_apic_idx = 88
        self.elec_soma_idx = 18
        self.soma_idx = 0
        self.apic_idx = 852
        # self.make_single_case_figure(0, -60)
        # self.make_single_case_figure(852, -60)
        # self.make_single_case_figure(0, -80)
        # self.make_single_case_figure(852, -80)
        self.make_dual_case_figure()

    def make_single_case_figure(self, input_idx, holding_potential):

        tvec = np.load(join(self.sim_folder, 'tvec_%s.npy' % self.cell_name))
        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name))

        ax_dict = {'xlim': [1, 500],
                   }

        print input_idx, holding_potential
        plt.close('all')
        fig = plt.figure(figsize=[10, 5])
        fig.subplots_adjust(hspace=0.5, wspace=0.35, bottom=0.2)
        ax_morph = plt.subplot(143)
        ax_morph.axis('off')
        self._draw_morph_to_axis(ax_morph, input_idx)
        ax_morph.scatter(elec_x[self.elec_apic_idx], elec_z[self.elec_apic_idx])
        ax_morph.scatter(elec_x[self.elec_soma_idx], elec_z[self.elec_soma_idx])
        vm_ax_a = fig.add_subplot(2, 4, 1, ylim=[1e-3, 1e-0], title='$V_m$', **ax_dict)
        vm_ax_s = fig.add_subplot(2, 4, 5, ylim=[1e-4, 1e-1], title='$V_m$', **ax_dict)
        im_ax_a = fig.add_subplot(2, 4, 2, ylim=[1e-4, 1e-2], title='$I_m$', **ax_dict)
        im_ax_s = fig.add_subplot(2, 4, 6, ylim=[1e-7, 1e-5], title='$I_m$', **ax_dict)
        ec_ax_a = fig.add_subplot(2, 4, 4, ylim=[1e-4, 1e-2], title='$\Phi$', **ax_dict)
        ec_ax_s = fig.add_subplot(2, 4, 8, ylim=[1e-5, 1e-3], title='$\Phi$', **ax_dict)
        sig_ax_list = [vm_ax_s, vm_ax_a, im_ax_s, im_ax_a, ec_ax_s, ec_ax_a]
        [ax.grid(True) for ax in sig_ax_list]
        mark_subplots([vm_ax_a, im_ax_a, vm_ax_s, im_ax_s])
        mark_subplots([ec_ax_a, ec_ax_s], 'FG')
        mark_subplots(ax_morph, 'E', xpos=0, ypos=1)

        for conductance_type in self.conductance_types:
            self._plot_sigs(input_idx, conductance_type, holding_potential,
                            sig_ax_list, tvec)
        simplify_axes(fig.axes)

        lines = []
        line_names = []
        for conductance_type in self.conductance_types:
            l, = plt.plot(0, 0, color=self.conductance_clr[conductance_type], lw=2)
            lines.append(l)
            line_names.append(self.conductance_dict[conductance_type])
        fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=5)
        fig.savefig(join(self.figure_folder, '%s_%s_%d_%d.png' % (self.figure_name, self.cell_name,
                                                                  holding_potential, input_idx)), dpi=150)

    def make_dual_case_figure(self):

        tvec = np.load(join(self.sim_folder, 'tvec_%s.npy' % self.cell_name))
        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name))

        ax_dict = {'xlim': [1, 500],
                   }

        plt.close('all')
        fig = plt.figure(figsize=[10, 10])
        fig.subplots_adjust(hspace=0.5, wspace=0.35, bottom=0.2)
        ax_apic = fig.subplot(243)
        ax_soma = fig.subplot(247)
        ax_apic.axis('off')
        ax_soma.axis('off')
        self._draw_morph_to_axis(ax_apic, self.apic_idx)
        self._draw_morph_to_axis(ax_soma, self.soma_idx)
        ax_apic.scatter(elec_x[[self.elec_apic_idx, self.elec_soma_idx]],
                        elec_z[[self.elec_apic_idx, self.elec_soma_idx]])
        ax_soma.scatter(elec_x[[self.elec_apic_idx, self.elec_soma_idx]],
                        elec_z[[self.elec_apic_idx, self.elec_soma_idx]])
        vm_ax_a = fig.add_subplot(2, 4, 1, ylim=[1e-3, 1e-0], title='$V_m$', **ax_dict)
        vm_ax_s = fig.add_subplot(2, 4, 5, ylim=[1e-4, 1e-1], title='$V_m$', **ax_dict)
        im_ax_a = fig.add_subplot(2, 4, 2, ylim=[1e-4, 1e-2], title='$I_m$', **ax_dict)
        im_ax_s = fig.add_subplot(2, 4, 6, ylim=[1e-7, 1e-5], title='$I_m$', **ax_dict)
        ec_ax_a = fig.add_subplot(2, 4, 4, ylim=[1e-4, 1e-2], title='$\Phi$', **ax_dict)
        ec_ax_s = fig.add_subplot(2, 4, 8, ylim=[1e-5, 1e-3], title='$\Phi$', **ax_dict)
        sig_ax_list = [vm_ax_s, vm_ax_a, im_ax_s, im_ax_a, ec_ax_s, ec_ax_a]
        [ax.grid(True) for ax in sig_ax_list]
        mark_subplots([vm_ax_a, im_ax_a, vm_ax_s, im_ax_s])
        mark_subplots([ec_ax_a, ec_ax_s], 'FG')
        mark_subplots(ax_morph, 'E', xpos=0, ypos=1)

        for conductance_type in self.conductance_types:
            self._plot_sigs(input_idx, conductance_type, holding_potential,
                            sig_ax_list, tvec)
        simplify_axes(fig.axes)

        lines = []
        line_names = []
        for conductance_type in self.conductance_types:
            l, = plt.plot(0, 0, color=self.conductance_clr[conductance_type], lw=2)
            lines.append(l)
            line_names.append(self.conductance_dict[conductance_type])
        fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=5)
        fig.savefig(join(self.figure_folder, '%s_%s_%d_%d.png' % (self.figure_name, self.cell_name,
                                                                  holding_potential, input_idx)), dpi=150)

    def _plot_sigs(self, input_idx, conductance_type, holding_potential, axes, tvec):
        vmem = np.load(join(self.sim_folder, 'vmem_%s_%d_%s_%+d.npy' % (self.cell_name, input_idx, conductance_type,
                                                                          holding_potential)))
        imem = np.load(join(self.sim_folder, 'imem_%s_%d_%s_%+d.npy' % (self.cell_name, input_idx, conductance_type,
                                                                          holding_potential)))
        LFP = 1000 * np.load(join(self.sim_folder, 'sig_%s_%d_%s_%+d.npy' % (self.cell_name, input_idx, conductance_type,
                                                                          holding_potential)))
        freqs, vmem_psd_soma = aLFP.return_freq_and_psd(tvec, vmem[self.soma_idx, :])
        freqs, vmem_psd_apic = aLFP.return_freq_and_psd(tvec, vmem[self.apic_idx, :])
        freqs, imem_psd_soma = aLFP.return_freq_and_psd(tvec, imem[self.soma_idx, :])
        freqs, imem_psd_apic = aLFP.return_freq_and_psd(tvec, imem[self.apic_idx, :])
        freqs, LFP_psd_soma = aLFP.return_freq_and_psd(tvec, LFP[self.elec_soma_idx, :])
        freqs, LFP_psd_apic = aLFP.return_freq_and_psd(tvec, LFP[self.elec_apic_idx, :])

        axes[0].loglog(freqs, vmem_psd_soma[0], c=self.conductance_clr[conductance_type])
        axes[1].loglog(freqs, vmem_psd_apic[0], c=self.conductance_clr[conductance_type])
        axes[2].loglog(freqs, imem_psd_soma[0], c=self.conductance_clr[conductance_type])
        axes[3].loglog(freqs, imem_psd_apic[0], c=self.conductance_clr[conductance_type])
        axes[2].loglog(freqs, imem_psd_soma[0], c=self.conductance_clr[conductance_type])
        axes[3].loglog(freqs, imem_psd_apic[0], c=self.conductance_clr[conductance_type])
        axes[4].loglog(freqs, LFP_psd_soma[0], c=self.conductance_clr[conductance_type])
        axes[5].loglog(freqs, LFP_psd_apic[0], c=self.conductance_clr[conductance_type])


class Figure4(PaperFigures):

    def __init__(self):
        PaperFigures.__init__(self)
        self.cell_name = 'c12861'
        self.figure_name = 'figure_4'
        self.sim_folder = join(self.root_folder, 'paper_simulations', 'intro_fig_white_noise')
        self.timeres = 2**-4
        self.conductance_dict = {'active': 'Active',
                                 'Ih_linearized': 'Linearized Ih',
                                 'passive': 'Passive',
                                 'Ih_linearized_frozen': 'Passive rescaled to Ih'}
        self.holding_potentials = [-80, -60]
        self.elec_apic_idx = 88
        self.elec_soma_idx = 18
        if self.cell_name == 'n120':

            self.conductance_types = ['active', 'passive']
            self.soma_idx = 0
            self.apic_idx = 685
            self.use_elec_idxs = [33, 2, 78, 61, 22]
            self.elec_apic_idx = 65
            self.elec_soma_idx = 18
            self.ax_dict = {'ylim': [-200, 700], 'xlim': [-250, 250]}

        elif self.cell_name == 'c12861':
            self.conductance_types = ['active', 'passive']
            self.soma_idx = 0
            self.apic_idx = 1128
            self.use_elec_idxs = [26, 2, 77, 68, 28]#[19, 8, 23, 71, 52]
            self.elec_apic_idx = 75
            self.elec_soma_idx = 18
            self.ax_dict = {'ylim': [-200, 750], 'xlim': [-300, 300]}
        # self.make_figure(0, -60)
        # self.make_figure(self.apic_idx, -60)
        # self.make_figure(0, -80)
        # self.make_figure(self.apic_idx, -60)
        # self.make_figure(self.apic_idx, -80)
        self.make_figure(self.soma_idx, -70)
        # self.make_figure(self.soma_idx, -80)

    def make_figure(self, input_idx, holding_potential):

        tvec = np.load(join(self.sim_folder, 'tvec_%s.npy' % self.cell_name))
        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name))

        ax_dict = {'xlim': [1, 500],
                   }

        print input_idx, holding_potential
        plt.close('all')
        fig = plt.figure(figsize=[10, 5])
        fig.subplots_adjust(hspace=0.5, wspace=0.35, bottom=0.2)
        ax_morph = plt.subplot(143)
        ax_morph.axis('off')
        self._draw_morph_to_axis(ax_morph, input_idx)
        ax_morph.scatter(elec_x[self.elec_apic_idx], elec_z[self.elec_apic_idx])
        ax_morph.scatter(elec_x[self.elec_soma_idx], elec_z[self.elec_soma_idx])
        vm_ax_a = fig.add_subplot(2, 4, 1, ylim=[1e-3, 1e-1], title='$V_m$', **ax_dict)
        vm_ax_s = fig.add_subplot(2, 4, 5, ylim=[1e-2, 1e-0], title='$V_m$', **ax_dict)
        im_ax_a = fig.add_subplot(2, 4, 2, ylim=[1e-7, 1e-5], title='$I_m$', **ax_dict)
        im_ax_s = fig.add_subplot(2, 4, 6, ylim=[1e-4, 1e-2], title='$I_m$', **ax_dict)
        ec_ax_a = fig.add_subplot(2, 4, 4, ylim=[1e-4, 1e-2], title='$\Phi$', **ax_dict)
        ec_ax_s = fig.add_subplot(2, 4, 8, ylim=[1e-4, 1e-2], title='$\Phi$', **ax_dict)
        sig_ax_list = [vm_ax_s, vm_ax_a, im_ax_s, im_ax_a, ec_ax_s, ec_ax_a]
        [ax.grid(True) for ax in sig_ax_list]
        mark_subplots([vm_ax_a, im_ax_a, vm_ax_s, im_ax_s])
        mark_subplots([ec_ax_a, ec_ax_s], 'FG')
        mark_subplots(ax_morph, 'E', xpos=0, ypos=1)

        for conductance_type in self.conductance_types:
            self._plot_sigs(input_idx, conductance_type, holding_potential,
                            sig_ax_list, tvec)
        simplify_axes(fig.axes)

        lines = []
        line_names = []
        for conductance_type in self.conductance_types:
            l, = plt.plot(0, 0, color=self.conductance_clr[conductance_type], lw=2)
            lines.append(l)
            line_names.append(self.conductance_dict[conductance_type])
        fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=5)
        fig.savefig(join(self.figure_folder, '%s_%s_%d_%d.png' % (self.figure_name, self.cell_name,
                                                                  holding_potential, input_idx)), dpi=150)

    def _plot_sigs(self, input_idx, conductance_type, holding_potential, axes, tvec):
        vmem = np.load(join(self.sim_folder, 'vmem_%s_%d_%s_%+d.npy' % (self.cell_name, input_idx, conductance_type,
                                                                          holding_potential)))
        imem = np.load(join(self.sim_folder, 'imem_%s_%d_%s_%+d.npy' % (self.cell_name, input_idx, conductance_type,
                                                                          holding_potential)))
        LFP = 1000 * np.load(join(self.sim_folder, 'sig_%s_%d_%s_%+d.npy' % (self.cell_name, input_idx, conductance_type,
                                                                          holding_potential)))
        freqs, vmem_psd_soma = aLFP.return_freq_and_psd(tvec, vmem[self.soma_idx, :])
        freqs, vmem_psd_apic = aLFP.return_freq_and_psd(tvec, vmem[self.apic_idx, :])
        freqs, imem_psd_soma = aLFP.return_freq_and_psd(tvec, imem[self.soma_idx, :])
        freqs, imem_psd_apic = aLFP.return_freq_and_psd(tvec, imem[self.apic_idx, :])
        freqs, LFP_psd_soma = aLFP.return_freq_and_psd(tvec, LFP[self.elec_soma_idx, :])
        freqs, LFP_psd_apic = aLFP.return_freq_and_psd(tvec, LFP[self.elec_apic_idx, :])

        axes[0].loglog(freqs, vmem_psd_soma[0], c=self.conductance_clr[conductance_type])
        axes[1].loglog(freqs, vmem_psd_apic[0], c=self.conductance_clr[conductance_type])
        axes[2].loglog(freqs, imem_psd_soma[0], c=self.conductance_clr[conductance_type])
        axes[3].loglog(freqs, imem_psd_apic[0], c=self.conductance_clr[conductance_type])
        axes[2].loglog(freqs, imem_psd_soma[0], c=self.conductance_clr[conductance_type])
        axes[3].loglog(freqs, imem_psd_apic[0], c=self.conductance_clr[conductance_type])
        axes[4].loglog(freqs, LFP_psd_soma[0], c=self.conductance_clr[conductance_type])
        axes[5].loglog(freqs, LFP_psd_apic[0], c=self.conductance_clr[conductance_type])


class Figure5(PaperFigures):

    def __init__(self):
        PaperFigures.__init__(self)
        self.cell_name = 'hay'
        self.figure_name = 'figure_5'
        self.sim_folder = join(self.root_folder, 'paper_simulations', 'generic_study')
        self.timeres = 2**-4
        self.holding_potential = -80
        self.elec_apic_idx = 1
        self.elec_soma_idx = 13
        self.soma_idx = 0
        self.tau_w = 10
        self.numrows = 6
        self.numcols = 4
        self.apic_idx = 605
        self.axis_w_shift = 0.25
        self.mus = [-0.5, 0, 2]
        self.mu_name_dict = {-0.5: 'Regenerative', 0: 'Passive', 2: 'Restorative'}
        self.input_type = 'wn'
        self.input_idxs = [self.apic_idx, self.soma_idx]
        self.elec_idxs = [self.elec_apic_idx, self.elec_soma_idx]
        self.distributions = ['uniform', 'linear_increase', 'linear_decrease']
        self.mu_clr = lambda mu: plt.cm.Dark2(int(256. * (mu - np.min(self.mus))/
                                                  (np.max(self.mus) - np.min(self.mus))))
        self._initialize_figure()
        self._draw_channels_distribution()
        self.make_figure()
        self._finitialize_figure()

    def _initialize_figure(self):
        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name))

        plt.close('all')
        self.fig = plt.figure(figsize=[10, 10])
        self.fig.subplots_adjust(hspace=0.6, wspace=0.45, bottom=0.1, left=0)
        self.ax_morph_apic = self.fig.add_axes([0.05, 0.5, 0.1, 0.25])
        self.ax_morph_soma = self.fig.add_axes([0.05, 0.075, 0.1, 0.25])
        self.ax_morph_apic.axis('off')
        self.ax_morph_soma.axis('off')

        dummy_ax = self.fig.add_subplot(311)
        dummy_ax.axis('off')
        mark_subplots(dummy_ax, xpos=0.05, ypos=0.95)
        mark_subplots([self.ax_morph_apic, self.ax_morph_soma], 'BC')
        self._draw_simplified_morph_to_axis(self.ax_morph_apic, 'apic')
        self._draw_simplified_morph_to_axis(self.ax_morph_soma, 'soma')
        self.ax_morph_apic.scatter(elec_x[[self.elec_apic_idx, self.elec_soma_idx]],
                              elec_z[[self.elec_apic_idx, self.elec_soma_idx]])
        self.ax_morph_soma.scatter(elec_x[[self.elec_apic_idx, self.elec_soma_idx]],
                              elec_z[[self.elec_apic_idx, self.elec_soma_idx]])
        self._make_ax_dict()

        arrow_dict = {'width': 20, 'lw': 1, 'clip_on': False, 'color': '0.3', 'zorder': 0}
        arrow_dx = 200
        arrow_dz = 50

        self.ax_morph_apic.arrow(elec_x[self.elec_apic_idx] + 80, elec_z[self.elec_apic_idx] + 10,
                                 arrow_dx, arrow_dz, **arrow_dict)
        self.ax_morph_apic.arrow(elec_x[self.elec_soma_idx] + 80, elec_z[self.elec_soma_idx] + 10,
                                 arrow_dx, -arrow_dz, **arrow_dict)
        self.ax_morph_soma.arrow(elec_x[self.elec_apic_idx] + 80, elec_z[self.elec_apic_idx] + 10,
                                 arrow_dx, arrow_dz, **arrow_dict)
        self.ax_morph_soma.arrow(elec_x[self.elec_soma_idx] + 80, elec_z[self.elec_soma_idx] + 10,
                                 arrow_dx, -arrow_dz, **arrow_dict)

    def _draw_channels_distribution(self):

        for dist_num, distribution in enumerate(self.distributions):
            sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, 0, 0,
                                                        self.holding_potential, distribution, self.tau_w)
            dist_dict = np.load(join(self.sim_folder, 'dist_dict_%s.npy' % sim_name)).item()
            ax_line = self.fig.add_axes([0.25 + self.axis_w_shift*dist_num, 0.81, 0.15, 0.1],
                                      xticks=[], yticks=[], ylim=[0, 0.0007])
            ax_morph = self.fig.add_axes([0.17 + self.axis_w_shift*dist_num, 0.81, 0.1, 0.1])

            ax_morph.axis('off')
            self._draw_simplified_morph_to_axis(ax_morph, grading=distribution)

            simplify_axes(ax_line)
            argsort = np.argsort(dist_dict['dist'])
            dist = dist_dict['dist'][argsort]
            g = dist_dict['g_w_QA'][argsort]
            x = [dist[0], dist[-1]]
            y = [g[0], g[-1]]
            ax_line.plot(x, y, lw=2)

    def _finitialize_figure(self):

        lines = []
        line_names = []
        textsize = 20
        for mu in self.mus:
            l, = plt.plot(0, 0, color=self.mu_clr(mu), lw=2)
            lines.append(l)
            line_names.append(self.mu_name_dict[mu])
        self.fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=5)
        self.fig.text(0.5, 0.97, 'Channel distributions', ha='center', size=textsize)
        self.fig.text(0.5, 0.75, 'Apical input', ha='center', size=textsize)
        self.fig.text(0.5, 0.4, 'Somatic input', ha='center', size=textsize)

        self.fig.text(0.28, 0.91, 'Uniform', rotation='horizontal')
        self.fig.text(0.52, 0.91, 'Linear increase', rotation='horizontal')
        self.fig.text(0.75, 0.91, 'Linear decrease', rotation='horizontal')
        simplify_axes(self.ax_dict.values())
        self.fig.savefig(join(self.figure_folder, '%s_%s.png' % (self.figure_name, self.cell_name)), dpi=150)

    def _make_ax_dict(self):
        self.ax_dict = {}

        ax_w = 0.15
        ax_h = 0.1

        ax_props = {'xlim': [1, 500], 'ylim': [1e-6, 1e-3], 'xscale': 'log', 'yscale': 'log'}
        for input_num, input_idx in enumerate(self.input_idxs):
            for elec_num, elec in enumerate(self.elec_idxs):
                for dist_num, distribution in enumerate(self.distributions):
                    ax_x_start = 0.25 + self.axis_w_shift * dist_num
                    ax_y_start = 0.63 - input_num * 0.37 - elec_num * ax_h*1.5
                    ax = self.fig.add_axes([ax_x_start, ax_y_start, ax_w, ax_h], **ax_props)
                    # ax.set_title('%d %d %s' % (input_idx, elec, distribution))
                    ax.grid(True)
                    self.ax_dict[input_idx, elec, distribution] = ax


    def make_figure(self):
        for [input_idx, elec, distribution], ax in self.ax_dict.items():
            for mu in self.mus:
                self._plot_sigs(input_idx, ax, distribution, mu, elec)

    def _plot_sigs(self, input_idx, ax, distribution, mu, elec):
        sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, mu,
                                                    self.holding_potential, distribution, self.tau_w)
        LFP = 1000 * np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))
        tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))
        freqs, LFP_psd = aLFP.return_freq_and_psd(tvec, LFP[elec, :])
        ax.loglog(freqs, LFP_psd[0], c=self.mu_clr(mu), lw=2)


if __name__ == '__main__':

    # IntroFigures('hay', 'figure_2').make_figure(do_simulations=False)
    Figure3()
    # Figure4()
    # Figure5()
    # IntroFigures('n120', 'figure_2').make_figure(do_simulations=False)
    # IntroFigures('c12861', 'figure_2').make_figure(do_simulations=False)