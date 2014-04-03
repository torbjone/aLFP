__author__ = 'torbjone'

import sys
import os
import numpy as np
from os.path import join
from plotting_convention import *
import scipy.fftpack as ff

class IntroFigures():
    np.random.seed(1234)
    conductance_clr = {'active': 'r', 'active_frozen': 'k', 'Ih_linearized': 'g', 'passive': 'b',
                       'Ih_linearized_frozen': 'c'}
    def __init__(self, cell_name, figure_name):
        self.cell_name = cell_name
        self.figure_name = figure_name
        self.figure_folder = join('/home', 'torbjone', 'work', 'aLFP', 'paper_figures')
        self.root_folder = join('/home', 'torbjone', 'work', 'aLFP')
        self.timeres = 2**-4
        self.holding_potentials = [-80, -70, -60]
        self._set_cell_specific_properties()
        self._set_figure_specific_properties()

    def _set_figure_specific_properties(self):

        if self.figure_name == 'figure_1':
            self.start_t = 0
            self.end_t = 80
            self.cut_off = 0
            self.sim_folder = join(self.root_folder, 'paper_simulations', 'intro_fig_synapse')
            self.stimuli_function = self._make_syaptic_stimuli
            self.ec_ax_dict = {'frameon': False,
                               'xticks': [],
                               'xticklabels': [],
                               'yticks': [],
                               'yticklabels': [],
                               'ylim': [0, 0.01],
                               'xlim': [0, 80]}
        elif self.figure_name == 'figure_2':
            self.start_t = 0
            self.end_t = 1000
            self.cut_off = 6000
            self.sim_folder = join(self.root_folder, 'paper_simulations', 'intro_fig_white_noise')
            self.stimuli_function = self._make_white_noise_stimuli
            self.ec_ax_dict = {'frameon': True,
                               'xticks': [1e0, 1e1, 1e2],
                               'xticklabels': [],
                               'yticks': [1e-4, 1e-3, 1e-2, 1e-1],
                               'yticklabels': [], 'xscale': 'log', 'yscale': 'log',
                               'ylim': [1e-5, 1e-3],
                               'xlim': [1, 500]}

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
            self.conductance_types = ['active',  'active_frozen', 'passive']
            self.soma_idx = 0
            self.apic_idx = 852
            self.use_elec_idxs = [8, 36, 25, 74, 86]
            self.ax_dict = {'ylim': [-300, 1300], 'xlim': [-300, 300]}

        elif self.cell_name == 'n120':

            elec_x, elec_z = np.meshgrid(np.linspace(-150, 150, 7),
                                         np.linspace(-150, 800, 15))
            elec_x = elec_x.flatten()
            elec_z = elec_z.flatten()
            elec_y = np.zeros(len(elec_x))
            self.plot_positions = np.array([elec_x, elec_y, elec_z]).T
            self.conductance_types = ['active', 'active_frozen', 'passive']
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
            self.conductance_types = ['active', 'active_frozen', 'passive']
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
        np.save(join(self.sim_folder, 'sig_%s.npy' % sim_name), electrode.LFP)
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
                'timeres_NEURON': self.timeres,   # dt of LFP and NEURON simulation.
                'timeres_python': self.timeres,
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
                    'timeres_NEURON': self.timeres,   # dt of LFP and NEURON simulation.
                    'timeres_python': self.timeres,
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

    def return_freq_and_psd(self, tvec, sig):
        """ Returns the power and freqency of the input signal"""
        sig = np.array(sig)
        if len(sig.shape) == 1:
            sig = np.array([sig])
        elif len(sig.shape) == 2:
            pass
        else:
            raise RuntimeError("Not compatible with given array shape!")
        sample_freq = ff.fftfreq(sig.shape[1], d=(tvec[1] - tvec[0])/1000.)
        pidxs = np.where(sample_freq >= 0)
        freqs = sample_freq[pidxs]
        Y = ff.fft(sig, axis=1)[:, pidxs[0]]
        power = np.abs(Y)/Y.shape[1]
        return freqs, power

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
            'tau': 10.,                # syn. time constant
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
                        'clip_on': False}
        ax_ = fig.add_axes(self.return_ax_coors(fig, ax, (elec_x_pos, elec_z_pos)), **self.ec_ax_dict)

        ax_.minorticks_off()
        simplify_axes(ax_)
        ax_.patch.set_alpha(0.0)
        ax_.plot(x_vec, signal[idx, :], **ec_plot_dict)

    def return_ax_coors(self, fig, mother_ax, pos, ax_w=0.1, ax_h=0.03):
        xstart, ystart = fig.transFigure.inverted().transform(mother_ax.transData.transform(pos))
        return [xstart, ystart, ax_w, ax_h]

    def _plot_one_signal(self, input_idx, conductance_type, holding_potential, start_idx, end_idx,
                        ax, fig, elec_x, elec_z, tvec):
        LFP = 1e3*np.load(join(self.sim_folder, 'sig_%s_%d_%s_%+d.npy' % (self.cell_name, input_idx, conductance_type,
                                                                       holding_potential)))[:, start_idx:end_idx]

        ax.scatter(elec_x[self.use_elec_idxs], elec_z[self.use_elec_idxs], c='k', s=3)

        if self.figure_name == 'figure_1':
            x_vec, y_vec = tvec, LFP
        elif self.figure_name == 'figure_2':
            x_vec, y_vec = self.return_freq_and_psd(tvec, LFP)
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
        fig = plt.figure(figsize=[10, 7])
        fig.subplots_adjust(hspace=0.2, top=0.95, bottom=0.1, left=0.05, right=0.95)
        ax1 = plt.subplot(231, **self.ax_dict)
        ax2 = plt.subplot(232, **self.ax_dict)
        ax3 = plt.subplot(233, **self.ax_dict)
        ax4 = plt.subplot(234, **self.ax_dict)
        ax5 = plt.subplot(235, **self.ax_dict)
        ax6 = plt.subplot(236, **self.ax_dict)

        ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]

        # ax1.plot([300, 300], [-500, -500 - scale_factor_length], 'k', lw=3)
        # ax1.text(310, -500 - scale_factor_length/2, '0.1 $\mu$V',
        #          verticalalignment='center',
        #          horizontalalignment='left')
        [self._draw_morph_to_axis(ax, self.soma_idx) for ax in [ax1, ax2, ax3]]
        [self._draw_morph_to_axis(ax, self.apic_idx) for ax in [ax4, ax5, ax6]]

        # ax1.text(190, -300, '200 $\mu$m', verticalalignment='center', horizontalalignment='right')
        ax_numb = 0
        for input_idx in [self.soma_idx, self.apic_idx]:
            for holding_potential in self.holding_potentials:
                for conductance_type in self.conductance_types:
                    name = 'Soma' if input_idx == 0 else 'Apic'
                    ax_list[ax_numb].text(0, self.ax_dict['ylim'][1], '%s %d mV' % (name, holding_potential),
                                          horizontalalignment='center')
                    self._plot_one_signal(input_idx, conductance_type, holding_potential, start_idx, end_idx,
                                         ax_list[ax_numb], fig, elec_x, elec_z, tvec)
                ax_numb += 1

        [ax1.plot(elec_x[idx], elec_z[idx], marker='$%d$' % idx, color='gray', alpha=0.2)
            for idx in xrange(len(elec_x))]

        lines = []
        line_names = []
        for conductance_type in self.conductance_types:
            l, = plt.plot(0, 0, color=self.conductance_clr[conductance_type], lw=2)
            lines.append(l)
            line_names.append(conductance_type)

        bar_ax = fig.add_axes(self.return_ax_coors(fig, ax3, (-500, -300)), **self.ec_ax_dict)
        bar_ax.axis('off')
        bar_ax.plot([0, 0], bar_ax.axis()[2:], lw=3, color='k')

        bar_ax.text(10, bar_ax.axis()[2] + (bar_ax.axis()[3] - bar_ax.axis()[2])/2, '%1.2f $\mu V$'
                                        % (bar_ax.axis()[3] - bar_ax.axis()[2]), verticalalignment='center')

        mark_subplots(ax_list, xpos=0.12, ypos=0.9)
        [ax.axis('off') for ax in ax_list]
        fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=3)
        fig.savefig(join(self.figure_folder, '%s_%s.png' % (self.figure_name, self.cell_name)), dpi=200)

if __name__ == '__main__':

    IntroFigures('hay', 'figure_2').make_figure(do_simulations=True)
    IntroFigures('n120', 'figure_2').make_figure(do_simulations=True)
    IntroFigures('c12861', 'figure_2').make_figure(do_simulations=True)