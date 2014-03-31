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
    def __init__(self):
        self.figure_folder = join('/home', 'torbjone', 'work', 'aLFP', 'paper_figures')
        self.sim_folder = join('/home', 'torbjone', 'work', 'aLFP', 'paper_simulations')
        self.root_folder = join('/home', 'torbjone', 'work', 'aLFP')
        elec_x, elec_z = np.meshgrid(np.linspace(-200, 200, 7),
                                     np.linspace(-200, 1200, 15))
        elec_x = elec_x.flatten()
        elec_z = elec_z.flatten()
        elec_y = np.zeros(len(elec_x))
        self.plot_positions = np.array([elec_x, elec_y, elec_z]).T
        self.conductance_types = ['active_frozen', 'Ih_linearized', 'Ih_linearized_frozen']
        # plot_positions = ((0, 0, -170), (-170, 0, 0), (100, 0, -30),
        #                                 (50, 0, 300), (-100, 0, 400), (70, 0, 600),
        #                                 (-80, 0, 900), (160, 0, 1000))
        self.holding_potentials = [-80, -70, -60]
        self.soma_idx = 0
        self.apic_idx = 852
        self.use_elec_idxs = [8, 36, 25, 74, 86]

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

        [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1, color='gray', zorder=0, alpha=0.3)
         for idx in xrange(len(xmid))]

    @staticmethod
    def save_neural_sim_data(cell, sim_folder, electrode, input_idx,
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

    def _do_single_neural_simulation(self, conductance_type, holding_potential,
                                    input_idx, sim_folder, stimuli_function, end_t, cut_off):
        import neuron
        from hay_active_declarations import active_declarations as hay_active
        import LFPy

        electrode_parameters = {
            'sigma': 0.3,
            'x': np.array(self.plot_positions)[:, 0].flatten(),
            'y': np.array(self.plot_positions)[:, 1].flatten(),
            'z': np.array(self.plot_positions)[:, 2].flatten()
        }
        electrode = LFPy.RecExtElectrode(**electrode_parameters)
        neuron.h('forall delete_section()')
        neuron_models = join(self.root_folder, 'neuron_models')
        timeres = 2**-4
        neuron.load_mechanisms(join(neuron_models, 'hay', 'mod'))
        neuron.load_mechanisms(join(neuron_models))
        cell_params = {
            'morphology': join(neuron_models, 'hay', 'lfpy_version', 'morphologies', 'cell1.hoc'),
            'v_init': holding_potential,
            'passive': False,           # switch on passive mechs
            'nsegs_method': 'lambda_f',  # method for setting number of segments,
            'lambda_f': 100,           # segments are isopotential at this frequency
            'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
            'timeres_python': timeres,
            'tstartms': -cut_off,          # start time, recorders start at t=0
            'tstopms': end_t,
            'custom_code': [join(neuron_models, 'hay', 'lfpy_version', 'custom_codes.hoc')],
            'custom_fun': [hay_active],  # will execute this function
            'custom_fun_args': [{'conductance_type': conductance_type,
                                 'hold_potential': holding_potential}]
        }
        cell = LFPy.Cell(**cell_params)
        #self.quickplot_exp_setup(cell, electrode)
        plt.seed(1234)
        cell, syn, noiseVec = stimuli_function(cell, input_idx)
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)
        #plt.plot(cell.tvec, cell.somav)
        #plt.show()
        #freq, psd = self.return_freq_and_psd(cell.tvec, cell.imem[0, :])
        #plt.loglog(freq, psd[0, :])
        #plt.show()

        self.save_neural_sim_data(cell, sim_folder, electrode, input_idx,
                                conductance_type, holding_potential)
        # del synapse, cell


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

    def _do_all_simulations(self, sim_folder, stimuli_function, end_t, cut_off):
        for holding_potential in self.holding_potentials:
            for input_idx in [self.apic_idx, self.soma_idx]:
                for conductance_type in self.conductance_types:
                    self._do_single_neural_simulation(conductance_type, holding_potential,
                                                 input_idx, sim_folder, stimuli_function, end_t, cut_off)

    @staticmethod
    def _find_closest_idx_to_pos(pos, elec_x, elec_y, elec_z):
        return np.argmin(np.sum(np.array([(elec_x - pos[0])**2, (elec_y - pos[1])**2, (elec_z - pos[2])**2]), axis=0))

    @classmethod
    def _return_idxs_from_positions(cls, position_list, elec_x, elec_y, elec_z):
        return [cls._find_closest_idx_to_pos(pos, elec_x, elec_y, elec_z) for pos in position_list]

    def _plot_EC_signal_to_ax(self, fig, ax, idx, signal, elec_x_pos, elec_z_pos, x_vec, conductance_type):
        # ec_ax_dict = {'frameon': True,
        #               'xticks': [1e0, 1e1, 1e2],
        #               'xticklabels': [],
        #               'yticks': [1e-4, 1e-3, 1e-2, 1e-1],
        #               'yticklabels': [], 'xscale': 'log', 'yscale': 'log',
        #               'ylim': [1e-5, 1e-3],
        #               'xlim': [1, 500]}

        ec_ax_dict = {'frameon': False,
                      'xticks': [],
                      'xticklabels': [],
                      'yticks': [],
                      'yticklabels': [],
                      'ylim': [0, 0.05],
                      'xlim': [0, 80]}

        ec_plot_dict = {'color': self.conductance_clr[conductance_type],
                        'lw': 2,
                        'clip_on': False}
        ax_ = fig.add_axes(self.return_ax_coors(fig, ax, (elec_x_pos, elec_z_pos)), **ec_ax_dict)

        ax_.minorticks_off()
        simplify_axes(ax_)
        ax_.patch.set_alpha(0.0)
        ax_.plot(x_vec, signal[idx, :], **ec_plot_dict)

    def return_ax_coors(self, fig, mother_ax, pos, ax_w=0.1, ax_h=0.03):
        xstart, ystart = fig.transFigure.inverted().transform(mother_ax.transData.transform(pos))
        return [xstart, ystart, ax_w, ax_h]

    def make_figure(self, figure_name, do_simulations=False):

        if figure_name == 'figure_1':
            start_t = 0
            end_t = 80
            cut_off = 1000
            sim_folder = join(self.root_folder, 'paper_simulations', 'intro_fig_synapse')
            stimuli_function = self._make_syaptic_stimuli
        elif figure_name == 'figure_2':
            start_t = 0
            end_t = 1000
            cut_off = 6000
            sim_folder = join(self.root_folder, 'paper_simulations', 'intro_fig_white_noise')
            stimuli_function = self._make_white_noise_stimuli
        else:
            raise RuntimeError("Unrecognized figure name")

        if do_simulations:
            self._do_all_simulations(sim_folder, stimuli_function, end_t, cut_off)
        tvec = np.load(join(sim_folder, 'tvec.npy'))
        elec_x = np.load(join(sim_folder, 'elec_x.npy'))
        elec_y = np.load(join(sim_folder, 'elec_y.npy'))
        elec_z = np.load(join(sim_folder, 'elec_z.npy'))

        if not elec_x.shape == self.plot_positions[:, 0].shape:
            raise RuntimeError('Loaded elec_x shape: %s, Current elec_x shape: %s' %
                               (elec_x.shape, self.plot_positions[:, 0].shape))

        start_idx = np.argmin(np.abs(tvec - start_t))
        end_idx = np.argmin(np.abs(tvec - end_t))

        tvec = tvec[start_idx:end_idx]
        ax_dict = {'ylim': [-300, 1300], 'xlim': [-300, 300]}
        plt.close('all')
        fig = plt.figure(figsize=[5, 10])
        fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.1)
        ax1 = plt.subplot(321, **ax_dict)
        ax2 = plt.subplot(322, **ax_dict)
        ax3 = plt.subplot(323, **ax_dict)
        ax4 = plt.subplot(324, **ax_dict)
        ax5 = plt.subplot(325, **ax_dict)
        ax6 = plt.subplot(326, **ax_dict)

        idx_list = self._return_idxs_from_positions(self.plot_positions, elec_x, elec_y, elec_z)
        ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]

        # ax1.plot([300, 300], [-500, -500 - scale_factor_length], 'k', lw=3)
        # ax1.text(310, -500 - scale_factor_length/2, '0.1 $\mu$V',
        #          verticalalignment='center',
        #          horizontalalignment='left')
        [self._draw_morph_to_axis(ax, sim_folder, self.soma_idx) for ax in [ax1, ax3, ax5]]
        [self._draw_morph_to_axis(ax, sim_folder, self.apic_idx) for ax in [ax2, ax4, ax6]]

        # ax1.text(190, -300, '200 $\mu$m', verticalalignment='center', horizontalalignment='right')
        ax_numb = 0
        for holding_potential in self.holding_potentials:
            for input_idx in [self.soma_idx, self.apic_idx]:
                for conductance_type in self.conductance_types:
                    name = 'Soma' if input_idx == 0 else 'Apic'
                    ax_list[ax_numb].text(0, 1400, '%s %d mV' % (name, holding_potential))
                    self._plot_one_signal(input_idx, conductance_type, holding_potential, start_idx, end_idx, idx_list,
                                         sim_folder, ax_list[ax_numb], fig, elec_x, elec_z, tvec, figure_name)
                ax_numb += 1

        [ax1.plot(elec_x[idx], elec_z[idx], marker='$%d$' % idx, color='gray', alpha=0.2)
            for idx in xrange(len(elec_x))]

        lines = []
        line_names = []
        for conductance_type in self.conductance_types:
            l, = plt.plot(0, 0, color=self.conductance_clr[conductance_type], lw=2)
            lines.append(l)
            line_names.append(conductance_type)

        mark_subplots(ax_list, xpos=0.12, ypos=0.9)
        [ax.axis('off') for ax in ax_list]
        fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=2)
        fig.savefig(join(self.figure_folder, '%s.png' % figure_name), dpi=150)

    def _plot_one_signal(self, input_idx, conductance_type, holding_potential, start_idx, end_idx, idx_list,
                        sim_folder, ax, fig, elec_x, elec_z, tvec, figure_name):
        LFP = 1e3*np.load(join(sim_folder, 'sig_%d_%s_%+d.npy' % (input_idx, conductance_type,
                                                                  holding_potential)))[:, start_idx:end_idx]
        if figure_name == 'figure_1':
            x_vec, y_vec = tvec, LFP
        elif figure_name == 'figure_2':
            x_vec, y_vec = self.return_freq_and_psd(tvec, LFP)

        for idx in self.use_elec_idxs:
            self._plot_EC_signal_to_ax(fig, ax, idx, y_vec, elec_x[idx], elec_z[idx], x_vec, conductance_type)

if __name__ == '__main__':

    # TODO: Make compatible with all cell models.
    introfig = IntroFigures()
    introfig.make_figure('figure_2', do_simulations=True)