__author__ = 'torbjone'
import os
if not 'DISPLAY' in os.environ:
    import matplotlib
    matplotlib.use('Agg')
    at_stallo = True
else:
    at_stallo = False
import sys
from os.path import join
import numpy as np
import pylab as plt
from plotting_convention import *
import neuron
import LFPy
import aLFP
from matplotlib.colors import LogNorm
from scipy import stats
nrn = neuron.h
class GenericStudy:

    def __init__(self, cell_name, input_type, conductance='generic', extended_electrode=False):
        self.mu_name_dict = {-0.5: 'Regenerative ($\mu_{factor} = -0.5$)',
                             0: 'Passive ($\mu_{factor} = 0$)',
                             2: 'Restorative ($\mu_{factor} = 2$)'}
        self.cell_name = cell_name
        self.conductance = conductance
        self.input_type = input_type
        self.username = os.getenv('USER')
        self.root_folder = join('/home', self.username, 'work', 'aLFP')
        if at_stallo:
            self.figure_folder = join('/global', self.username, 'work', 'aLFP', 'generic_study')
            self.sim_folder = join('/global', self.username, 'work', 'aLFP', 'generic_study', cell_name)
        else:
            self.figure_folder = join('/home', self.username, 'work', 'aLFP', 'generic_study')
            self.sim_folder = join('/home', self.username, 'work', 'aLFP', 'generic_study', cell_name)
        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
        if not os.path.isdir(self.sim_folder):
            os.mkdir(self.sim_folder)
        # self.holding_potentials = [-80, -70, -60]
        self.plot_frequencies = [2, 10, 100]
        self.holding_potential = -80
        self.mus = [-0.5, 0, 2]
        self.mu_clr = {-0.5: 'r',
                       0: 'k',
                       2: 'b'}
        self.sec_clr_dict = {'soma': '0.3', 'dend': '0.5', 'apic': '0.7', 'axon': '0.1'}
        self._set_electrode_specific_properties(extended_electrode)
        self._set_input_specific_properties()
        self.num_tsteps = round(self.end_t/self.timeres_python + 1)
        self.divide_into_welch = 16.
        self.welch_dict = {'Fs': 1000 / self.timeres_python,
                           'NFFT': int(self.num_tsteps/self.divide_into_welch),
                           'noverlap': int(self.num_tsteps/self.divide_into_welch/2.),
                           'window': plt.window_hanning,
                           'detrend': plt.detrend_mean,
                           'scale_by_freq': True,
                           }

    def _set_input_specific_properties(self):
        if self.input_type == 'wn':
            print "white noise input"
            self.plot_psd = True
            self._single_neural_sim_function = self._run_single_wn_simulation
            self.timeres_NEURON = 2**-4
            self.timeres_python = 2**-4
            self.cut_off = 0
            self.end_t = 1000
            self.max_freq = 500
            self.short_list_elecs = [1, 1 + 6, 1 + 6 * 2]
        if self.input_type == 'distributed_synaptic':
            print "Distributed synaptic input"
            self.plot_psd = True
            self._single_neural_sim_function = self._run_distributed_synaptic_simulation
            self.timeres_NEURON = 2**-4
            self.timeres_python = 2**-4
            self.cut_off = 100
            self.end_t = 20000
            self.max_freq = 500
            self.short_list_elecs = [1, 1 + 6, 1 + 6 * 2]
        elif self.input_type == 'real_wn':
            print "REAL white noise input"
            self.plot_psd = True
            self.timeres_NEURON = 2**-4
            self.timeres_python = 2**-4
            self._single_neural_sim_function = self._run_single_wn_simulation
            self.cut_off = 0
            self.end_t = 5000
            self.max_freq = 500
            self.short_list_elecs = [1, 1 + 6, 1 + 6 * 2]
        elif self.input_type == 'synaptic':
            print "synaptic input"
            self.plot_psd = False
            self._single_neural_sim_function = self._run_single_synaptic_simulation
            self.timeres_NEURON = 2**-4
            self.timeres_python = 2**-4
            self.cut_off = 0
            self.end_t = 80
            self.short_list_elecs = [1, 1 + 6, 1 + 6 * 2]
        else:
            raise RuntimeError("Unrecognized input type.")

    def _set_electrode_specific_properties(self, extended_electrode):
        if self.cell_name in ['hay', 'zuchkova']:
            self.zmax = 1000
            if self.conductance is 'active':
                self.cell_plot_idxs = [805, 611, 0]
            elif self.conductance is 'Ih_linearized':
                self.cell_plot_idxs = [805, 611, 0]
            else:
                self.cell_plot_idxs = [605, 455, 0]
        elif self.cell_name == 'n120':
            self.zmax = 700
            self.cell_plot_idxs = [762, 827, 0]
        elif self.cell_name == 'c12861':
            self.zmax = 700
            self.cell_plot_idxs = [975, 762, 0]
        if extended_electrode:
            self._set_extended_electrode()
        else:
            self.elec_x = np.ones(3) * 100
            self.elec_y = np.zeros(3)
            self.elec_z = np.linspace(self.zmax, 0, 3)
            self.electrode_parameters = {
                'sigma': 0.3,
                'x': self.elec_x.flatten(),
                'y': self.elec_y.flatten(),
                'z': self.elec_z.flatten()
            }

    def _set_extended_electrode(self):
        self.distances = np.array([50, 100, 200, 400, 1600, 6400])
        # self.use_elec_idxs = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 23])

        elec_x, elec_z = np.meshgrid(self.distances,
                                     np.linspace(self.zmax, 0, 3))
        self.elec_x = elec_x.flatten()
        self.elec_z = elec_z.flatten()
        self.elec_y = np.zeros(len(self.elec_z))

        self.electrode_parameters = {
                'sigma': 0.3,
                'x': self.elec_x,
                'y': self.elec_y,
                'z': self.elec_z
        }

    def _return_cell(self, holding_potential, conductance_type, mu, distribution, tau_w):
        from hay_active_declarations import active_declarations as hay_active
        from ca1_sub_declarations import active_declarations as ca1_active
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
                                     'mu_factor': mu,
                                     'g_pas': 0.0002, # / 5,
                                     'distribution': distribution,
                                     'tau_w': tau_w,
                                     'total_w_conductance': 6.23843378791,# / 5,
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

    def save_neural_sim_single_input_data(self, cell, electrode, input_idx,
                             mu, distribution, taum, weight=None):

        if not os.path.isdir(self.sim_folder):
            os.mkdir(self.sim_folder)

        dist_dict = {'dist': np.zeros(cell.totnsegs),
             'sec_clrs': np.zeros(cell.totnsegs, dtype='|S3'),
             'g_pas_QA': np.zeros(cell.totnsegs),
             'V_rest_QA': np.zeros(cell.totnsegs),
             'v': np.zeros(cell.totnsegs),
             'mu_QA': np.zeros(cell.totnsegs),
             'tau_w_QA': np.zeros(cell.totnsegs),
             'g_w_QA': np.zeros(cell.totnsegs),
             }

        dist_dict = self._get_distribution(dist_dict, cell)
        if type(input_idx) is int:
            sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f_%1.4f' % (self.cell_name, self.input_type, input_idx, mu,
                                                        self.holding_potential, distribution, taum, weight)
        elif type(input_idx) in [list, np.ndarray]:
            sim_name = '%s_%s_multiple_%1.2f_%1.1f_%+d_%s_%1.2f_%1.4f' % (self.cell_name, self.input_type, weight, mu,
                                                                    self.holding_potential, distribution, taum, weight)
        elif type(input_idx) is str:
            sim_name = '%s_%s_%s_%1.1f_%+d_%s_%1.2f_%1.4f' % (self.cell_name, self.input_type, input_idx, mu,
                                                        self.holding_potential, distribution, taum, weight)
        else:
            print input_idx, type(input_idx)
            raise RuntimeError("input_idx is not recognized!")

        np.save(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)), cell.tvec)
        np.save(join(self.sim_folder, 'dist_dict_%s.npy' % sim_name), dist_dict)
        np.save(join(self.sim_folder, 'sig_%s.npy' % sim_name), np.dot(electrode.electrodecoeff, cell.imem))
        np.save(join(self.sim_folder, 'vmem_%s.npy' % sim_name), cell.vmem[self.cell_plot_idxs])
        np.save(join(self.sim_folder, 'imem_%s.npy' % sim_name), cell.imem[self.cell_plot_idxs])
        np.save(join(self.sim_folder, 'synidx_%s.npy' % sim_name), cell.synidx)

        np.save(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name), electrode.x)
        np.save(join(self.sim_folder, 'elec_y_%s.npy' % self.cell_name), electrode.y)
        np.save(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name), electrode.z)

        np.save(join(self.sim_folder, 'xstart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xstart)
        np.save(join(self.sim_folder, 'ystart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.ystart)
        np.save(join(self.sim_folder, 'zstart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zstart)
        np.save(join(self.sim_folder, 'xend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xend)
        np.save(join(self.sim_folder, 'yend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.yend)
        np.save(join(self.sim_folder, 'zend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zend)
        np.save(join(self.sim_folder, 'xmid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xmid)
        np.save(join(self.sim_folder, 'ymid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.ymid)
        np.save(join(self.sim_folder, 'zmid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zmid)
        np.save(join(self.sim_folder, 'diam_%s_%s.npy' % (self.cell_name, self.conductance)), cell.diam)

    def _get_distribution(self, dist_dict, cell):
        nrn.distance()
        idx = 0
        for sec in cell.allseclist:
            for seg in sec:
                for key in dist_dict:
                    if key == 'dist':
                        dist_dict[key][idx] = nrn.distance(seg.x)
                    elif key == 'sec_clrs':
                        dist_dict[key][idx] = self.sec_clr_dict[sec.name()[:4]]
                    else:
                        if hasattr(seg, key):
                            dist_dict[key][idx] = eval('seg.%s' % key)
                        else:
                            pass
                idx += 1
        return dist_dict

    def _recalculate_EP_with_new_elec(self, distribution, tau_w, input_idx, mu):

        cell = self._return_cell(self.holding_potential, 'generic', mu, distribution, tau_w)
        cell.tstartms = 0
        cell.tstopms = 1
        cell.simulate(rec_imem=True, rec_vmem=True)
        sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, mu,
                                                   self.holding_potential, distribution, tau_w)
        cell.imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))
        cell.tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))
        electrode = LFPy.RecExtElectrode(cell, **self.electrode_parameters)
        electrode.calc_lfp()

        np.save(join(self.sim_folder, 'sig_extended_%s.npy' % sim_name), electrode.LFP)

    def recalculate_EP(self):

        self._set_extended_electrode()
        for tau_w in [0.1, 1, 10, 100]:
            for mu in self.mus:
                for distribution in ['uniform', 'linear_increase', 'linear_decrease']:
                    for input_idx in self.cell_plot_idxs:
                        print tau_w, mu, distribution, input_idx
                        self._recalculate_EP_with_new_elec(distribution, tau_w, input_idx, mu)

    def _return_elec_subplot_number_with_distance(self, elec_number):
        """ Return the subplot number for the distance study
        """
        num_elec_cols = len(set(self.elec_x))
        num_elec_rows = len(set(self.elec_z))
        elec_idxs = np.arange(len(self.elec_x)).reshape(num_elec_rows, num_elec_cols)
        row, col = np.array(np.where(elec_idxs == elec_number))[:, 0]
        num_plot_cols = num_elec_cols + 5
        plot_number = row * num_plot_cols + col + 4
        # print self.elec_x[elec_number], self.elec_z[elec_number], plot_number
        return plot_number

    def _return_elec_row_col(self, elec_number):
        """ Return the subplot number for the distance study
        """
        num_elec_cols = len(set(self.elec_x))
        num_elec_rows = len(set(self.elec_z))
        elec_idxs = np.arange(len(self.elec_x)).reshape(num_elec_rows, num_elec_cols)
        row, col = np.array(np.where(elec_idxs == elec_number))[:, 0]
        return row, col

    def _return_elec_dist_idx(self, elec_number):
        """ Return the subplot number for the distance study
        """
        num_elec_cols = len(set(self.elec_x))
        num_elec_rows = len(set(self.elec_z))
        elec_idxs = np.arange(len(self.elec_x)).reshape(num_elec_rows, num_elec_cols)
        row, col = np.array(np.where(elec_idxs == elec_number))[:, 0]
        return col

    def _draw_all_elecs_with_distance(self, fig, distribution, tau_w, input_idx, weight):

        tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))
        num_elec_cols = len(set(self.elec_x))
        num_elec_rows = len(set(self.elec_z))
        num_plot_cols = num_elec_cols + 5
        num_plot_rows = num_elec_rows

        all_elec_ax = []
        for elec in xrange(len(self.elec_z)):
            plot_number = self._return_elec_subplot_number_with_distance(elec)
            ax = fig.add_subplot(num_plot_rows, num_plot_cols, plot_number, aspect='equal',
                                 title='$x=%d\mu m$' % self.elec_x[elec],
                                 xlim=[1e-1, 5e2])

            if self.elec_x[elec] == 50 and self.elec_z[elec] == 0:
                ax.set_xlabel('Hz')
                ax.set_ylabel('$\mu V$')
            ax.grid(True)
            simplify_axes(ax)
            all_elec_ax.append(ax)

        freq_ax = []
        freq_ax_norm = []
        for row in xrange(num_elec_rows):
            ax = fig.add_subplot(num_plot_rows, num_plot_cols, (row + 1) * num_plot_cols - 1, aspect='equal',
                                 title='Amp vs dist', xlim=[10, 10000], xlabel='$\mu m$')
            ax_norm = fig.add_subplot(num_plot_rows, num_plot_cols, (row + 1) * num_plot_cols, aspect='equal',
                                 title='Norm Amp vs dist', xlim=[10, 10000], xlabel='$\mu m$')

            ax.grid(True)
            ax_norm.grid(True)
            simplify_axes([ax, ax_norm])
            freq_ax.append(ax)
            freq_ax_norm.append(ax_norm)

        lines = []
        line_names = []
        freq_line_styles = ['-', '--', ':']
        for mu in self.mus:
            if type(input_idx) in [list, np.ndarray]:
                sim_name = '%s_%s_multiple_%1.2f_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, weight, mu,
                                                                  self.holding_potential, distribution,
                                                                  tau_w)
            elif type(input_idx) is str:
                sim_name = '%s_%s_%s_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, mu,
                                                                  self.holding_potential, distribution,
                                                                  tau_w)
            else:
                sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, mu,
                                                            self.holding_potential, distribution,
                                                            tau_w)
            # LFP = np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))[self.use_elec_idxs, :]
            LFP = np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))[:, :]

            freq_with_dist = np.zeros((num_elec_rows, num_elec_cols, len(self.plot_frequencies)))
            freqs_welch, sig_psd_welch = aLFP.return_freq_and_psd_welch(LFP, self.welch_dict)
            # freqs_welch, sig_psd_welch = aLFP.return_freq_and_psd(self.timeres_python/1000., LFP)

            idxs = [np.argmin(np.abs(freqs_welch - freq)) for freq in self.plot_frequencies]
            for elec in xrange(len(self.elec_z)):
                row, col = self._return_elec_row_col(elec)
                dist_idx = self._return_elec_dist_idx(elec)
                freq_with_dist[row, dist_idx, :] = sig_psd_welch[elec, idxs]
                all_elec_ax[elec].loglog(freqs_welch, sig_psd_welch[elec, :], color=self.mu_clr[mu], lw=2)

            for row in xrange(num_elec_rows):
                for freq_idx, freq in enumerate(self.plot_frequencies):
                    freq_ax[row].loglog(self.distances, freq_with_dist[row, :, freq_idx],
                                        freq_line_styles[freq_idx], color=self.mu_clr[mu], lw=3, alpha=0.7)
                    freq_ax_norm[row].loglog(self.distances,
                                             freq_with_dist[row, :, freq_idx] / freq_with_dist[row, 0, freq_idx],
                                             freq_line_styles[freq_idx], color=self.mu_clr[mu], lw=3, alpha=0.7)

            lines.append(plt.plot(0, 0, color=self.mu_clr[mu], lw=2)[0])
            line_names.append(self.mu_name_dict[mu])

        for ax in all_elec_ax + freq_ax:
            ax.set_ylim([1e-10, 1e-4])
        for ax in freq_ax_norm:
            ax.set_ylim([1e-4, 2e0])
        fig.legend(lines, line_names, frameon=False, ncol=3, loc='lower center')
        letter_list = 'HIJKLMMNOPQRRSTUVWXYZ'
        for elec in xrange(len(self.elec_z)):
            # row, col = self._return_elec_row_col(elec)
            # dist_idx = self._return_elec_dist_idx(elec)
            mark_subplots(all_elec_ax[elec], letter_list[elec])

    def _draw_all_elecs_with_distance_active(self, fig, input_idx):

        tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))
        num_elec_cols = len(set(self.elec_x))
        num_elec_rows = len(set(self.elec_z))
        num_plot_cols = num_elec_cols + 5
        num_plot_rows = num_elec_rows

        all_elec_ax = []
        for elec in xrange(len(self.elec_z)):
            plot_number = self._return_elec_subplot_number_with_distance(elec)
            ax = fig.add_subplot(num_plot_rows, num_plot_cols, plot_number, aspect='equal',
                                 title='$x=%d\mu m$' % self.elec_x[elec],
                                 xlim=[1, 510])
            ax.grid(True)
            simplify_axes(ax)
            all_elec_ax.append(ax)

        freq_ax = []
        freq_ax_norm = []
        for row in xrange(num_elec_rows):
            ax = fig.add_subplot(num_plot_rows, num_plot_cols, (row + 1) * num_plot_cols - 1, aspect='equal',
                                 title='Amp vs dist', xlim=[10, 10000], xlabel='$\mu m$')
            ax_norm = fig.add_subplot(num_plot_rows, num_plot_cols, (row + 1) * num_plot_cols, aspect='equal',
                                 title='Norm Amp vs dist', xlim=[10, 10000], xlabel='$\mu m$')

            ax.grid(True)
            ax_norm.grid(True)
            simplify_axes([ax, ax_norm])
            freq_ax.append(ax)
            freq_ax_norm.append(ax_norm)

        lines = []
        line_names = []
        freq_line_styles = ['-', '--', ':']
        sim_name = '%s_%s_%d_%+d_%s' % (self.cell_name, self.input_type, input_idx,
                                        self.holding_potential, self.conductance)
        LFP = np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))[:, :]
        freq_with_dist = np.zeros((num_elec_rows, num_elec_cols, len(self.plot_frequencies)))
        freqs, LFP_psd = aLFP.return_freq_and_psd(tvec, LFP)

        for elec in xrange(len(self.elec_z)):
            row, col = self._return_elec_row_col(elec)
            dist_idx = self._return_elec_dist_idx(elec)
            # sig_psd_welch, freqs_welch = mlab.psd(LFP[elec], **self.welch_dict)
            idxs = [np.argmin(np.abs(freqs - freq)) for freq in self.plot_frequencies]
            freq_with_dist[row, dist_idx, :] = LFP_psd[elec, idxs]#np.sqrt(sig_psd_welch[idxs])
            # print sig_psd_welch, freqs_welch
            # all_elec_ax[elec].loglog(xvec, yvec[elec, :], color=self.mu_clr[mu], lw=0.3, alpha=0.1)
            all_elec_ax[elec].loglog(freqs, LFP_psd[elec], color='k', lw=2, alpha=1)
            # all_elec_ax[elec].loglog(freqs_welch, np.sqrt(sig_psd_welch), color='k', lw=2, alpha=1)

        for row in xrange(num_elec_rows):
            for freq_idx, freq in enumerate(self.plot_frequencies):
                freq_ax[row].loglog(self.distances, freq_with_dist[row, :, freq_idx],
                                    freq_line_styles[freq_idx], color='k', lw=3, alpha=0.7)
                freq_ax_norm[row].loglog(self.distances,
                                         freq_with_dist[row, :, freq_idx] / freq_with_dist[row, 0, freq_idx],
                                         freq_line_styles[freq_idx], color='k', lw=3, alpha=0.7)

        lines.append(plt.plot(0, 0, color='k', lw=2)[0])
        # line_names.append('$\mu_{factor} = %1.1f$' % mu)

        for freq_idx, freq in enumerate(self.plot_frequencies):
            lines.append(plt.plot(0, 0, freq_line_styles[freq_idx], color='k', lw=2)[0])
            line_names.append('%d Hz' % freq)
        # [ax.set_xticks(ax.get_xticks()[::2]) for ax in all_elec_ax]
        # [ax.set_yticks(ax.get_yticks()[::2]) for ax in all_elec_ax]
        for ax in all_elec_ax + freq_ax:
            ax.set_ylim([1e-12, 1e-5])
        for ax in freq_ax_norm:
            ax.set_ylim([1e-4, 2e0])
        fig.legend(lines, line_names, frameon=False, ncol=2, loc='lower right')

    def _draw_membrane_signals_to_axes_distance_study(self, fig, distribution, tau_w, input_idx, weight):

        tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))
        num_elec_cols = len(set(self.elec_x))
        num_elec_rows = len(set(self.elec_z))
        num_plot_cols = num_elec_cols + 4
        num_plot_rows = num_elec_rows

        ax_vmem_1 = fig.add_subplot(num_plot_rows, num_plot_cols, 1)
        ax_vmem_2 = fig.add_subplot(num_plot_rows, num_plot_cols, 1 + num_plot_cols)
        ax_vmem_3 = fig.add_subplot(num_plot_rows, num_plot_cols, 1 + 2*num_plot_cols)

        ax_imem_1 = fig.add_subplot(num_plot_rows, num_plot_cols, 2)
        ax_imem_2 = fig.add_subplot(num_plot_rows, num_plot_cols, 2 + num_plot_cols)
        ax_imem_3 = fig.add_subplot(num_plot_rows, num_plot_cols, 2 + 2*num_plot_cols)

        mark_subplots([ax_vmem_1, ax_imem_1, ax_vmem_2, ax_imem_2, ax_vmem_3, ax_imem_3])
        simplify_axes([ax_vmem_1, ax_imem_1, ax_vmem_2, ax_imem_2, ax_vmem_3, ax_imem_3])

        # ax_sig_1.set_title('Extracellular\npotential', color='g')

        xlabel = 'Hz' if self.plot_psd else 'ms'
        [ax.set_ylabel('$nA$') for ax in [ax_imem_3, ax_imem_2, ax_imem_1]]
        [ax.set_xlabel('$Hz$') for ax in [ax_imem_3, ax_imem_2, ax_imem_1, ax_vmem_3, ax_vmem_2, ax_vmem_1]]
        [ax.set_ylabel('$mV$') for ax in [ax_vmem_3, ax_vmem_2, ax_vmem_1]]
        for mu in self.mus:
            if type(input_idx) in [list, np.ndarray]:
                sim_name = '%s_%s_multiple_%1.2f_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, weight, mu,
                                                                  self.holding_potential, distribution, tau_w)
            elif type(input_idx) is str:
                sim_name = '%s_%s_%s_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, mu,
                                                            self.holding_potential, distribution, tau_w)
            else:
                sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, mu,
                                                            self.holding_potential, distribution, tau_w)
            vmem = np.load(join(self.sim_folder, 'vmem_%s.npy' % sim_name))
            imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))

            # self._plot_sig_to_axes([ax_vmem_1, ax_vmem_2, ax_vmem_3], vmem[self.cell_plot_idxs], tvec, mu)
            # self._plot_sig_to_axes([ax_imem_1, ax_imem_2, ax_imem_3], imem[self.cell_plot_idxs], tvec, mu)
            self._plot_sig_to_axes([ax_vmem_1, ax_vmem_2, ax_vmem_3], vmem, tvec, mu)
            self._plot_sig_to_axes([ax_imem_1, ax_imem_2, ax_imem_3], imem, tvec, mu)

        ax_list = [ax_imem_1, ax_imem_2, ax_imem_3, ax_vmem_1, ax_vmem_2, ax_vmem_3]
        if self.plot_psd:
            [ax.set_xticks([1, 10, 100]) for ax in ax_list]
            [ax.set_xscale('log') for ax in ax_list]
            [ax.set_yscale('log') for ax in ax_list]
            [ax.grid(True) for ax in ax_list]
            [ax.set_xlim([1e-1, 5e2]) for ax in ax_list]

            for ax in [ax_vmem_1, ax_vmem_2, ax_vmem_3]:
                max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()[1:]) for l in ax.get_lines()])))
                ax.set_ylim([10**(max_exponent - 4), 10**max_exponent])

            for ax in [ax_imem_1, ax_imem_2, ax_imem_3]:
                max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()) for l in ax.get_lines()])))
                ax.set_ylim([10**(max_exponent - 4), 10**max_exponent])

    def _draw_membrane_signals_to_axes_distance_study_active(self, fig, input_idx):

        tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))
        num_elec_cols = len(set(self.elec_x))
        num_elec_rows = len(set(self.elec_z))
        num_plot_cols = num_elec_cols + 4
        num_plot_rows = num_elec_rows

        ax_vmem_1 = fig.add_subplot(num_plot_rows, num_plot_cols, 1)
        ax_vmem_2 = fig.add_subplot(num_plot_rows, num_plot_cols, 1 + num_plot_cols)
        ax_vmem_3 = fig.add_subplot(num_plot_rows, num_plot_cols, 1 + 2*num_plot_cols)

        ax_imem_1 = fig.add_subplot(num_plot_rows, num_plot_cols, 2)
        ax_imem_2 = fig.add_subplot(num_plot_rows, num_plot_cols, 2 + num_plot_cols)
        ax_imem_3 = fig.add_subplot(num_plot_rows, num_plot_cols, 2 + 2*num_plot_cols)

        ax_imem_1.set_title('Transmembrane\ncurrents', color='b')
        ax_vmem_1.set_title('Membrane\npotential', color='b')

        xlabel = '$Hz$' if self.plot_psd else '$ms$'
        [ax.set_ylabel('$nA$', color='b') for ax in [ax_imem_3, ax_imem_2, ax_imem_1]]
        # [ax.set_ylabel('$\mu V$', color='g') for ax in [ax_sig_3, ax_sig_2, ax_sig_1]]
        [ax.set_ylabel('$mV$', color='b') for ax in [ax_vmem_3, ax_vmem_2, ax_vmem_1]]

        sim_name = '%s_%s_%d_%+d_%s' % (self.cell_name, self.input_type, input_idx, self.holding_potential,
                                        self.conductance)
        vmem = np.load(join(self.sim_folder, 'vmem_%s.npy' % sim_name))
        imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))

        # plt.close('all')
        # clr = lambda idx: plt.cm.jet(int(256. * idx/(vmem.shape[0] - 1)))
        #
        # for idx in [0, 455, 805, 50, 100, 500, 600, 605, 800, 850, 900, 950]:
        #     y, x = mlab.psd(vmem[idx, :], **self.welch_dict)
        #     print x[1], np.sqrt(np.max(y[1:]) / y[1]), np.sqrt(np.max(y[1:])), np.sqrt(y[1])
        #     plt.loglog(x[:1000], np.sqrt(y[:1000]), color=clr(idx))
        #
        # plt.show()

        self._plot_sig_to_axes_active([ax_vmem_1, ax_vmem_2, ax_vmem_3], vmem[self.cell_plot_idxs], tvec)
        self._plot_sig_to_axes_active([ax_imem_1, ax_imem_2, ax_imem_3], imem[self.cell_plot_idxs], tvec)

        ax_list = [ax_imem_1, ax_imem_2, ax_imem_3, ax_vmem_1, ax_vmem_2, ax_vmem_3]
        if self.plot_psd:
            [ax.set_xticks([1, 10, 100]) for ax in ax_list]
            [ax.set_xscale('log') for ax in ax_list]
            [ax.set_yscale('log') for ax in ax_list]
            [ax.grid(True) for ax in ax_list]
            [ax.set_xlim([1, 450]) for ax in ax_list]

            for ax in [ax_vmem_1, ax_vmem_2, ax_vmem_3]:
                max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()[1:]) for l in ax.get_lines()])))
                ax.set_ylim([10**(max_exponent - 4), 10**max_exponent])

            for ax in [ax_imem_1, ax_imem_2, ax_imem_3]:
                max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()[1:]) for l in ax.get_lines()])))
                ax.set_ylim([10**(max_exponent - 4), 10**max_exponent])

    def _draw_membrane_signals_to_axes_q_value(self, fig, distribution, input_idx, weight):

        # tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))
        num_elec_cols = len(set(self.elec_x))
        num_elec_rows = len(set(self.elec_z))
        num_plot_cols = num_elec_cols + 4
        num_plot_rows = num_elec_rows

        ax_vmem_1 = fig.add_subplot(num_plot_rows, num_plot_cols, 1)
        ax_vmem_2 = fig.add_subplot(num_plot_rows, num_plot_cols, 1 + num_plot_cols)
        ax_vmem_3 = fig.add_subplot(num_plot_rows, num_plot_cols, 1 + 2*num_plot_cols)

        ax_imem_1 = fig.add_subplot(num_plot_rows, num_plot_cols, 2)
        ax_imem_2 = fig.add_subplot(num_plot_rows, num_plot_cols, 2 + num_plot_cols)
        ax_imem_3 = fig.add_subplot(num_plot_rows, num_plot_cols, 2 + 2*num_plot_cols)

        vmem_ax = [ax_vmem_1, ax_vmem_2, ax_vmem_3]
        imem_ax = [ax_imem_1, ax_imem_2, ax_imem_3]
        ax_imem_1.set_title('Transmembrane\ncurrents')
        ax_vmem_1.set_title('Membrane\npotential')

        tau_ws = [0.1, 1.0, 10., 100.]

        if self.plot_q:
            [ax.set_ylabel('Q') for ax in [ax_imem_3, ax_imem_2, ax_imem_1]]
            # [ax.set_ylabel('$\mu V$', color='g') for ax in [ax_sig_3, ax_sig_2, ax_sig_1]]
            [ax.set_ylabel('Q') for ax in [ax_vmem_3, ax_vmem_2, ax_vmem_1]]
        else:
            [ax.set_ylabel('$nA$') for ax in [ax_imem_3, ax_imem_2, ax_imem_1]]
            # [ax.set_ylabel('$\mu V$', color='g') for ax in [ax_sig_3, ax_sig_2, ax_sig_1]]
            [ax.set_ylabel('$mV$') for ax in [ax_vmem_3, ax_vmem_2, ax_vmem_1]]

        for mu in self.mus:
            dc_vmem = np.zeros((len(self.cell_plot_idxs), len(tau_ws)))
            dc_imem = np.zeros((len(self.cell_plot_idxs), len(tau_ws)))
            max_value_vmem = np.zeros((len(self.cell_plot_idxs), len(tau_ws)))
            max_value_imem = np.zeros((len(self.cell_plot_idxs), len(tau_ws)))
            freq_at_max_vmem = np.zeros((len(self.cell_plot_idxs), len(tau_ws)))
            freq_at_max_imem = np.zeros((len(self.cell_plot_idxs), len(tau_ws)))
            for tau_idx, tau_w in enumerate(tau_ws):
                if type(input_idx) in [list, np.ndarray]:
                    sim_name = '%s_%s_multiple_%1.2f_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, weight, mu,
                                                                      self.holding_potential, distribution, tau_w)
                else:
                    sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, mu,
                                                                self.holding_potential, distribution, tau_w)

                vmem = np.load(join(self.sim_folder, 'vmem_%s.npy' % sim_name))
                imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))

                for numb, idx in enumerate(self.cell_plot_idxs):
                    freqs, vmem_psd = aLFP.return_freq_and_psd(self.timeres_python/1000., vmem[idx, :])
                    freqs, imem_psd = aLFP.return_freq_and_psd(self.timeres_python/1000., imem[idx, :])

                    vmem_psd = vmem_psd[0]
                    imem_psd = imem_psd[0]

                    dc_vmem[numb, tau_idx] = vmem_psd[1]
                    dc_imem[numb, tau_idx] = imem_psd[1]
                    max_value_vmem[numb, tau_idx] = np.max(vmem_psd[1:])
                    max_value_imem[numb, tau_idx] = np.max(imem_psd[1:])
                    freq_at_max_vmem[numb, tau_idx] = freqs[np.argmax(vmem_psd[1:])]
                    freq_at_max_imem[numb, tau_idx] = freqs[np.argmax(imem_psd[1:])]

            for numb, idx in enumerate(self.cell_plot_idxs):
                if self.plot_q:
                    vmem_ax[numb].plot(tau_ws, max_value_vmem[numb] / dc_vmem[numb], 'x-', color=self.mu_clr[mu], lw=2)
                    imem_ax[numb].plot(tau_ws, max_value_imem[numb] / dc_imem[numb], 'x-', color=self.mu_clr[mu], lw=2)
                else:
                    vmem_ax[numb].plot(tau_ws, dc_vmem[numb], 'x-', color=self.mu_clr[mu], lw=2)
                    imem_ax[numb].plot(tau_ws, dc_imem[numb], 'x-', color=self.mu_clr[mu], lw=2)
                    vmem_ax[numb].plot(tau_ws, max_value_vmem[numb], '--', color=self.mu_clr[mu])
                    imem_ax[numb].plot(tau_ws, max_value_imem[numb], '--', color=self.mu_clr[mu])

        for ax in [ax_imem_1, ax_imem_2, ax_imem_3, ax_vmem_1, ax_vmem_2, ax_vmem_3]:
            if self.plot_q:
                ax.set_ylim([0.9e0, 10])
            ax.set_xscale('log')
            ax.grid(True)
            # ax.set_yscale('log')
            ax.set_xlabel(r'$\tau_w$')
            simplify_axes(ax)

    def _draw_all_elecs_q_value(self, fig, distribution, input_idx, weight):

        num_elec_cols = len(set(self.elec_x))
        num_elec_rows = len(set(self.elec_z))
        num_plot_cols = num_elec_cols + 5
        num_plot_rows = num_elec_rows

        tau_ws = [0.1, 1., 10., 100.]
        all_elec_ax = []
        for elec in xrange(len(self.elec_z)):
            plot_number = self._return_elec_subplot_number_with_distance(elec)

            ax = fig.add_subplot(num_plot_rows, num_plot_cols, plot_number,
                                 title='$x=%d\mu m$' % self.elec_x[elec],
                                 xlim=[0.1, 100])
            ax.grid(True)
            simplify_axes(ax)
            all_elec_ax.append(ax)

        lines = []
        line_names = []
        for mu in self.mus:
            dc = np.zeros((len(self.elec_z), len(tau_ws)))
            max_value = np.zeros((len(self.elec_z), len(tau_ws)))
            freq_at_max = np.zeros((len(self.elec_z), len(tau_ws)))
            for tau_idx, tau_w in enumerate(tau_ws):
                if type(input_idx) in [list, np.ndarray]:
                    sim_name = '%s_%s_multiple_%1.2f_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, weight, mu,
                                                                      self.holding_potential, distribution, tau_w)
                else:
                    sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, mu,
                                                                self.holding_potential, distribution, tau_w)
                # LFP = np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))[self.use_elec_idxs, :]
                LFP = np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))[:, :]
                freqs, LFP_psd = aLFP.return_freq_and_psd(self.timeres_python/1000., LFP)

                for elec in xrange(len(self.elec_z)):
                    # sig_psd_welch, freqs_welch = mlab.psd(LFP[elec], **self.welch_dict)
                    # sig_psd = np.sqrt(sig_psd_welch)
                    dc[elec, tau_idx] = LFP_psd[elec, 1]

                    max_value[elec, tau_idx] = np.max(LFP_psd[elec])
                    freq_at_max[elec, tau_idx] = freqs[np.argmax(LFP_psd[elec, 1:])]

            for elec in xrange(len(self.elec_z)):
                if self.plot_q:
                    all_elec_ax[elec].semilogx(tau_ws, max_value[elec, :] / dc[elec, :], 'x-', color=self.mu_clr[mu],
                                               lw=2, alpha=1)
                else:
                    all_elec_ax[elec].loglog(tau_ws, dc[elec, :], 'x-', color=self.mu_clr[mu], lw=2, alpha=1)
                    all_elec_ax[elec].loglog(tau_ws, max_value[elec, :], '--', color=self.mu_clr[mu], lw=1, alpha=1)
            lines.append(plt.plot(0, 0, color=self.mu_clr[mu], lw=2)[0])
            line_names.append('$\mu_{factor} = %1.1f$' % mu)

        if self.plot_q:
            for ax in all_elec_ax:
                ax.set_ylim([0.9e0, 3])

        fig.legend(lines, line_names, frameon=False, ncol=2, loc='lower right')

    def _plot_LFP_with_distance(self, distribution, tau_w, input_idx, weight=None):
        plt.close('all')
        fig = plt.figure(figsize=[24, 12])
        fig.subplots_adjust(hspace=0.5, wspace=0.5, top=0.9, bottom=0.13,
                            left=0.04, right=0.98)
        self._draw_setup_to_axis(fig, input_idx, plotpos=(1, 11, 3))
        self._draw_all_elecs_with_distance(fig, distribution, tau_w, input_idx, weight)
        self._draw_membrane_signals_to_axes_distance_study(fig, distribution, tau_w, input_idx, weight)

        fig.text(0.075, 0.95, 'Membrane\npotential', ha='center')
        fig.text(0.175, 0.95, 'Transmembrane\ncurrents', ha='center')
        fig.text(0.450, 0.95, 'Extracellular potential', ha='center')

        if type(input_idx) in [list, np.ndarray]:
            filename = ('LFP_with_distance_%s_multiple_%1.2f_%s_%1.2f' % (self.cell_name, weight, distribution, tau_w))
        elif type(input_idx) is str:
            filename = ('LFP_with_distance_%s_%s_%s_%1.2f' % (self.cell_name, input_idx, distribution, tau_w))
        else:
            filename = ('LFP_with_distance_%s_%d_%s_%1.2f' % (self.cell_name, input_idx, distribution, tau_w))
        fig.savefig(join(self.figure_folder, '%s.png' % filename), dpi=150)
        # sys.exit()

    def _plot_LFP_with_distance_active(self, input_idx):
        plt.close('all')
        fig = plt.figure(figsize=[24, 12])
        fig.subplots_adjust(hspace=0.5, wspace=0.5, top=0.9, bottom=0.13,
                            left=0.04, right=0.98)
        self._draw_setup_to_axis(fig, input_idx, plotpos=(1, 12, 3))
        self._draw_all_elecs_with_distance_active(fig, input_idx)
        self._draw_membrane_signals_to_axes_distance_study_active(fig, input_idx)

        filename = ('LFP_with_distance_%s_%d_%+d_%s' % (self.cell_name, input_idx,
                                                        self.holding_potential, self.conductance))
        fig.savefig(join(self.figure_folder, '%s.png' % filename), dpi=150)

    def _plot_q_value(self, distribution, input_idx, weight=None):
        plt.close('all')
        fig = plt.figure(figsize=[24, 12])
        self.plot_q = True

        fig.subplots_adjust(hspace=0.5, wspace=0.5, top=0.9, bottom=0.13, left=0.04, right=0.98)
        self._draw_all_elecs_q_value(fig, distribution, input_idx, weight)
        self._draw_membrane_signals_to_axes_q_value(fig, distribution, input_idx, weight)
        self._draw_setup_to_axis(fig, input_idx, plotpos=(1, 12, 3))

        if type(input_idx) in [list, np.ndarray]:
            filename = ('q_value_%s_multiple_%1.2f_%s' % (self.cell_name, weight, distribution))
        else:
            filename = ('q_value_%s_%d_%s' % (self.cell_name, input_idx,
                                              distribution))
        if not self.plot_q:
            filename += '_dc'
        fig.savefig(join(self.figure_folder, '%s.png' % filename), dpi=150)

    def plot_distributions(self, holding_potential):
        cell = self._return_cell(holding_potential, 'Ih_linearized')

        dist_dict = {'dist': np.zeros(cell.totnsegs),
                     'sec_clrs': np.zeros(cell.totnsegs, dtype='|S1'),
                     'g_pas': np.zeros(cell.totnsegs),
                     'cm': np.zeros(cell.totnsegs),
                     'e_pas': np.zeros(cell.totnsegs),
                     'v': np.zeros(cell.totnsegs),
                     'mu_Ih_linearized_v2': np.zeros(cell.totnsegs),
                     'gIhbar_Ih_linearized_v2': np.zeros(cell.totnsegs),
                     'wTau_Ih_linearized_v2': np.zeros(cell.totnsegs),
                     'wInf_Ih_linearized_v2': np.zeros(cell.totnsegs),
                     }
        dist_dict = self._get_distribution(dist_dict, cell)

        fig = plt.figure(figsize=[12, 8])
        fig.subplots_adjust(left=0.01, bottom=0.05, right=0.97)

        ax1 = fig.add_subplot(141)
        ax2 = fig.add_subplot(342, title='$g_L$', xlabel='$\mu m$', ylim=[0, 0.0001], xticks=[0, 400, 800, 1200],
                              ylabel='$S/cm^2$')
        ax3 = fig.add_subplot(343, title='$E_L$', xlabel='$\mu m$', xticks=[0, 400, 800, 1200], ylabel='mV')
        ax4 = fig.add_subplot(344, title='$V_R$', xlabel='$\mu m$', xticks=[0, 400, 800, 1200], ylabel='mV',
                              ylim=[np.min(dist_dict['v']) - 0.5, np.max(dist_dict['v']) + 0.5])
        ax5 = fig.add_subplot(346, title='$\mu= \overline{g}_{Ih} \partial w_\infty (E_{Ih} - V_R)$', xlabel='$\mu m$', xticks=[0, 400, 800, 1200])
        ax6 = fig.add_subplot(347, title='$\overline{g}_{Ih}$', xlabel='$\mu m$', ylim=[0, 0.02], xticks=[0, 400, 800, 1200],
                              ylabel='$S/cm^2$')
        ax7 = fig.add_subplot(348, title=r'$\tau_{w}$', xlabel='$\mu m$', ylim=[55, 56], xticks=[0, 400, 800, 1200])
        ax8 = fig.add_subplot(3, 4, 10, title='$w_\infty$', xlabel='$\mu m$', ylim=[0.048, 0.05],
                              xticks=[0, 400, 800, 1200])
        ax9 = fig.add_subplot(3, 4, 11, title='$\gamma_R=1+\overline{g}_{Ih}/g_L \cdot w_\infty$', xlabel='$\mu m$',
                               xticks=[0, 400, 800, 1200])

        ax10 = fig.add_subplot(3, 4, 12, title='$c_m$', xlabel='$\mu m$',
                               xticks=[0, 400, 800, 1200])

        gamma_R = 1 + dist_dict['gIhbar_Ih_linearized_v2'] / dist_dict['g_pas'] * dist_dict['wInf_Ih_linearized_v2']

        ax1.axis('off')
        [ax1.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], lw=2,
                  color=dist_dict['sec_clrs'][idx], zorder=0) for idx in xrange(len(cell.xmid))]
        ax1.plot(cell.xmid[0], cell.zmid[0], 'o', color=dist_dict['sec_clrs'][0], zorder=0, ms=10, mec='none')
        ax2.scatter(dist_dict['dist'], dist_dict['g_pas'], c=dist_dict['sec_clrs'], edgecolor='none')
        ax3.scatter(dist_dict['dist'], dist_dict['e_pas'], c=dist_dict['sec_clrs'], edgecolor='none')
        ax4.scatter(dist_dict['dist'], dist_dict['v'], c=dist_dict['sec_clrs'], edgecolor='none')
        ax5.scatter(dist_dict['dist'], dist_dict['mu_Ih_linearized_v2'], c=dist_dict['sec_clrs'], edgecolor='none')
        ax6.scatter(dist_dict['dist'], dist_dict['gIhbar_Ih_linearized_v2'], c=dist_dict['sec_clrs'], edgecolor='none')
        ax7.scatter(dist_dict['dist'], dist_dict['wTau_Ih_linearized_v2'], c=dist_dict['sec_clrs'], edgecolor='none')
        ax8.scatter(dist_dict['dist'], dist_dict['wInf_Ih_linearized_v2'], c=dist_dict['sec_clrs'], edgecolor='none')
        ax9.scatter(dist_dict['dist'], gamma_R, c=dist_dict['sec_clrs'], edgecolor='none')
        ax10.scatter(dist_dict['dist'], dist_dict['cm'], c=dist_dict['sec_clrs'], edgecolor='none')
        plt.savefig(join(self.figure_folder, 'Hay_linearized_params_%d.png' % holding_potential), dpi=150)

    def _plot_parameter_distributions(self, fig, input_idx, distribution, taum):

        ax0 = fig.add_subplot(351, xlabel='$\mu m$', ylim=[0, 0.001],
                              xticks=[0, 400, 800, 1200], ylabel='$S/cm^2$')
        ax1 = fig.add_subplot(356, title='$\mu$',
                              xlabel='$\mu m$', xticks=[0, 400, 800, 1200])
        ax2 = fig.add_subplot(3, 5, 11, title=r'$\tau_w$', ylim=[0, taum*2],
                              xlabel='$\mu m$', xticks=[0, 400, 800, 1200])
        mark_subplots([ax0, ax1, ax2], 'abc')
        simplify_axes([ax0, ax1, ax2])

        for mu in self.mus:
            sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, mu,
                                               self.holding_potential, distribution, taum)
            dist_dict = np.load(join(self.sim_folder, 'dist_dict_%s.npy' % sim_name)).item()
            largest_dist_idx = np.argmax(dist_dict['dist'])
            ax0.scatter(dist_dict['dist'], dist_dict['g_w_QA'], c=dist_dict['sec_clrs'],
                        edgecolor='none', label='g_w')

            ax0.scatter(dist_dict['dist'], dist_dict['g_pas_QA'], c=dist_dict['sec_clrs'],
                        edgecolor='none', label='g_pas')
            ax0.text(dist_dict['dist'][largest_dist_idx] + 100,
                     dist_dict['g_pas_QA'][largest_dist_idx], 'g_pas')
            ax0.text(dist_dict['dist'][largest_dist_idx] + 100,
                     dist_dict['g_w_QA'][largest_dist_idx], 'g_w')
            ax1.scatter(dist_dict['dist'], dist_dict['mu_QA'], c=dist_dict['sec_clrs'],
                        edgecolor='none')

            ax2.scatter(dist_dict['dist'], dist_dict['tau_w_QA'], c=dist_dict['sec_clrs'],
                        edgecolor='none')
        return [ax0, ax1, ax2]

    def _plot_sig_to_axes(self, ax_list, sig, tvec, mu):
        if not len(ax_list) == len(sig):
            raise RuntimeError("Something wrong with number of electrodes!")

        for idx, ax in enumerate(ax_list):
            if self.plot_psd:
                # xvec_w, yvec_w = aLFP.return_freq_and_psd(tvec, sig[idx, :])
                xvec_w, yvec_w = aLFP.return_freq_and_psd_welch(sig[idx, :-1], self.welch_dict)
                yvec_w = yvec_w[0]
                # yvec_w, xvec_w = mlab.psd(sig[idx, :], **self.welch_dict)
                # yvec_w = np.sqrt(yvec_w)
            else:
                xvec = tvec
                yvec = sig[idx]
            # ax.plot(xvec, yvec, color=self.mu_clr[mu], lw=1, alpha=0.1)

            max_freq = np.argmax(yvec_w[1:])
            ax.set_title('Mean: %1.3f\nSTD: %1.3f' % (np.mean(sig[idx, :-1]), np.std(sig[idx, :-1])))
            ax.plot(xvec_w, yvec_w, color=self.mu_clr[mu], lw=2)
            # if mu == 2.0:
            #     ax.set_xlabel('Max f: %1.1f Hz' % xvec_w[max_freq])
            # ax.plot(xvec_w[max_freq], yvec_w[max_freq], 'o', c='r', lw=1, alpha=1)
            # ax.text(xvec_w[max_freq] + 10, yvec_w[max_freq], '%1.1fHz' % xvec_w[max_freq])

            # print len(xvec), len(xvec_w)
            # fi = plt.figure()
            # ax_ = fi.add_subplot(111, xlim=[1, 1000])
            # ax_.loglog(xvec, yvec, color=self.mu_clr[mu], lw=2)
            # ax_.loglog(xvec_w, yvec_w, color=self.mu_clr[mu], lw=1)
            # plt.show()
            if self.plot_psd:
                ax.set_yscale('log')
                ax.set_xscale('log')

    def _plot_sig_to_axes_active(self, ax_list, sig, tvec):
        if not len(ax_list) == len(sig):
            raise RuntimeError("Something wrong with number of electrodes!")
        for idx, ax in enumerate(ax_list):
            if self.plot_psd:
                xvec, yvec = aLFP.return_freq_and_psd(tvec, sig[idx, :])
                yvec = yvec[0]
                # yvec_w, xvec_w = mlab.psd(sig[idx, :], **self.welch_dict)
                # yvec_w = np.sqrt(yvec_w)
            else:
                xvec = tvec
                yvec = sig[idx]
            if idx == 2:
                print yvec[1], np.max(yvec[1:]), np.max(yvec[1:]) / yvec[1], xvec[np.argmax(yvec[1:])]
            ax.plot(xvec, yvec, color='k', lw=2)
            # ax.plot(xvec_w, yvec_w, color='k', lw=2)

    def _plot_signals(self, fig, input_idx, distribution, tau_w):
        ax_vmem_1 = fig.add_subplot(3, 5, 3, ylim=[1e-7, 1e-2])
        ax_vmem_2 = fig.add_subplot(3, 5, 8, ylim=[1e-7, 1e-2])
        ax_vmem_3 = fig.add_subplot(3, 5, 13, ylim=[1e-7, 1e-2])

        ax_imem_1 = fig.add_subplot(3, 5, 4, ylim=[1e-9, 1e-5])
        ax_imem_2 = fig.add_subplot(3, 5, 9, ylim=[1e-9, 1e-5])
        ax_imem_3 = fig.add_subplot(3, 5, 14, ylim=[1e-9, 1e-5])

        ax_sig_1 = fig.add_subplot(3, 5, 5, ylim=[1e-9, 1e-6])
        ax_sig_2 = fig.add_subplot(3, 5, 10, ylim=[1e-9, 1e-6])
        ax_sig_3 = fig.add_subplot(3, 5, 15, ylim=[1e-9, 1e-6])

        ax_imem_1.set_title('Transmembrane\ncurrents')
        ax_vmem_1.set_title('Membrane\npotential')
        ax_sig_1.set_title('Extracellular\npotential')

        xlabel = '$Hz$' if self.plot_psd else '$ms$'
        [ax.set_ylabel('$nA$') for ax in [ax_imem_3, ax_imem_2, ax_imem_1]]
        [ax.set_ylabel('$\mu V$') for ax in [ax_sig_3, ax_sig_2, ax_sig_1]]
        [ax.set_ylabel('$mV$') for ax in [ax_vmem_3, ax_vmem_2, ax_vmem_1]]
        [ax.set_xlabel(xlabel) for ax in [ax_vmem_3, ax_vmem_2, ax_vmem_1]]
        [ax.set_xlabel(xlabel) for ax in [ax_imem_3, ax_imem_2, ax_imem_1]]
        [ax.set_xlabel(xlabel) for ax in [ax_sig_3, ax_sig_2, ax_sig_1]]

        tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))

        if not len(tvec) == self.num_tsteps:
            raise RuntimeError("Not the expected number of time steps %d, %d" % (len(tvec), self.num_tsteps))
        lines = []
        line_names = []

        for mu in self.mus:
            sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, mu,
                                               self.holding_potential, distribution, tau_w)
            LFP = np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))
            if hasattr(self, 'short_list_elecs'):
                LFP = LFP[self.short_list_elecs, :]
            vmem = np.load(join(self.sim_folder, 'vmem_%s.npy' % sim_name))
            imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))

            lines.append(plt.plot(0, 0, color=self.mu_clr[mu], lw=2)[0])
            line_names.append('$\mu_{factor} = %1.1f$' % mu)

            self._plot_sig_to_axes([ax_sig_1, ax_sig_2, ax_sig_3], LFP, tvec, mu)
            self._plot_sig_to_axes([ax_vmem_1, ax_vmem_2, ax_vmem_3], vmem[self.cell_plot_idxs], tvec, mu)
            self._plot_sig_to_axes([ax_imem_1, ax_imem_2, ax_imem_3], imem[self.cell_plot_idxs], tvec, mu)

        ax_list = [ax_vmem_1, ax_vmem_2, ax_vmem_3, ax_imem_1, ax_imem_2, ax_imem_3,
                   ax_sig_1, ax_sig_2, ax_sig_3]

        if self.plot_psd:
            [ax.set_xticks([1, 10, 100]) for ax in ax_list]
            [ax.set_xscale('log') for ax in ax_list]
            [ax.set_yscale('log') for ax in ax_list]
            [ax.grid(True) for ax in ax_list]
            [ax.set_xlim([1, self.max_freq - 50]) for ax in ax_list]

            # for ax in [ax_vmem_1, ax_vmem_2, ax_vmem_3]:
            #     max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()[1:]) for l in ax.get_lines()])))
            #     ax.set_ylim([10**(max_exponent - 4), 10**max_exponent])
            #
            # for ax in [ax_imem_1, ax_imem_2, ax_imem_3]:
            #     max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()) for l in ax.get_lines()])))
            #     ax.set_ylim([10**(max_exponent - 4), 10**max_exponent])
            #
            # for ax in [ax_sig_1, ax_sig_2, ax_sig_3]:
            #     max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()) for l in ax.get_lines()])))
            #     ax.set_ylim([10**(max_exponent - 4), 10**max_exponent])
        else:
            [ax.set_xticks(ax.get_xticks()[::2]) for ax in ax_list]
            [ax.set_yticks(ax.get_yticks()[::2]) for ax in ax_list]
            # [ax.set_ylim([self.holding_potential - 1, self.holding_potential + 10])
            #     for ax in [ax_vmem_1, ax_vmem_2, ax_imem_3]]

        # color_axes([ax_sig_3, ax_sig_2, ax_sig_1], 'g')
        # color_axes([ax_imem_3, ax_imem_2, ax_imem_1, ax_vmem_3, ax_vmem_2, ax_vmem_1], 'b')
        simplify_axes(ax_list)
        mark_subplots(ax_list, 'efghijklmn')
        fig.legend(lines, line_names, frameon=False, ncol=3, loc='lower right')

    def plot_summary(self, input_idx, distribution, tau_w):

        plt.close('all')
        fig = plt.figure(figsize=[12, 8])
        fig.subplots_adjust(hspace=0.5, wspace=0.7, top=0.9, bottom=0.13,
                            left=0.1, right=0.95)

        # self._draw_setup_to_axis(fig, input_idx, distribution)
        # self._plot_parameter_distributions(fig, input_idx, distribution, tau_w)
        self._plot_signals(fig, input_idx, distribution, tau_w)
        filename = ('generic_summary_%s_%s_%d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx,
                                                           distribution, tau_w))
        filename = '%s_psd' % filename if self.plot_psd else filename
        fig.savefig(join(self.figure_folder, '%s.png' % filename))

    def _draw_setup_to_axis(self, fig, input_idx, distribution=None, plotpos=152):
        if type(plotpos) is int:
            ax = fig.add_subplot(plotpos)
        else:
            # ax = fig.add_subplot(plotpos[0], plotpos[1], plotpos[2], aspect='equal')
            ax = fig.add_axes([0.15, 0.15, 0.2, 0.7], aspect='equal')

        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name))

        xstart = np.load(join(self.sim_folder, 'xstart_%s_%s.npy' % (self.cell_name, self.conductance)))
        zstart = np.load(join(self.sim_folder, 'zstart_%s_%s.npy' % (self.cell_name, self.conductance)))
        xend = np.load(join(self.sim_folder, 'xend_%s_%s.npy' % (self.cell_name, self.conductance)))
        zend = np.load(join(self.sim_folder, 'zend_%s_%s.npy' % (self.cell_name, self.conductance)))
        xmid = np.load(join(self.sim_folder, 'xmid_%s_%s.npy' % (self.cell_name, self.conductance)))
        zmid = np.load(join(self.sim_folder, 'zmid_%s_%s.npy' % (self.cell_name, self.conductance)))

        if distribution is None:
            mark_subplots(ax, 'G', xpos=0, ypos=1)
            sec_clrs = ['0.7'] * len(xmid)
        else:
            mark_subplots(ax, 'd', xpos=0, ypos=1)
            example_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, 0,
                                               self.holding_potential, distribution, 30)
            dist_dict = np.load(join(self.sim_folder, 'dist_dict_%s.npy' % example_name)).item()
            sec_clrs = dist_dict['sec_clrs']

        [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
                 color=sec_clrs[idx], zorder=0) for idx in xrange(len(xmid))]
        ax.plot(xmid[0], zmid[0], 'o', color=sec_clrs[0], zorder=0, ms=15, mec='none')
        ax.plot(xmid[self.cell_plot_idxs], zmid[self.cell_plot_idxs], 'o', c='orange',
                zorder=2, ms=15, mec='none')
        if type(input_idx) is str:
            sim_name = '%s_%s_%s_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, 0,
                                                        self.holding_potential, 'linear_increase', 30)
            synidx = np.load(join(self.sim_folder, 'synidx_%s.npy' % sim_name))
            ax.plot(xmid[synidx], zmid[synidx], 'y.', zorder=3, ms=10)
        else:
            ax.plot(xmid[input_idx], zmid[input_idx], 'y*', zorder=3, ms=20)

        # ax.plot(xmid[self.cell_plot_idxs], zmid[self.cell_plot_idxs], 'D', color='m', zorder=0, ms=10, mec='none')

        if hasattr(self, 'short_list_elecs'):
            ax.scatter(elec_x[self.short_list_elecs], elec_z[self.short_list_elecs],
                       c='c', edgecolor='none', s=200)
        else:
            ax.scatter(elec_x[self.short_list_elecs], elec_z[self.short_list_elecs],
                       c='c', edgecolor='none', s=200)

        arrow_dict = {'width': 2, 'lw': 1, 'clip_on': False, 'color': 'c', 'zorder': 0}
        arrow_dx = 70
        arrow_dz = -30

        for row, elec in enumerate(self.short_list_elecs):
            ax.arrow(elec_x[elec] + 80, elec_z[elec] - 10 * (row - 1), arrow_dx, arrow_dz * (row - 1), **arrow_dict)
        arrow_dict.update({'color': 'orange'})
        for row, comp in enumerate(self.cell_plot_idxs):
            ax.arrow(xmid[comp] - 80, zmid[comp] - 10 * (row - 1), - arrow_dx,
                     arrow_dz * (row - 1), **arrow_dict)
        ax.axis('off')

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

    def _make_WN_input(self, cell, max_freq):
        """ White Noise input ala Linden 2010 is made """
        tot_ntsteps = round((cell.tstopms - cell.tstartms)/\
                      cell.timeres_NEURON + 1)
        I = np.zeros(tot_ntsteps)
        tvec = np.arange(tot_ntsteps) * cell.timeres_NEURON
        for freq in xrange(1, max_freq + 1):
            I += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())
        return I

    def _make_white_noise_stimuli(self, cell, input_idx, weight=None):

        if self.input_type == 'wn':
            input_scaling = 0.0005
            max_freq = 500
            input_array = input_scaling * self._make_WN_input(cell, max_freq)
        elif self.input_type == 'real_wn':
            tot_ntsteps = round((cell.tstopms - cell.tstartms)/\
                          cell.timeres_NEURON + 1)
            input_scaling = .1
            input_array = input_scaling * (np.random.random(tot_ntsteps) - 0.5)
        else:
            raise RuntimeError("Unrecognized input_type!")
        noise_vec = neuron.h.Vector(input_array) if weight is None else neuron.h.Vector(input_array * weight)
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
        noise_vec.play(syn._ref_amp, cell.timeres_NEURON)
        return cell, syn, noise_vec

    def _quickplot_setup(self, cell, electrode, input_idx=None):
        plt.plot(cell.xmid[self.cell_plot_idxs], cell.zmid[self.cell_plot_idxs], 'bD', zorder=2, ms=5, mec='none')
        [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], lw=2, zorder=0, color='gray')
         for idx in xrange(len(cell.xmid))]
        plt.plot(cell.xmid[0], cell.zmid[0], 'o', zorder=0, ms=10, mec='none', color='gray')
        if not input_idx is None:
            plt.plot(cell.xmid[input_idx], cell.zmid[input_idx], '*', zorder=0, ms=10, color='y')
        plt.scatter(electrode.x, electrode.z, c='g', edgecolor='none', s=50)
        plt.show()
        sys.exit()

    def _run_single_synaptic_simulation(self, mu, input_idx, distribution, tau_w):
        plt.seed(1234)
        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        neuron.h('forall delete_section()')
        cell = self._return_cell(self.holding_potential, 'generic', mu, distribution, tau_w)
        # self._quickplot_setup(cell, electrode)
        self._make_syaptic_stimuli(cell, input_idx)
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)
        self.save_neural_sim_single_input_data(cell, electrode, input_idx, mu, distribution, tau_w)

    def _run_single_wn_simulation(self, mu, input_idx, distribution, tau_w):
        plt.seed(1234)
        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        neuron.h('forall delete_section()')
        cell = self._return_cell(self.holding_potential, 'generic', mu, distribution, tau_w)
        # self._quickplot_setup(cell, electrode)
        cell, syn, noiseVec = self._make_white_noise_stimuli(cell, input_idx)
        print "Starting simulation ..."
        # dist = []
        # gh = []
        # nrn.distance()
        # for sec in cell.allseclist:
        #     for seg in sec:
        #         dist.append(nrn.distance(seg.x))
        #         gh.append(seg.g_w_QA)
        #
        # plt.plot(dist, gh, 'o')
        # plt.show()
        # np.save(join(self.root_folder, 'linear_increase.npy'), [dist, gh])

        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)
        # plt.close('all')
        # [plt.plot(cell.vmem[idx,:]) for idx in xrange(cell.vmem.shape[0])]
        # plt.show()
        self.save_neural_sim_single_input_data(cell, electrode, input_idx, mu, distribution, tau_w)

    def run_all_distributed_synaptic_input_simulations(self, mu, weight):
        distributions = ['linear_increase']
        input_poss = ['tuft']
        tau_ws = [30]
        tot_sims = len(input_poss) * len(tau_ws) * len(distributions) * len(self.mus)
        i = 1
        for distribution in distributions:
            for input_pos in input_poss:
                for tau_w in tau_ws:
                    #for mu in self.mus:
                    print distribution, input_pos, mu, weight
                    self._run_distributed_synaptic_simulation(mu, input_pos, distribution, tau_w, weight)
                    #i += 1

    def _run_distributed_synaptic_simulation(self, mu, input_idx, distribution, tau_w, weight):
        plt.seed(1234)
        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        cell = self._return_cell(self.holding_potential, 'generic', mu, distribution, tau_w)
        cell, syn, noiseVec = self._make_distributed_synaptic_stimuli(cell, input_idx, weight)
        print "Starting simulation ..."
        #import ipdb; ipdb.set_trace()
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)
        self.save_neural_sim_single_input_data(cell, electrode, input_idx, mu, distribution, tau_w, weight)
        neuron.h('forall delete_section()')
        del cell, syn, noiseVec, electrode

    def _make_distributed_synaptic_stimuli(self, cell, input_sec, weight=0.001, **kwargs):

        # Define synapse parameters
        synapse_params = {
            'e': 0.,                   # reversal potential
            'syntype': 'ExpSyn',       # synapse type
            'tau': 2.,                # syn. time constant
            'weight': weight,            # syn. weight
            'record_current': False,
        }
        if input_sec == 'tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 900
        elif input_sec == 'homogeneous':
            input_pos = ['apic', 'dend']
            maxpos = 10000
            minpos = -10000
        else:
            input_pos = input_sec
            maxpos = 10000
            minpos = -10000

        num_synapses = 1000
        cell_input_idxs = cell.get_rand_idx_area_norm(section=input_pos, nidx=num_synapses,
                                                      z_min=minpos, z_max=maxpos)
        spike_trains = LFPy.inputgenerators.stationary_poisson(num_synapses, 5, cell.tstartms, cell.tstopms)
        synapses = self.set_input_spiketrain(cell, cell_input_idxs, spike_trains, synapse_params)
        #self.input_plot(cell, cell_input_idxs, spike_trains)

        return cell, synapses, None

    def input_plot(self, cell, cell_input_idxs, spike_trains):

        plt.close('all')
        plt.subplot(121)
        plt.plot(cell.xmid, cell.zmid, 'ko')
        plt.plot(cell.xmid[cell_input_idxs], cell.zmid[cell_input_idxs], 'r.')
        plt.axis('equal')
        plt.subplot(122)
        [plt.plot(spike_trains[idx], np.ones(len(spike_trains[idx])) * idx, 'k.') for idx in xrange(len(spike_trains))]
        plt.show()

    def set_input_spiketrain(self, cell, cell_input_idxs, spike_trains, synapse_params):
        synapse_list = []
        for number, comp_idx in enumerate(cell_input_idxs):
            synapse_params.update({'idx': int(comp_idx)})
            s = LFPy.Synapse(cell, **synapse_params)
            s.set_spike_times(spike_trains[number])
            synapse_list.append(s)
        return synapse_list


    def calculate_total_conductance(self, distribution):
        cell = self._return_cell(self.holding_potential, 'generic', 0, distribution, 1)
        total_conductance = 0
        for sec in cell.allseclist:
            for seg in sec:
                # Never mind the units, as long as it is consistent
                total_conductance += nrn.area(seg.x) * seg.gm_QA
        print distribution, total_conductance

    def _run_multiple_wn_simulation(self, mu, input_idxs, distribution, tau_w, weight):
        plt.seed(1234)
        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        neuron.h('forall delete_section()')
        cell = self._return_cell(self.holding_potential, 'generic', mu, distribution, tau_w)
        syns = []
        noise_vecs = []
        if not len(input_idxs) == 2:
            raise RuntimeError("Unusable weight procedure")
        syn_weights = [weight, 1]
        # if np.abs(np.sum(syn_weights) - 1) > 1e-12:
        #     raise RuntimeError("Weights doesn't sum to one")
        for numb, input_idx in enumerate(input_idxs):
            cell, syn, noise_vec = self._make_white_noise_stimuli(cell, input_idx, syn_weights[numb])
            syns.append(syn)
            noise_vecs.append(noise_vec)

        print "Starting simulation ..."
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)
        self.save_neural_sim_single_input_data(cell, electrode, input_idxs, mu, distribution, tau_w, weight)

    def _plot_batch_of_EC_signals(self, fig, input_idx, distributions, tau_w):

        num_cols = len(distributions) + 1
        tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))

        lines = []
        line_names = []
        ax_dict = {'xlim': [1, 500], 'ylim': [1e-7, 1e-4]}
        for col_numb, dist in enumerate(distributions):

            ax_1 = fig.add_subplot(3, num_cols, 0 * num_cols + col_numb + 2, title=dist, **ax_dict)
            ax_2 = fig.add_subplot(3, num_cols, 1 * num_cols + col_numb + 2, **ax_dict)
            ax_3 = fig.add_subplot(3, num_cols, 2 * num_cols + col_numb + 2, **ax_dict)
            for mu in self.mus:
                sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, mu,
                                                   self.holding_potential, dist, tau_w)
                LFP = np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))
                lines.append(plt.plot(0, 0, color=self.mu_clr[mu], lw=2)[0])
                line_names.append('$\mu_{factor} = %1.1f$' % mu)
                self._plot_sig_to_axes([ax_3, ax_2, ax_1], LFP, tvec, mu)
            for ax in [ax_1, ax_2, ax_3]:
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True)

            simplify_axes([ax_1, ax_2, ax_3])
        mark_subplots(fig.axes)
        fig.legend(lines[:3], line_names[:3], frameon=False, ncol=3, loc='lower right')

    def _plot_multiple_input_EC_signals(self, fig, input_idxs, distributions, tau_w):

        num_cols = len(distributions) + 1
        tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))

        lines = []
        line_names = []
        ax_dict = {'xlim': [1, 500], 'ylim': [1e-8, 1e-5]}
        for col_numb, dist in enumerate(distributions):

            ax_1 = fig.add_subplot(3, num_cols, 0 * num_cols + col_numb + 2, title=dist, **ax_dict)
            ax_2 = fig.add_subplot(3, num_cols, 1 * num_cols + col_numb + 2, **ax_dict)
            ax_3 = fig.add_subplot(3, num_cols, 2 * num_cols + col_numb + 2, **ax_dict)
            for mu in self.mus:
                sim_name = '%s_%s_multiple_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, mu,
                                                                  self.holding_potential, dist, tau_w)
                LFP = np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))
                lines.append(plt.plot(0, 0, color=self.mu_clr[mu], lw=2)[0])
                line_names.append('$\mu_{factor} = %1.1f$' % mu)
                self._plot_sig_to_axes([ax_3, ax_2, ax_1], LFP, tvec, mu)
            for ax in [ax_1, ax_2, ax_3]:
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True)

            simplify_axes([ax_1, ax_2, ax_3])
        mark_subplots(fig.axes)
        fig.legend(lines[:3], line_names[:3], frameon=False, ncol=3, loc='lower right')

    def combine_extracellular_traces(self, input_idx):
        plt.close('all')
        distributions = ['uniform', 'linear_decrease', 'linear_increase']
        tau_w = 100
        fig = plt.figure(figsize=[12, 8])
        fig.subplots_adjust(hspace=0.5, wspace=0.7, top=0.9, bottom=0.13,
                            left=0.1, right=0.95)

        self._draw_setup_to_axis(fig, input_idx, plotpos=141)
        self._plot_batch_of_EC_signals(fig, input_idx, distributions, tau_w)
        filename = ('extracellular_combined_%s_%s_%d_%1.2f' % (self.cell_name, self.input_type,
                                                    input_idx, tau_w))
        fig.savefig(join(self.figure_folder, '%s.png' % filename))

    def plot_multiple_input_EC_signals(self, tau_w):
        print "Plotting tau_w: ", tau_w
        plt.close('all')
        distributions = ['uniform', 'linear_decrease', 'linear_increase']
        fig = plt.figure(figsize=[12, 8])
        fig.subplots_adjust(hspace=0.5, wspace=0.7, top=0.9, bottom=0.13,
                            left=0.1, right=0.95)
        input_idxs = [0, 455, 605]
        self._draw_setup_to_axis(fig, input_idxs, plotpos=141)
        self._plot_multiple_input_EC_signals(fig, input_idxs, distributions, tau_w)
        filename = ('multiple_input_%s_%s_%1.2f' % (self.cell_name, self.input_type, tau_w))
        fig.savefig(join(self.figure_folder, '%s.png' % filename))

    def run_all_multiple_input_simulations(self):
        distributions = ['linear_increase']#'uniform']#, 'linear_decrease', 'linear_increase']
        input_idxs = [0, 605]
        tau_ws = [10, 100, 1, 0.1]
        weights = np.linspace(0, 1, 5)
        for tau_w in tau_ws[::-1]:
            for distribution in distributions:
                for weight in weights:
                    for mu in self.mus:
                        self._run_multiple_wn_simulation(mu, input_idxs, distribution, tau_w, weight)
            # self.plot_multiple_input_EC_signals(tau_w)

    def plot_summaries(self):
        distributions = ['uniform', 'linear_decrease', 'linear_increase']
        taums = [1, 0.1, 10, 100]
        input_idxs = [0, 605, 455]
        for distribution in distributions:
            for taum in taums:
                for input_idx in input_idxs:
                    self.plot_summary(input_idx, distribution, taum)

    def run_all_single_simulations(self):
        distributions = ['linear_increase', 'linear_decrease', 'uniform']
        input_idxs = [605, 0]
        tau_ws = [10]
        make_summary_plot = True
        tot_sims = len(input_idxs) * len(tau_ws) * len(distributions) * len(self.mus)
        i = 1
        for distribution in distributions:
            for input_idx in input_idxs:
                for tau_w in tau_ws:
                    for mu in self.mus:
                        print "%d / %d" % (i, tot_sims)
                        self._single_neural_sim_function(mu, input_idx, distribution, tau_w)
                        i += 1
                    if make_summary_plot:
                        self._plot_LFP_with_distance(distribution, tau_w, input_idx)
                        # self.plot_summary(input_idx, distribution, tau_w)
                # self._plot_q_value(distribution, input_idx)

    def LFP_with_distance_study(self):
        self._set_extended_electrode()
        weights = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        for tau_w in [30]: #, 1, 100]:
            for distribution in ['linear_increase']:#, 'uniform', 'linear_decrease']:
                for weight in weights:
                    for input_idx in ['tuft']: #self.cell_plot_idxs[:1]:
                        print tau_w, distribution, input_idx
                        self._plot_LFP_with_distance(distribution, tau_w, input_idx)
                    # self._plot_LFP_with_distance(distribution, tau_w, self.cell_plot_idxs[::2], weight)

    def q_value_study(self):
        self._set_extended_electrode()
        weights = np.linspace(0, 1, 5)
        for distribution in ['linear_increase', 'linear_decrease', 'uniform']:
            for input_idx in self.cell_plot_idxs:
                print distribution, input_idx
                self._plot_q_value(distribution, input_idx)
            # for weight in weights:
            #     self._plot_q_value(distribution, self.cell_plot_idxs[::2], weight)

    def plot_original_distance_study(self, input_idx):
        self._plot_LFP_with_distance_active(input_idx)

    def _draw_membrane_signals_to_axes_q_value_colorplot(self, fig, input_idx, distribution, tau_w):

        ax_imem = fig.add_subplot(161, title='Transmembrane current')
        ax_vmem = fig.add_subplot(162, title='Membrane potential')

        vmin = 1.
        vmax = 3.5
        q_clr = lambda q: plt.cm.jet(int(256. * (q - vmin) / (vmax - vmin)))

        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name))
        xstart = np.load(join(self.sim_folder, 'xstart_%s_%s.npy' % (self.cell_name, self.conductance)))
        zstart = np.load(join(self.sim_folder, 'zstart_%s_%s.npy' % (self.cell_name, self.conductance)))
        xend = np.load(join(self.sim_folder, 'xend_%s_%s.npy' % (self.cell_name, self.conductance)))
        zend = np.load(join(self.sim_folder, 'zend_%s_%s.npy' % (self.cell_name, self.conductance)))
        xmid = np.load(join(self.sim_folder, 'xmid_%s_%s.npy' % (self.cell_name, self.conductance)))
        zmid = np.load(join(self.sim_folder, 'zmid_%s_%s.npy' % (self.cell_name, self.conductance)))
        if self.conductance is 'active':
            sim_name = '%s_%s_%d_%+d_active' % (self.cell_name, self.input_type, input_idx, self.holding_potential)
        else:
            sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, 2.0,
                                                        self.holding_potential, distribution, tau_w)

        vmem = np.load(join(self.sim_folder, 'vmem_%s.npy' % sim_name))
        imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))
        # freqs, vmem_psd = aLFP.return_freq_and_psd(self.timeres_python/1000., vmem[:, :])
        # freqs, imem_psd = aLFP.return_freq_and_psd(self.timeres_python/1000., imem[:, :])
        freqs, vmem_psd = aLFP.return_freq_and_psd_welch(vmem[:, :], self.welch_dict)
        freqs, imem_psd = aLFP.return_freq_and_psd_welch(imem[:, :], self.welch_dict)

        dc_vmem = vmem_psd[:, 1]
        dc_imem = imem_psd[:, 1]
        max_value_vmem = np.max(vmem_psd[:, 1:], axis=1)
        max_value_imem = np.max(imem_psd[:, 1:], axis=1)
        q_vmem = max_value_vmem / dc_vmem
        q_imem = max_value_imem / dc_imem

        ax_imem.plot(xmid[input_idx], zmid[input_idx], 'y*', zorder=1, ms=15)
        ax_vmem.plot(xmid[input_idx], zmid[input_idx], 'y*', zorder=1, ms=15)

        # mark_subplots([ax_imem, ax_vmem], 'ab', xpos=0, ypos=1)

        [ax_imem.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2, color=q_clr(q_imem[idx]), zorder=0)
        for idx in xrange(len(xmid))]

        [ax_vmem.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2, color=q_clr(q_vmem[idx]), zorder=0)
         for idx in xrange(len(xmid))]

        # ax.plot(xmid[0], zmid[0], 'o', color=sec_clrs[0], zorder=0, ms=10, mec='none')
        # ax.plot(xmid[self.cell_plot_idxs], zmid[self.cell_plot_idxs], 'bD', zorder=2, ms=5, mec='none')
        # ax.plot(xmid[self.cell_plot_idxs], zmid[self.cell_plot_idxs], 'D', color='m', zorder=0, ms=10, mec='none')

        ax_imem.scatter(elec_x[self.short_list_elecs], elec_z[self.short_list_elecs], c='g', edgecolor='none', s=50)
        ax_vmem.scatter(elec_x[self.short_list_elecs], elec_z[self.short_list_elecs], c='g', edgecolor='none', s=50)

        ax_imem.axis('off')
        ax_vmem.axis('off')

    def _draw_elecs_q_value_colorplot(self, fig, input_idx, distribution, tau_w):

        amp_ax = fig.add_subplot(231, title='Max amplitude')
        dc_ax = fig.add_subplot(232)
        res_ax = fig.add_subplot(233)

        freq_ax = fig.add_subplot(234)#axes([0.01, 0.1, 0.4, 0.8])
        q_ax = fig.add_subplot(235)#axes([0.45, 0.1, 0.4, 0.8])
        q_res_ax = fig.add_subplot(236)#axes([0.45, 0.1, 0.4, 0.8])

        mark_subplots(fig.axes)

        input_name_dict = {605: 'Apical', 0: 'Somatic', 455: 'Middle'}
        fig.suptitle("Input: %s, Distribution: %s, tau: %1.2f" %(input_name_dict[input_idx], distribution, tau_w))
        if self.conductance is 'active':
            sim_name = '%s_%s_%d_%+d_active' % (self.cell_name, self.input_type, input_idx, self.holding_potential)
        else:
            sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, 2.0,
                                                        self.holding_potential, distribution, tau_w)
        distances = np.linspace(10, 4000, 39)
        heights = np.linspace(1200, -200, 29)
        elec_x, elec_z = np.meshgrid(distances, heights)
        elec_x = elec_x.flatten()
        elec_z = elec_z.flatten()
        elec_y = np.zeros(len(elec_z))

        electrode_parameters = {
                'sigma': 0.3,
                'x': elec_x,
                'y': elec_y,
                'z': elec_z
        }
        LFP_psd = np.load(join(self.sim_folder, 'LFP_psd_%s.npy' % sim_name))
        freqs = np.load(join(self.sim_folder, 'LFP_freq_%s.npy' % sim_name))

        # if 1:
        #    print "Recalculating extracellular potential"
        #    cell = self._return_cell(self.holding_potential, 'generic', 2.0, distribution, tau_w)
        #    cell.tstartms = 0
        #    cell.tstopms = 1
        #    cell.simulate(rec_imem=True)
        #    cell.imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))
        #    cell.tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))
        #    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
        #    electrode.calc_lfp()
        #    LFP = electrode.LFP
        #else:
        #    LFP = np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))
        # freqs, LFP_psd = aLFP.return_freq_and_psd(self.timeres_python/1000., LFP)
        #print "PSD calculations"
        # freqs, LFP_psd = aLFP.return_freq_and_psd_welch(LFP, self.welch_dict)
        print sim_name, freqs, len(freqs)
        np.save(join(self.sim_folder, 'LFP_psd_%s.npy' % sim_name), LFP_psd)
        np.save(join(self.sim_folder, 'LFP_freq_%s.npy' % sim_name), freqs)
        upper_idx_limit = np.argmin(np.abs(freqs - 500))
        res_idx = np.argmin(np.abs(freqs - 57.6))

        num_elec_cols = len(set(elec_x))
        num_elec_rows = len(set(elec_z))
        dc = np.zeros((num_elec_rows, num_elec_cols))
        res_amp = np.zeros((num_elec_rows, num_elec_cols))
        max_value = np.zeros((num_elec_rows, num_elec_cols))
        freq_at_max = np.zeros((num_elec_rows, num_elec_cols))

        for elec in xrange(len(elec_z)):
            num_elec_cols = len(set(elec_x))
            num_elec_rows = len(set(elec_z))
            elec_idxs = np.arange(len(elec_x)).reshape(num_elec_rows, num_elec_cols)
            row, col = np.array(np.where(elec_idxs == elec))[:, 0]
            dc[row, col] = LFP_psd[elec, 1]
            res_amp[row, col] = LFP_psd[elec, res_idx]
            max_value[row, col] = np.max(LFP_psd[elec, 1:upper_idx_limit])
            freq_at_max[row, col] = freqs[np.argmax(LFP_psd[elec, 1:upper_idx_limit])]
        freq_ax.set_title('Max frequency (Mode: %1.1f Hz)' % stats.mode(freq_at_max, axis=None)[0][0])
        dc_ax.set_title('Amplitude at %1.2f Hz' % freqs[1])
        res_ax.set_title('Amplitude at %1.2f Hz' % freqs[res_idx])
        q_ax.set_title('q-value amp(max freq) / amp(%1.2f)' % freqs[1])
        q_res_ax.set_title('q-value amp(%1.2f) / amp(%1.2f)' % (freqs[res_idx], freqs[1]))
        q = max_value / dc
        img_freq = freq_ax.imshow(freq_at_max, extent=[np.min(distances), np.max(distances), np.min(heights), np.max(heights)],
                        vmin=1, vmax=500., aspect=2, norm=LogNorm())
        img_q = q_ax.imshow(q[:, :], extent=[np.min(distances), np.max(distances), np.min(heights), np.max(heights)],
                        vmin=1, vmax=7., aspect=2)
        img_q_res = q_res_ax.imshow(res_amp/dc, extent=[np.min(distances), np.max(distances), np.min(heights), np.max(heights)],
                        vmin=1, vmax=7., aspect=2)
        img_amp = amp_ax.imshow(max_value[:, :], extent=[np.min(distances), np.max(distances), np.min(heights), np.max(heights)],
                        aspect=2, norm=LogNorm(), vmin=1e-10, vmax=1e-5)
        img_dc = dc_ax.imshow(dc[:, :], extent=[np.min(distances), np.max(distances), np.min(heights), np.max(heights)],
                        aspect=2, norm=LogNorm(), vmin=1e-10, vmax=1e-5)
        img_res = res_ax.imshow(res_amp[:, :], extent=[np.min(distances), np.max(distances), np.min(heights), np.max(heights)],
                        aspect=2, norm=LogNorm(), vmin=1e-10, vmax=1e-5)

        plt.colorbar(img_q, ax=q_ax)
        plt.colorbar(img_q_res, ax=q_res_ax)
        plt.colorbar(img_freq, ax=freq_ax)
        plt.colorbar(img_amp, ax=amp_ax)
        plt.colorbar(img_res, ax=res_ax)
        plt.colorbar(img_dc, ax=dc_ax)

    def _q_value_study_colorplot(self, input_idx, distribution=None, tau_w=None):
        plt.close('all')
        fig = plt.figure(figsize=[16, 12])

        fig.subplots_adjust(hspace=0.5, wspace=0.5, top=0.9, bottom=0.13, left=0.04, right=0.98)
        # self._draw_membrane_signals_to_axes_q_value_colorplot(fig, input_idx, distribution, tau_w)
        self._draw_elecs_q_value_colorplot(fig, input_idx, distribution, tau_w)
        if self.conductance is 'active':
            filename = ('color_q_value_%s_%d_%s' % (self.cell_name, input_idx, self.conductance))
        else:
            filename = ('color_q_value_%s_%d_%s_%s_%1.2f' %
                        (self.cell_name, input_idx, self.conductance, distribution, tau_w))
        fig.savefig(join(self.figure_folder, 'q_value', '%s_freq.png' % filename), dpi=150)

    def active_q_values_colorplot(self):
        for input_idx in [0, 370, 415, 514, 717, 743, 762, 827, 915, 957]:
            self._q_value_study_colorplot(input_idx)

    def generic_q_values_colorplot(self):
        for tau_w in [10, 1.0, 100]:
            for distribution in ['linear_increase', 'linear_decrease', 'uniform']:
                for input_idx in [605, 0]:
                    print distribution, input_idx, tau_w
                    self._q_value_study_colorplot(input_idx, distribution, tau_w)
                    sys.exit()

    def test_original_hay(self, input_idx):
        import neuron
        from hay_active_declarations import active_declarations as hay_active
        import LFPy
        plt.seed(1234)

        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        neuron.h('forall delete_section()')

        neuron_models = join(self.root_folder, 'neuron_models')
        neuron.load_mechanisms(join(neuron_models))

        neuron.load_mechanisms(join(neuron_models, 'hay', 'mod'))
        cell_params = {
            'morphology': join(neuron_models, 'hay', 'lfpy_version', 'morphologies', 'cell1.hoc'),
            'v_init': self.holding_potential,
            'passive': False,           # switch on passive mechs
            'nsegs_method': 'lambda_f',  # method for setting number of segments,
            'lambda_f': 100,           # segments are isopotential at this frequency
            'timeres_NEURON': self.timeres_NEURON,   # dt of LFP and NEURON simulation.
            'timeres_python': self.timeres_python,
            'tstartms': -self.cut_off,          # start time, recorders start at t=0
            'tstopms': self.end_t,
            'custom_code': [join(neuron_models, 'hay', 'lfpy_version', 'custom_codes.hoc')],
            'custom_fun': [hay_active],  # will execute this function
            'custom_fun_args': [{'conductance_type': self.conductance,
                                 'hold_potential': self.holding_potential}]
        }
        cell = LFPy.Cell(**cell_params)
        # self._quickplot_setup(cell, electrode, input_idx)
        cell, syn, noiseVec = self._make_white_noise_stimuli(cell, input_idx)
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)

        sim_name = '%s_%s_%d_%+d_%s' % (self.cell_name, self.input_type, input_idx,
                                        self.holding_potential, self.conductance)
        # np.save(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)), cell.tvec)
        np.save(join(self.sim_folder, 'sig_%s.npy' % sim_name), electrode.LFP)
        np.save(join(self.sim_folder, 'vmem_%s.npy' % sim_name), cell.vmem)
        np.save(join(self.sim_folder, 'imem_%s.npy' % sim_name), cell.imem)

        np.save(join(self.sim_folder, 'xstart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xstart)
        np.save(join(self.sim_folder, 'ystart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.ystart)
        np.save(join(self.sim_folder, 'zstart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zstart)
        np.save(join(self.sim_folder, 'xend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xend)
        np.save(join(self.sim_folder, 'yend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.yend)
        np.save(join(self.sim_folder, 'zend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zend)
        np.save(join(self.sim_folder, 'xmid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xmid)
        np.save(join(self.sim_folder, 'ymid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.ymid)
        np.save(join(self.sim_folder, 'zmid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zmid)
        np.save(join(self.sim_folder, 'diam_%s_%s.npy' % (self.cell_name, self.conductance)), cell.diam)

    def test_original_zuchkova(self, input_idx):
        import neuron
        from hay_active_declarations import active_declarations as hay_active
        import LFPy
        plt.seed(1234)

        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        neuron.h('forall delete_section()')

        neuron_models = join(self.root_folder, 'neuron_models')
        neuron.load_mechanisms(join(neuron_models))

        neuron.load_mechanisms(join(neuron_models, 'hay', 'mod'))
        cell_params = {
            'morphology': join(neuron_models, 'hay', 'lfpy_version', 'morphologies', 'cell1.hoc'),
            'v_init': self.holding_potential,
            'passive': False,           # switch on passive mechs
            'nsegs_method': 'lambda_f',  # method for setting number of segments,
            'lambda_f': 100,           # segments are isopotential at this frequency
            'timeres_NEURON': self.timeres_NEURON,   # dt of LFP and NEURON simulation.
            'timeres_python': self.timeres_python,
            'tstartms': -self.cut_off,          # start time, recorders start at t=0
            'tstopms': self.end_t,
            'custom_code': [join(neuron_models, 'hay', 'lfpy_version', 'custom_codes.hoc')],
            'custom_fun': [hay_active],  # will execute this function
            'custom_fun_args': [{'conductance_type': 'zuchkova',
                                 'hold_potential': self.holding_potential}]
        }
        cell = LFPy.Cell(**cell_params)
        # self._quickplot_setup(cell, electrode, input_idx)
        cell, syn, noiseVec = self._make_white_noise_stimuli(cell, input_idx)

        comp = 0
        nrn.distance()
        for sec in cell.allseclist:
            for seg in sec:
                print nrn.distance(seg.x), sec.name(), comp
                comp += 1

        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)

        sim_name = '%s_%s_%d_%+d_%s' % (self.cell_name, self.input_type, input_idx,
                                        self.holding_potential, self.conductance)
        np.save(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)), cell.tvec)
        np.save(join(self.sim_folder, 'sig_%s.npy' % sim_name), electrode.LFP)
        np.save(join(self.sim_folder, 'vmem_%s.npy' % sim_name), cell.vmem)
        np.save(join(self.sim_folder, 'imem_%s.npy' % sim_name), cell.imem)

        np.save(join(self.sim_folder, 'xstart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xstart)
        np.save(join(self.sim_folder, 'ystart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.ystart)
        np.save(join(self.sim_folder, 'zstart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zstart)
        np.save(join(self.sim_folder, 'xend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xend)
        np.save(join(self.sim_folder, 'yend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.yend)
        np.save(join(self.sim_folder, 'zend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zend)
        np.save(join(self.sim_folder, 'xmid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xmid)
        np.save(join(self.sim_folder, 'ymid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.ymid)
        np.save(join(self.sim_folder, 'zmid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zmid)
        np.save(join(self.sim_folder, 'diam_%s_%s.npy' % (self.cell_name, self.conductance)), cell.diam)
        np.save(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name), electrode.x)
        np.save(join(self.sim_folder, 'elec_y_%s.npy' % self.cell_name), electrode.y)
        np.save(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name), electrode.z)

    def test_original_hu(self, input_idx):
        import neuron
        from ca1_sub_declarations import active_declarations as ca1_active
        import LFPy

        plt.seed(1234)

        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        neuron.h('forall delete_section()')

        neuron_models = join(self.root_folder, 'neuron_models')
        neuron.load_mechanisms(join(neuron_models))

        use_channels = ['Ih', 'Im', 'INaP']

        neuron.load_mechanisms(join(neuron_models, 'ca1_sub'))
        cell_params = {
                'morphology': join(neuron_models, 'ca1_sub', self.cell_name, '%s.hoc' % self.cell_name),
                'v_init': self.holding_potential,             # initial crossmembrane potential
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
                                     'hold_potential': self.holding_potential}],
                }
        cell = LFPy.Cell(**cell_params)
        # self._quickplot_setup(cell, electrode, input_idx)
        cell, syn, noiseVec = self._make_white_noise_stimuli(cell, input_idx)
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)

        sim_name = '%s_%s_%d_%+d_active' % (self.cell_name, self.input_type, input_idx, self.holding_potential)
        np.save(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)), cell.tvec)
        # np.save(join(self.sim_folder, 'sig_%s.npy' % sim_name), electrode.LFP)
        np.save(join(self.sim_folder, 'sig_%s.npy' % sim_name), np.dot(electrode.electrodecoeff, cell.imem))
        np.save(join(self.sim_folder, 'vmem_%s.npy' % sim_name), cell.vmem)
        np.save(join(self.sim_folder, 'imem_%s.npy' % sim_name), cell.imem)

        np.save(join(self.sim_folder, 'xstart_%s_%s.npy' % (self.cell_name, 'active')), cell.xstart)
        np.save(join(self.sim_folder, 'ystart_%s_%s.npy' % (self.cell_name, 'active')), cell.ystart)
        np.save(join(self.sim_folder, 'zstart_%s_%s.npy' % (self.cell_name, 'active')), cell.zstart)
        np.save(join(self.sim_folder, 'xend_%s_%s.npy' % (self.cell_name, 'active')), cell.xend)
        np.save(join(self.sim_folder, 'yend_%s_%s.npy' % (self.cell_name, 'active')), cell.yend)
        np.save(join(self.sim_folder, 'zend_%s_%s.npy' % (self.cell_name, 'active')), cell.zend)
        np.save(join(self.sim_folder, 'xmid_%s_%s.npy' % (self.cell_name, 'active')), cell.xmid)
        np.save(join(self.sim_folder, 'ymid_%s_%s.npy' % (self.cell_name, 'active')), cell.ymid)
        np.save(join(self.sim_folder, 'zmid_%s_%s.npy' % (self.cell_name, 'active')), cell.zmid)
        np.save(join(self.sim_folder, 'diam_%s_%s.npy' % (self.cell_name, 'active')), cell.diam)
        np.save(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name), electrode.x)
        np.save(join(self.sim_folder, 'elec_y_%s.npy' % self.cell_name), electrode.y)
        np.save(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name), electrode.z)

    def test_original_hay_simple_ratio(self, input_idx):
        """ Will compare stady state input to resonant frequencies to compare to other results
        """
        import neuron
        from hay_active_declarations import active_declarations as hay_active
        import LFPy
        plt.seed(1234)

        neuron.h('forall delete_section()')

        neuron_models = join(self.root_folder, 'neuron_models')
        neuron.load_mechanisms(join(neuron_models))

        input_amp = 0.02
        input_freq = 10.

        neuron.load_mechanisms(join(neuron_models, 'hay', 'mod'))
        cell_params = {
            'morphology': join(neuron_models, 'hay', 'lfpy_version', 'morphologies', 'cell1.hoc'),
            'v_init': self.holding_potential,
            'passive': False,
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
            'timeres_NEURON': self.timeres_NEURON,
            'timeres_python': self.timeres_python,
            'tstartms': -6000,          # start time, recorders start at t=0
            'tstopms': 1000,
            'custom_code': [join(neuron_models, 'hay', 'lfpy_version', 'custom_codes.hoc')],
            'custom_fun': [hay_active],  # will execute this function
            'custom_fun_args': [{'conductance_type': self.conductance,
                                 'hold_potential': self.holding_potential}]
        }
        steady_stim_params = {
            'idx': input_idx,
            'record_current': True,
            'pptype': 'IClamp',
            'amp': input_amp,
            'dur': 1e9,
            'delay': 10,
        }
        sin_stim_params = {
            'idx': input_idx,
            'record_current': True,
            'dur': 10000.,
            'delay': 10,
            'freq': input_freq,
            'phase': 0,
            'pkamp': input_amp,
            'pptype': 'SinIClamp',
        }
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.2)

        cell = LFPy.Cell(**cell_params)
        stim = LFPy.StimIntElectrode(cell, **steady_stim_params)
        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)
        # plt.plot(cell.tvec, cell.vmem[input_idx, :])
        plt.plot(cell.tvec, cell.vmem[0, :] - cell.vmem[0, 0])
        steady_input_imp = (cell.vmem[0, -1] - cell.vmem[0, 0]) / input_amp

        cell = LFPy.Cell(**cell_params)
        stim = LFPy.StimIntElectrode(cell, **sin_stim_params)
        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)
        # plt.plot(cell.tvec, cell.vmem[input_idx, :])
        plt.plot(cell.tvec, cell.vmem[0, :] - cell.vmem[0, 0])
        sin_input_imp = (np.max(cell.vmem[0, cell.vmem.shape[1]/2:]) - cell.vmem[0, 0]) / input_amp

        q_value = sin_input_imp / steady_input_imp

        print steady_input_imp, sin_input_imp, q_value
        plt.xlabel('Time')
        plt.ylabel('Voltage deflection in soma')
        plt.title('Input in distal apical dendrite at %d mV. Q-value %1.2f' %
                  (self.holding_potential, q_value))
        plt.savefig(join(self.root_folder, 'Vm_deflection_control_%dmV_%dHz_%s.png' %
                                           (self.holding_potential, input_freq, self.conductance)))

        # sim_name = '%s_%s_%d_%+d_%s' % (self.cell_name, 'stationary', input_idx,
        #                                 self.holding_potential, self.conductance)

        # np.save(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)), cell.tvec)
        # np.save(join(self.sim_folder, 'sig_%s.npy' % sim_name), electrode.LFP)
        # np.save(join(self.sim_folder, 'vmem_%s.npy' % sim_name), cell.vmem)
        # np.save(join(self.sim_folder, 'imem_%s.npy' % sim_name), cell.imem)
        # np.save(join(self.sim_folder, 'xstart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xstart)
        # np.save(join(self.sim_folder, 'ystart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.ystart)
        # np.save(join(self.sim_folder, 'zstart_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zstart)
        # np.save(join(self.sim_folder, 'xend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xend)
        # np.save(join(self.sim_folder, 'yend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.yend)
        # np.save(join(self.sim_folder, 'zend_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zend)
        # np.save(join(self.sim_folder, 'xmid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.xmid)
        # np.save(join(self.sim_folder, 'ymid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.ymid)
        # np.save(join(self.sim_folder, 'zmid_%s_%s.npy' % (self.cell_name, self.conductance)), cell.zmid)
        # np.save(join(self.sim_folder, 'diam_%s_%s.npy' % (self.cell_name, self.conductance)), cell.diam)

    def all_resonance_plots(self):
        for rotation in np.linspace(0, 2*np.pi, 24):
            self.one_resonance_plot(rotation)

    def _return_LFP_from_mu(self, rotation, distribution, tau_w, electrode_parameters, input_idx, mu):
        sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, mu,
                                                    self.holding_potential, distribution, tau_w)
        neuron.h('forall delete_section()')
        cell = self._return_cell(self.holding_potential, 'generic', 0, distribution, tau_w)
        cell.tstartms = 0
        cell.tstopms = 1
        cell.simulate(rec_imem=True)
        cell.imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))
        cell.tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))
        cell.set_rotation(z=rotation)
        electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
        electrode.calc_lfp()
        LFP = electrode.LFP
        return aLFP.return_freq_and_psd(cell.tvec[1] / 1000., LFP)

    def one_resonance_plot(self, rotation):
        distribution = 'uniform'
        tau_w = 10
        input_idx = 0

        distances = np.linspace(-200, 200, 5)
        heights = np.linspace(1000, -150, 5)
        elec_x, elec_z = np.meshgrid(distances, heights)
        elec_x = elec_x.flatten()
        elec_z = elec_z.flatten()
        elec_y = np.ones(len(elec_z)) * 20

        electrode_parameters = {
                'sigma': 0.3,
                'x': elec_x,
                'y': elec_y,
                'z': elec_z
        }
        freqs, LFP_passive = self._return_LFP_from_mu(rotation, distribution, tau_w, electrode_parameters, input_idx, 0)
        freqs, LFP_restor = self._return_LFP_from_mu(rotation, distribution, tau_w, electrode_parameters, input_idx, 2)

        cell = self._return_cell(self.holding_potential, 'generic', 0, distribution, tau_w)
        cell.set_rotation(z=rotation)
        plt.close('all')
        fig = plt.figure(figsize=[10, 10])
        fig.suptitle('Input: %s  Distribution: %s  tau_w: %1.1f ms' % ('Soma' if input_idx == 0 else 'Apical',
                                                                   distribution, tau_w))
        fig.subplots_adjust(left=0.01, right=0.98)
        ax_morph = fig.add_subplot(1, len(distances) + 1, 1)
        ax_morph.plot(cell.xmid[input_idx], cell.zmid[input_idx], 'y*', zorder=1, ms=15)

        [ax_morph.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], lw=2,
                 color='gray', zorder=0) for idx in xrange(len(cell.xmid))]
        ax_morph.plot(cell.xmid[0], cell.zmid[0], 'o', color='gray', zorder=0, ms=10, mec='none')

        ax_morph.scatter(elec_x, elec_z, c='g', edgecolor='none', s=50)
        ax_morph.axis('off')
        for elec in xrange(len(elec_x)):
            row = np.argmin(np.abs(distances - elec_x[elec]))
            col = np.argmin(np.abs(heights - elec_z[elec]))
            plotnum = row + col * (len(distances) + 1) + 2

            ax = fig.add_subplot(len(heights), len(distances) + 1, plotnum, xticklabels=[], yticklabels=[],
                                 xlim=[1, 500], ylim=[1e-8, 1e-6])
            # ax.set_title('%d %d' % (elec_x[elec], elec_z[elec]))
            simplify_axes(ax)
            ax.grid(True)
            ax.loglog(freqs, LFP_restor[elec], 'r', lw=2)
            ax.loglog(freqs, LFP_passive[elec], 'b', lw=2)

        l1, = plt.plot(0, 0, 'r', lw=2)
        l2, = plt.plot(0, 0, 'b', lw=2)
        fig.legend([l2, l1], ['Passive', 'Restorative'], frameon=False, ncol=2, loc='lower center')
        fig.savefig(join(self.figure_folder, 'resonance_test_%s_%d_%1.5f.png' % (distribution, input_idx, rotation)))


if __name__ == '__main__':

    gs = GenericStudy('hay', 'distributed_synaptic', conductance='generic', extended_electrode=True)

    # gs.all_resonance_plots()
    # gs.run_all_multiple_input_simulations()
    # gs.generic_q_values_colorplot()
    # gs.q_value_study()
    # gs.active_q_values_colorplot()
    # gs.run_all_single_simulations()
    if len(sys.argv[1]) >= 2:
        gs.run_all_distributed_synaptic_input_simulations(float(sys.argv[1]), float(sys.argv[2]))
    else:
        gs.LFP_with_distance_study()
