#!/usr/bin/env python
import os
import sys
from os.path import join
import random
if not 'DISPLAY' in os.environ:
    import matplotlib
    matplotlib.use('Agg')
    at_stallo = True
else:
    at_stallo = False
import numpy as np
import pylab as plt
import neuron
import LFPy
import aLFP
import scipy.fftpack as ff
import scipy.signal

model = 'hay'
if at_stallo:
    neuron_model = join('/home', 'torbness', 'work', 'aLFP', 'neuron_models', model)
else:
    neuron_model = join('/home', 'torbjone', 'work', 'aLFP', 'neuron_models', model)
neuron.load_mechanisms(join(neuron_model, 'mod'))
neuron.load_mechanisms(join(neuron_model, '..'))
from hay_active_declarations import active_declarations as hay_active

class Population():

    def __init__(self, holding_potential=-80, conductance_type='active', correlation=0.0,
                 weight=0.0001, input_region='homogeneous', distribution='linear_increase', initialize=False):

        self.model = 'hay'
        self.sim_name = 'hay_generic' #
        self.conductance_clr = {'active': 'r', 'passive': 'k', 'Ih_linearized': 'g',
                                -0.5: 'r', 0.0: 'k', 2.0: 'b'}

        if at_stallo:
            self.neuron_model = join('/home', 'torbness', 'work', 'aLFP', 'neuron_models', self.model)
            self.fig_folder = join('/global', 'work', 'torbness', 'aLFP', 'correlated_population')
            self.data_folder = join('/global', 'work', 'torbness', 'aLFP', 'correlated_population', self.sim_name)
        else:
            self.fig_folder = join('/home', 'torbjone', 'work', 'aLFP', 'correlated_population')
            self.data_folder = join('/home', 'torbjone', 'work', 'aLFP', 'correlated_population', self.sim_name)
            self.neuron_model = join('/home', 'torbjone', 'work', 'aLFP', 'neuron_models', self.model)

        if not os.path.isdir(self.data_folder):
            os.mkdir(self.data_folder)
        self.distribution = distribution
        self.generic_conductance = type(conductance_type) in [float, int]

        self.timeres = 2**-4
        self.cut_off = 500 if at_stallo else 500
        self.end_t = 10000 if at_stallo else 1000
        self.num_cells = 100 if at_stallo else 4

        self.population_radius = 100.
        self.dr = 25.
        self.population_sizes = np.arange(self.dr, self.population_radius + self.dr, self.dr)

        self.num_synapses = 1000
        self.correlation = correlation
        self.holding_potential = holding_potential
        self.conductance_type = conductance_type
        self.input_region = input_region
        self.weight = weight
        self.num_tsteps = round((self.end_t - 0) / self.timeres + 1)
        self.tvec = np.linspace(0, self.end_t, self.num_tsteps)

        self.stem = self.get_simulation_stem(model, input_region, conductance_type, holding_potential,
                                             correlation, weight)

        self.divide_into_welch = 4.
        self.welch_dict = {'Fs': 1000 / self.timeres,
                           'NFFT': int(self.num_tsteps/self.divide_into_welch),
                           'noverlap': int(self.num_tsteps/self.divide_into_welch/2.),
                           'window': plt.window_hanning,
                           'detrend': plt.detrend_mean,
                           'scale_by_freq': True,
                           }

        self.set_cell_params()
        self.set_input_params()
        self.initialize_electrodes()
        if initialize:
            self.distribute_cells()
            self.make_all_input_trains()
            self.plot_set_up()

    def get_simulation_stem(self, model, input_region, conductance_type, holding_potential, correlation, weight,
                            distribution=None):

        if self.generic_conductance:
            dist = self.distribution if distribution is None else distribution
            stem = '%s_%s_%1.1f_%s_c%1.2f_w%1.5f_R%d_N%d' % (model, input_region, conductance_type, dist,
                                                             correlation, weight, self.population_radius,
                                                             self.num_cells)
        else:
            stem = '%s_%s_%s_%dmV_c%1.2f_w%1.5f_R%d_N%d' % (model, input_region, conductance_type, holding_potential,
                                                            correlation, weight, self.population_radius,
                                                            self.num_cells)
        return stem

    def distribute_cells(self):
        x_y_z_rot = np.zeros((4, self.num_cells))
        for cell_idx in xrange(self.num_cells):
            x = 2 * self.population_radius * (np.random.random() - 0.5)
            y = 2 * self.population_radius * (np.random.random() - 0.5)
            while x**2 + y**2 > self.population_radius**2:
                x = 2 * self.population_radius * (np.random.random() - 0.5)
                y = 2 * self.population_radius * (np.random.random() - 0.5)
            x_y_z_rot[:2, cell_idx] = x, y

        x_y_z_rot[2, :] = 500*(np.random.random(size=self.num_cells) - 0.5)  # Ness et al. 2015
        x_y_z_rot[3, :] = 2*np.pi*np.random.random(size=self.num_cells)
        np.save(join(self.data_folder, 'x_y_z_rot_%d_%d.npy' %
                     (self.num_cells, self.population_radius)), x_y_z_rot)

    def plot_set_up(self):
        x_y_z_rot = np.load(join(self.data_folder, 'x_y_z_rot_%d_%d.npy' %
                                (self.num_cells, self.population_radius)))
        plt.close('all')
        plt.subplot(121, xlabel='x', ylabel='z', xlim=[-self.population_radius, self.population_radius], 
                            aspect=1.)
        plt.scatter(self.elec_x, self.elec_z)
        plt.scatter(x_y_z_rot[0, :], x_y_z_rot[2, :], c=x_y_z_rot[3, :],
                    edgecolor='none', s=4, cmap='hsv')
        plt.subplot(122, xlabel='x', ylabel='y', xlim=[-self.population_radius, self.population_radius], 
                            aspect=1.)
        plt.scatter(x_y_z_rot[0, :], x_y_z_rot[1, :], c=x_y_z_rot[3, :],
                    edgecolor='none', s=4, cmap='hsv')
        plt.scatter(self.elec_x, self.elec_y)
        plt.savefig(join(self.fig_folder, 'cell_distribution_%d_%d.png' %
                         (self.num_cells, self.population_radius)))
        
    def initialize_electrodes(self):

        n_elecs_center = 9
        elec_x_center = np.zeros(n_elecs_center)
        elec_y_center = np.zeros(n_elecs_center)
        elec_z_center = np.linspace(-200, 1400, n_elecs_center)

        n_elecs_lateral = 9
        elec_x_lateral = np.linspace(0, np.sqrt(self.population_radius * 10), n_elecs_lateral)**2
        elec_y_lateral = np.zeros(n_elecs_lateral)
        elec_z_lateral = np.zeros(n_elecs_lateral)

        self.elec_x = np.r_[elec_x_center, elec_x_lateral]
        self.elec_y = np.r_[elec_y_center, elec_y_lateral]
        self.elec_z = np.r_[elec_z_center, elec_z_lateral]

        self.center_idxs = np.arange(n_elecs_center)[1:-1]
        self.lateral_idxs = np.arange(n_elecs_center, n_elecs_lateral + n_elecs_center)

        self.num_elecs = len(self.elec_x)
        # np.save(join(self.folder, 'elec_x.npy'), elec_x)
        # np.save(join(self.folder, 'elec_y.npy'), elec_y)
        # np.save(join(self.folder, 'elec_z.npy'), elec_z)

        self.electrode_params = {
            'sigma': 0.3,      # extracellular conductivity
            'x': self.elec_x,  # electrode requires 1d vector of positions
            'y': self.elec_y,
            'z': self.elec_z
        }

    def set_cell_params(self):

        if self.generic_conductance:
            self.cell_params = {
                'morphology': join(self.neuron_model, 'lfpy_version', 'morphologies', 'cell1.hoc'),
                'v_init': self.holding_potential,
                'passive': False,           # switch on passive mechs
                'nsegs_method': 'lambda_f',  # method for setting number of segments,
                'lambda_f': 100,           # segments are isopotential at this frequency
                'timeres_NEURON': self.timeres,   # dt of LFP and NEURON simulation.
                'timeres_python': self.timeres,
                'tstartms': -self.cut_off,          # start time, recorders start at t=0
                'tstopms': self.end_t,
                'custom_code': [join(self.neuron_model, 'lfpy_version', 'custom_codes.hoc')],
                'custom_fun': [hay_active],  # will execute this function
                'custom_fun_args': [{'conductance_type': 'generic',
                                     'mu_factor': self.conductance_type,
                                     'g_pas': 0.00005,#0.0002, # / 5,
                                     'distribution': self.distribution,
                                     'tau_w': 'auto',
                                     #'total_w_conductance': 6.23843378791,# / 5,
                                     'avrg_w_bar': 0.00005 * 2,
                                     'hold_potential': self.holding_potential}]
            }
        else:
            self.cell_params = {
                'morphology': join(self.neuron_model, 'lfpy_version', 'morphologies', 'cell1.hoc'),
                'v_init': self.holding_potential,
                'passive': False,           # switch on passive mechs
                'nsegs_method': 'lambda_f',  # method for setting number of segments,
                'lambda_f': 100,           # segments are isopotential at this frequency
                'timeres_NEURON': self.timeres,   # dt of LFP and NEURON simulation.
                'timeres_python': self.timeres,
                'tstartms': -self.cut_off,          # start time, recorders start at t=0
                'tstopms': self.end_t,
                'custom_code': [join(self.neuron_model, 'lfpy_version', 'custom_codes.hoc')],
                'custom_fun': [hay_active],  # will execute this function
                'custom_fun_args': [{'conductance_type': self.conductance_type,
                                     'hold_potential': self.holding_potential}]
            }

    def plot_cell_to_ax(self, ax, xstart, xmid, xend, ystart, ymid, yend,
                            elec_x, elec_z, electrodes, apic_idx, elec_clr):
        [ax.plot([xstart[idx], xend[idx]], [ystart[idx], yend[idx]], color='grey')
                        for idx in xrange(len(xstart))]
        [ax.plot(elec_x[electrodes[idx]], elec_z[electrodes[idx]], 'o', color=elec_clr(idx))
                        for idx in xrange(len(electrodes))]
        ax.plot(xmid[apic_idx], ymid[apic_idx], 'g*', ms=10)
        ax.plot(xmid[0], ymid[0], 'yD')

    def synapse_pos_plot(self, cell, cell_input_idxs):
        plt.close('all')
        plt.plot(cell.xmid, cell.zmid, 'ko')
        plt.plot(cell.xmid[cell_input_idxs], cell.zmid[cell_input_idxs], 'rD')
        plt.axis('equal')
        plt.show()

    def set_input_params(self):
        if self.input_region == 'distal_tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 900
        elif self.input_region == 'tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 600
        elif self.input_region == 'homogeneous':
            input_pos = ['apic', 'dend']
            maxpos = 10000
            minpos = -10000
        elif self.input_region == 'basal':
            input_pos = ['dend']
            maxpos = 10000
            minpos = -10000
        else:
            print "Unrecognized input_region"
            raise RuntimeError("Unrecognized input_region")

        self.syn_pos_params = {'section': input_pos,
                               'nidx': self.num_synapses,
                               'z_min': minpos,
                               'z_max': maxpos}

        self.spiketrain_params = {'n': self.num_synapses,
                                  'spTimesFun': LFPy.inputgenerators.stationary_poisson,
                                  'args': [1, 5, self.cell_params['tstartms'], self.cell_params['tstopms']]
                                  }

        self.synapse_params = {
            'e': 0.,                   # reversal potential
            'syntype': 'ExpSyn',       # synapse type
            'tau': 2.,                # syn. time constant
            'weight': self.weight,            # syn. weight
        }

    def make_all_input_trains(self):
        """ Makes all the input spike trains. Totally N / 0.01, since that is the
        maximum number of needed ones"""
        num_trains = int(self.num_synapses/0.01)
        all_spiketimes = {}
        for idx in xrange(num_trains):
            all_spiketimes[idx] = self.spiketrain_params['spTimesFun'](*self.spiketrain_params['args'])[0]
        np.save(join(self.data_folder, 'all_spike_trains.npy'), all_spiketimes)

    def return_cell(self, cell_idx):
        x, y, z, rotation = np.load(join(self.data_folder, 'x_y_z_rot_%d_%d.npy' %
                                    (self.num_cells, self.population_radius)))[:, cell_idx]
        rot_params = {'x': 0,
                      'y': 0,
                      'z': rotation
                      }
        pos_params = {'xpos': x,
                      'ypos': y,
                      'zpos': 0,
                      }
        neuron.h('forall delete_section()')
        cell = LFPy.Cell(**self.cell_params)
        cell.set_rotation(**rot_params)
        cell.set_pos(**pos_params)
        return cell

    def set_synaptic_input(self, cell):

        cell_input_idxs = cell.get_rand_idx_area_norm(**self.syn_pos_params)
        #self.synapse_pos_plot(cell, cell_input_idxs)

        if np.abs(self.correlation) > 1e-6:
            # The spiketrain indexes are drawn from a common pool without replacement.
            # The size of the common pool descides the average correlation
            all_spike_trains = np.load(join(self.data_folder, 'all_spike_trains.npy')).item()
            spike_train_idxs = np.array(random.sample(np.arange(int(self.spiketrain_params['n']/self.correlation)),
                                                      self.spiketrain_params['n']))
        else:
            # If the correlation is zero-like, we just make new spike trains
            all_spike_trains = {}
            for idx in xrange(self.spiketrain_params['n']):
                all_spike_trains[idx] = self.spiketrain_params['spTimesFun'](*self.spiketrain_params['args'])[0]
            spike_train_idxs = np.arange(self.spiketrain_params['n'])

        self.set_input_spiketrain(cell, all_spike_trains, cell_input_idxs, spike_train_idxs)
        return cell

    def run_single_cell_simulation(self, cell_idx):

        sim_name = '%s_cell_%d' % (self.stem, cell_idx)
        # if os.path.isfile(join(self.data_folder, 'lfp_%s.npy' % sim_name)):
        #     print "Skipping", sim_name
        #     return None

        plt.seed(cell_idx)
        cell = self.return_cell(cell_idx)
        cell = self.set_synaptic_input(cell)

        electrode = LFPy.RecExtElectrode(**self.electrode_params)
        cell.simulate(electrode=electrode, rec_imem=False, rec_vmem=False)

        if np.max(cell.somav) > -40:
            is_spiking = True
        else:
            is_spiking = False

        plt.close('all')
        plt.title('%s\nMEAN: %f mV, STD: %f mV' % (sim_name, np.mean(cell.somav), np.std(cell.somav)))
        plt.plot(cell.tvec, cell.somav)
        plt.savefig(join(self.fig_folder, '%s.png' % sim_name))

        #electrode.calc_lfp()
        np.save(join(self.data_folder, 'somav_%s.npy' % sim_name), cell.somav)
        np.save(join(self.data_folder, 'lfp_%s.npy' % sim_name), 1000*electrode.LFP)

    def sum_signals(self):
        total_signal = np.zeros((self.num_elecs, self.num_tsteps))
        session_name = join(self.data_folder, 'lfp_%s' % self.stem)
        for cell_idx in xrange(self.num_cells):
            total_signal += np.load('%s_cell_%d.npy' % (session_name, cell_idx))
        np.save('%s_total.npy' % session_name, total_signal)

    def sum_signals_by_population_size(self):
        x_y_z_rot = np.load(join(self.data_folder, 'x_y_z_rot_%d_%d.npy' % (self.num_cells, self.population_radius)))
        session_name = join(self.data_folder, 'lfp_%s' % self.stem)
        for size in self.population_sizes:
            total_signal = np.zeros((self.num_elecs, self.num_tsteps))
            for cell_idx in xrange(self.num_cells):
                if np.sqrt(np.sum(x_y_z_rot[:2, cell_idx]**2)) <= size:
                    #print 'x, y, size:', x_y_z_rot[:2, cell_idx], size
                    total_signal += np.load('%s_cell_%d.npy' % (session_name, cell_idx))
            np.save('%s_total_%d.npy' % (session_name, size), total_signal)


    def set_input_spiketrain(self, cell, all_spike_trains, cell_input_idxs, spike_train_idxs):
        """ Makes synapses and feeds them predetermined spiketimes """
        for number, comp_idx in enumerate(cell_input_idxs):
            self.synapse_params.update({'idx': int(comp_idx)})
            s = LFPy.Synapse(cell, **self.synapse_params)
            spiketrain_idx = spike_train_idxs[number]
            s.set_spike_times(all_spike_trains[spiketrain_idx])


    def plot_cell_population_to_ax(self, ax):
        x_y_z_rot = np.load(join(self.data_folder, 'x_y_z_rot_%d_%d.npy' %
                                (self.num_cells, self.population_radius)))

        ax.scatter(x_y_z_rot[0, :], x_y_z_rot[2, :], c='gray', edgecolor='none', s=10, zorder=1)

        plot_idx = np.argmin(np.abs(x_y_z_rot[0, :]))
        plt.seed(plot_idx)
        cell = self.return_cell(plot_idx)
        cell = self.set_synaptic_input(cell)
        [ax.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], color='k', zorder=0)
                        for idx in xrange(len(cell.xstart))]
        ax.plot(cell.xmid[cell.synidx], cell.zmid[cell.synidx], '.', c='r', ms=2)

    def plot_central_LFP(self, conductance_list):
        plt.close('all')
        fig = plt.figure(figsize=[8, 12])
        fig.subplots_adjust(left=0, right=0.95)
        elec_axs = {}
        elec_psd_axs = {}
        elec_ax_dict = {'xticks': []}
        elec_psd_ax_dict = {'ylim': [1e-7, 1e-0], 'xlim': [1e0, 1e3]}

        ax0 = fig.add_subplot(131, aspect=1, frameon=False, xticks=[], yticks=[])
        ax0.scatter(self.elec_x[self.center_idxs], self.elec_z[self.center_idxs], zorder=2, s=50)
        self.plot_cell_population_to_ax(ax0)

        for numb, elec_idx in enumerate(self.center_idxs):
            title = 'x: %d$\mu$m, z: %d$\mu$m' % (self.elec_x[elec_idx], self.elec_z[elec_idx])
            #elec_axs[elec_idx] = fig.add_subplot(len(self.center_idxs), 3, 3*(len(self.center_idxs) - numb) - 1,
            #                                     title=title, **elec_ax_dict)
            elec_psd_axs[elec_idx] = fig.add_subplot(len(self.center_idxs), 2, 2*(len(self.center_idxs) - numb),
                                                     title=title, **elec_psd_ax_dict)
            elec_psd_axs[elec_idx].grid(True)
            if self.elec_z[elec_idx] == np.min(self.elec_z):
                #elec_axs[elec_idx].set_xticks([0, self.end_t])
                #elec_axs[elec_idx].set_xlabel('Time [ms]')
                #elec_axs[elec_idx].set_ylabel('$\mu$V')
                elec_psd_axs[elec_idx].set_ylabel('$\mu$V$^2$/Hz')
                elec_psd_axs[elec_idx].set_xlabel('Frequency [Hz]')

        aLFP.simplify_axes(elec_psd_axs.values())
        #aLFP.simplify_axes(elec_axs.values())
        lfp_max = 0
        for conductance_type in conductance_list:
            local_stem = self.get_simulation_stem(self.model, self.input_region, conductance_type,
                                                  self.holding_potential, self.correlation, self.weight)
            session_name = join(self.data_folder, 'lfp_%s_total.npy' % local_stem)
            lfp = np.load(session_name)

            freq, psd = aLFP.return_freq_and_psd_welch(lfp, self.welch_dict)
            for elec_idx in self.center_idxs:
                lfp_max = np.max([lfp_max, np.max(np.abs(lfp[elec_idx]))])
                #elec_axs[elec_idx].plot(self.tvec, lfp[elec_idx], c=self.conductance_clr[conductance_type], lw=1)
                elec_psd_axs[elec_idx].loglog(freq, psd[elec_idx], c=self.conductance_clr[conductance_type], lw=1)

        #[ax.set_ylim([-lfp_max, lfp_max]) for ax in elec_axs.values()]
        #[ax.set_yticks([-1.5, 1.5]) for ax in elec_axs.values()]
        [ax.set_yticks(ax.get_yticks()[::2]) for ax in elec_psd_axs.values()]
        fig.savefig(join(self.fig_folder, 'center_LFP_%s.png' % self.stem))

    def plot_latteral_LFP(self, conductance_list):
        plt.close('all')
        fig = plt.figure(figsize=[12, 8])
        fig.subplots_adjust(left=0.08, right=0.99, top=1.0)
        elec_axs = {}
        elec_psd_axs = {}
        elec_ax_dict = {'xticks': []}
        elec_psd_ax_dict = {'ylim': [1e-7, 1e-0], 'xlim': [1e0, 1e3]}

        plot_idxs = self.lateral_idxs[:5]

        ax0 = fig.add_subplot(211, aspect=1, frameon=False, xticks=[], yticks=[])
        ax0.scatter(self.elec_x[plot_idxs], self.elec_z[plot_idxs], zorder=2, s=50)
        self.plot_cell_population_to_ax(ax0)

        for numb, elec_idx in enumerate(plot_idxs):
            title = 'x: %d$\mu$m, z: %d$\mu$m' % (self.elec_x[elec_idx], self.elec_z[elec_idx])
            #elec_axs[elec_idx] = fig.add_subplot(len(plot_idxs), 3, 3*(len(plot_idxs) - numb) - 1,
            #                                     title=title, **elec_ax_dict)
            elec_psd_axs[elec_idx] = fig.add_subplot(2, len(plot_idxs), len(plot_idxs) + numb + 1,
                                                     title=title, **elec_psd_ax_dict)
            elec_psd_axs[elec_idx].grid(True)
            if self.elec_x[elec_idx] == np.min(self.elec_x):
                #elec_axs[elec_idx].set_xticks([0, self.end_t])
                #elec_axs[elec_idx].set_xlabel('Time [ms]')
                #elec_axs[elec_idx].set_ylabel('$\mu$V')
                elec_psd_axs[elec_idx].set_ylabel('$\mu$V$^2$/Hz')
                elec_psd_axs[elec_idx].set_xlabel('Frequency [Hz]')

        aLFP.simplify_axes(elec_psd_axs.values())
        #aLFP.simplify_axes(elec_axs.values())
        lfp_max = 0
        for conductance_type in conductance_list:
            local_stem = self.get_simulation_stem(self.model, self.input_region, conductance_type, self.holding_potential,
                                                  self.correlation, self.weight)
            session_name = join(self.data_folder, 'lfp_%s_total.npy' % local_stem)
            lfp = np.load(session_name)

            freq, psd = aLFP.return_freq_and_psd_welch(lfp, self.welch_dict)
            for elec_idx in plot_idxs:
                lfp_max = np.max([lfp_max, np.max(np.abs(lfp[elec_idx]))])
                #elec_axs[elec_idx].plot(self.tvec, lfp[elec_idx], c=self.conductance_clr[conductance_type], lw=1)
                elec_psd_axs[elec_idx].loglog(freq, psd[elec_idx], c=self.conductance_clr[conductance_type], lw=1)

        #[ax.set_ylim([-lfp_max, lfp_max]) for ax in elec_axs.values()]
        #[ax.set_yticks([-1.5, 1.5]) for ax in elec_axs.values()]
        [ax.set_yticks(ax.get_yticks()[::2]) for ax in elec_psd_axs.values()]
        fig.savefig(join(self.fig_folder, 'latteral_LFP_%s.png' % self.stem))

def alternative_MPI_dist():

    from mpi4py import MPI

    class Tags():
        def __init__(self):
            self.READY = 0
            self.DONE = 1
            self.EXIT = 2
            self.START = 3
    tags = Tags()
    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    num_workers = size - 1

    if rank == 0:
        # Master process executes code below
        print "NOT NOT NOT Initializing population"
        # pop = Population(initialize=True)

        correlations = [0.0, 0.1]
        conductance_types = ['active']#, 'passive']
        input_regions = ['homogeneous']#, 'tuft']

        num_cells = 5 #pop.num_cells

        print("Master starting with %d workers" % num_workers)
        for correlation in correlations:
            for input_region in input_regions:
                for conductance in conductance_types:
                    pop = Population(conductance_type=conductance,
                                     correlation=correlation,
                                     input_region=input_region)
                    for cell_idx in xrange(num_cells):
                        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                        source = status.Get_source()
                        tag = status.Get_tag()
                        if tag == tags.READY and source % 2:
                            comm.send([pop, cell_idx], dest=source, tag=tags.START)
                            print("Sending task to worker %d" % source)
                            #task_index += 1
                        elif tag == tags.DONE:
                            print("Worker %d completed task" % source)
        for worker in range(1, num_workers + 1):
            comm.send([None, None], dest=worker, tag=tags.EXIT)
        print("Master finishing")
    else:
        # Worker processes execute code below
        #print("I am a worker with rank %d." % rank)
        while True:
            if not rank % 2:
                break
            comm.send(None, dest=0, tag=tags.READY)
            [pop, cell_idx] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                print rank, "put to work"
                plt.pause(1 + np.random.random())
                comm.send(None, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                print rank, "exiting"
                break
        comm.send(None, dest=0, tag=tags.EXIT)

def MPI_population_size_sum():
    """ Run with
        openmpirun -np 4 python example_mpi.py
    """
    from mpi4py import MPI

    class Tags():
        def __init__(self):
            self.READY = 0
            self.DONE = 1
            self.EXIT = 2
            self.START = 3
            self.ERROR = 4
    tags = Tags()
    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    num_workers = size - 1

    if rank == 0:
        # Master process executes code below
        correlations = [0.0, 0.1, 1.0]
        conductance_types = ['active', 'passive']
        input_regions = ['homogeneous', 'distal_tuft', 'basal']

        print("\033[95m Master starting with %d workers" % num_workers)
        task = 0
        num_tasks = len(correlations) * len(conductance_types) * len(input_regions)
        for correlation in correlations:
            for input_region in input_regions:
                for conductance in conductance_types:
                    pop = Population(conductance_type=conductance,
                                     correlation=correlation,
                                     input_region=input_region)
                    task += 1
                    sent = False
                    while not sent:
                        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                        source = status.Get_source()
                        tag = status.Get_tag()
                        if tag == tags.READY:
                            comm.send(pop, dest=source, tag=tags.START)
                            print "\033[95m Sending task %d/%d to worker %d\033[0m" % (task, num_tasks, source)
                            sent = True
                        elif tag == tags.DONE:
                            print "\033[95m Worker %d completed task %d/%d\033[0m" % (source, task, num_tasks)
                        elif tag == tags.ERROR:
                            print "\033[91mMaster detected ERROR at node %d. Aborting...\033[0m" % source
                            for worker in range(1, num_workers + 1):
                                print "sending"
                                comm.send(None, dest=worker, tag=tags.EXIT)
                            sys.exit()

        for worker in range(1, num_workers + 1):
            comm.send(None, dest=worker, tag=tags.EXIT)
        print("\033[95m Master finishing\033[0m")
    else:
        #print("I am a worker with rank %d." % rank)
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            pop = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                print "\033[93m%d put to work\033[0m" % rank
                # plt.pause(1 + np.random.random())
                try:
                    pop.sum_signals_by_population_size()
                except:
                    print "\033[91mNode %d exiting with ERROR\033[0m" % rank
                    comm.send(None, dest=0, tag=tags.ERROR)
                    sys.exit()

                comm.send(None, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                print "\033[93m%d exiting\033[0m" % rank
                break
        comm.send(None, dest=0, tag=tags.EXIT)


def MPI_population_simulation():
    """ Run with
        openmpirun -np 4 python example_mpi.py
    """
    from mpi4py import MPI

    class Tags():
        def __init__(self):
            self.READY = 0
            self.DONE = 1
            self.EXIT = 2
            self.START = 3
            self.ERROR = 4
    tags = Tags()
    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    num_workers = size - 1

    if size == 1:
        sys.exit()

    if rank == 0:
        pop = Population(initialize=True)
        # Master process executes code below
        correlations = [0.0, 1.0]
        conductance_types = [-0.5, 0.0, 2.0]
        distributions = ['linear_increase']
        input_regions = ['homogeneous', 'distal_tuft', 'basal']

        print("\033[95m Master starting with %d workers\033[0m" % num_workers)
        task = 0
        num_cells = Population().num_cells
        num_tasks = len(correlations) * len(conductance_types) * len(input_regions) * len(distributions) * num_cells
        for correlation in correlations:
            for input_region in input_regions:
                for distribution in distributions:
                    for conductance in conductance_types:

                        pop_dict = {'conductance_type': conductance,
                                    'correlation': correlation,
                                    'input_region': input_region,
                                    'distribution': distribution
                                    }

                        for cell_idx in xrange(num_cells):
                            task += 1
                            sent = False
                            while not sent:
                                data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                                source = status.Get_source()
                                tag = status.Get_tag()
                                if tag == tags.READY:
                                    comm.send([pop_dict, cell_idx], dest=source, tag=tags.START)
                                    print "\033[95m Sending task %d/%d to worker %d\033[0m" % (task, num_tasks, source)
                                    sent = True
                                elif tag == tags.DONE:
                                    print "\033[95m Worker %d completed task %d/%d\033[0m" % (source, task, num_tasks)
                                elif tag == tags.ERROR:
                                    print "\033[91mMaster detected ERROR at node %d. Aborting...\033[0m" % source
                                    for worker in range(1, num_workers + 1):
                                        #print "sending"
                                        comm.send([{}, None], dest=worker, tag=tags.EXIT)
                                    sys.exit()

        for worker in range(1, num_workers + 1):
            comm.send([{}, None], dest=worker, tag=tags.EXIT)
        print("\033[95m Master finishing\033[0m")
    else:
        #print("I am a worker with rank %d." % rank)
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            [pop_dict, cell_idx] = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                print "\033[93m%d put to work on %s cell %d\033[0m" % (rank, str(pop_dict.values()), cell_idx)
                # try:
                pop = Population(**pop_dict)
                pop.run_single_cell_simulation(cell_idx)
                # except:
                #     print "\033[91mNode %d exiting with ERROR\033[0m" % rank
                #     comm.send(None, dest=0, tag=tags.ERROR)
                #     sys.exit()
                comm.send(None, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                print "\033[93m%d exiting\033[0m" % rank
                break
        comm.send(None, dest=0, tag=tags.EXIT)





def distribute_cellsims_MPI():
    """ Run with
        openmpirun -np 4 python example_mpi.py
    """
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    #if RANK == 0:
    #    print "Initializing population"
    #    pop = Population(initialize=True)
    #COMM.Barrier()

    correlations = [0.0, 0.1, 1.0]
    conductance_types = ['active', 'passive']
    input_regions = ['basal', 'distal_tuft', 'homogeneous']
    sim_num = 0
    #if RANK % 2:
    #    return
    #for correlation in correlations:
    #    for input_region in input_regions:
    #        for conductance in conductance_types:
    #            pop = Population(conductance_type=conductance, correlation=correlation, input_region=input_region)
    #            for cell_idx in xrange(pop.num_cells):
    #                if divmod(sim_num, SIZE / 2)[1] == RANK / 2:
    #                    print "Rank %d simulating %s_cell_%d" % (RANK, pop.stem, cell_idx)
    #                    pop.run_single_cell_simulation(cell_idx)
    #                sim_num += 1
    #COMM.Barrier()
    #if RANK == 0:
    #    print "Summing signals."
    #sim_num = 0
    #for correlation in correlations:
    #    for input_region in input_regions:
    #        for conductance in conductance_types:
    #
    #            pop = Population(conductance_type=conductance, correlation=correlation, input_region=input_region)
    #            if divmod(sim_num, SIZE / 2)[1] == RANK / 2:
    #                print correlation, input_region, conductance
    #                print RANK, "summing", pop.stem
    #                #pop.sum_signals()
    #            sim_num += 1
    #COMM.Barrier()
    if RANK == 0:
        print "Plotting LFPs"
    sim_num = 0
    for correlation in correlations:
        for input_region in input_regions:
            pop = Population(correlation=correlation, input_region=input_region)
            if divmod(sim_num, SIZE / 2)[1] == RANK / 2:
                print RANK, "plotting", pop.stem
                pop.plot_central_LFP(conductance_types)
            sim_num += 1


def plot_all_center_LFPs():

    correlations = [0.0, 0.1, 1.0]
    conductance_types = ['active', 'passive']
    input_regions = ['basal', 'homogeneous', 'distal_tuft']
    for correlation in correlations:
        for input_region in input_regions:
            pop = Population(correlation=correlation, input_region=input_region)
            pop.plot_central_LFP(conductance_types)

def plot_all_latteral_LFPs():

    correlations = [0.0, 0.1, 1.0]
    conductance_types = ['active', 'passive']
    input_regions = ['basal', 'homogeneous', 'distal_tuft']
    for correlation in correlations:
        for input_region in input_regions:
            pop = Population(correlation=correlation, input_region=input_region)
            pop.plot_latteral_LFP(conductance_types)


def test_sim():

    # TODO: FIND WAY TO COMPARE CONDUCTANCE TYPES
    pop = None
    for correlation in [0.0]:
        for conductance in ['passive']:
            for cell_idx in xrange(6, 36):
                pop = Population(conductance_type=conductance, correlation=correlation)
                pop.run_single_cell_simulation(cell_idx)

if __name__ == '__main__':
    #test_sim()
    #distribute_cellsims_MPI()
    # alternative_MPI_dist()
    #pop = Population(correlation=0.0, input_region='basal')
    #pop.plot_LFP(['active', 'passive'])
    #plot_all_LFPs()
    #MPI_population_size_sum()
    #plot_all_latteral_LFPs()
    MPI_population_simulation()