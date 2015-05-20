#!/usr/bin/env python
import os
import sys
from os.path import join
import random
import numpy as np
import pylab as plt
import neuron
import LFPy

if not 'DISPLAY' in os.environ:
    import matplotlib
    matplotlib.use('Agg')
    at_stallo = True
else:
    at_stallo = False
import aLFP
import scipy.fftpack as ff
import scipy.signal


class Population():

    def __init__(self, holding_potential=-70, conductance_type='active', correlation=0.0,
                 weight=0.0001, input_region='homogeneous', initialize=False):

        self.model = 'hay'
        self.sim_name = 'hay_smaller_pop'

        if at_stallo:
            self.neuron_model = join('/home', 'torbness', 'work', 'aLFP', 'neuron_models', self.model)
            self.fig_folder = join('/home', 'torbness', 'work', 'aLFP', 'correlated_population')
            self.data_folder = join('/home', 'torbness', 'work', 'aLFP', 'correlated_population', self.sim_name)
            self.cut_off = 500
        else:
            self.fig_folder = join('/home', 'torbjone', 'work', 'aLFP', 'correlated_population')
            self.data_folder = join('/home', 'torbjone', 'work', 'aLFP', 'correlated_population', self.sim_name)
            self.neuron_model = join('/home', 'torbjone', 'work', 'aLFP', 'neuron_models', self.model)
            self.cut_off = 500

        if not os.path.isdir(self.data_folder):
            os.mkdir(self.data_folder)

        self.timeres = 2**-4
        self.num_cells = 100
        self.population_radius = 100.
        self.num_synapses = 1000
        self.correlation = correlation
        self.holding_potential = holding_potential
        self.conductance_type = conductance_type
        self.input_region = input_region
        self.weight = weight
        self.end_t = 10000
        self.num_tsteps = round((self.end_t - 0) / self.timeres)
        self.tvec = np.linspace(0, self.end_t, self.num_tsteps)
        self.stem = '%s_%s_%1.2f_%1.5f' % (self.sim_name, self.conductance_type, self.correlation, self.weight)

        self.set_cell_params()
        self.set_input_params()
        self.initialize_electrodes()
        if initialize:
            self.distribute_cells()
            self.make_all_input_trains()
            self.plot_set_up()

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

        n_elecs_lateral = 41
        elec_x_lateral = np.linspace(0, 10000, n_elecs_lateral)
        elec_y_lateral = np.zeros(n_elecs_lateral)
        elec_z_lateral = np.zeros(n_elecs_lateral)

        self.elec_x = np.r_[elec_x_center, elec_x_lateral]
        self.elec_y = np.r_[elec_y_center, elec_y_lateral]
        self.elec_z = np.r_[elec_z_center, elec_z_lateral]

        self.center_idxs = np.arange(n_elecs_center)
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
        neuron.load_mechanisms(join(self.neuron_model, 'mod'))
        neuron.load_mechanisms(join(self.neuron_model, '..'))
        from hay_active_declarations import active_declarations as hay_active
        
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
        if self.input_region is 'distal_tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 900
        elif self.input_region is 'tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 600
        elif self.input_region is 'homogeneous':
            input_pos = ['apic', 'dend']
            maxpos = 10000
            minpos = -10000
        elif self.input_region is 'basal':
            input_pos = ['dend']
            maxpos = 10000
            minpos = -10000
        else:
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
        num_trains = int(self.num_cells/0.01)
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
        plt.seed(cell_idx)
        cell = self.return_cell(cell_idx)
        cell = self.set_synaptic_input(cell)

        sim_name = '%s_cell_%d' % (self.stem, cell_idx)

        cell.simulate(rec_imem=True, rec_vmem=True)

        if np.max(cell.somav) > -40:
            is_spiking = True
        else:
            is_spiking = False

        plt.close('all')
        plt.plot(cell.tvec, cell.somav)
        plt.savefig(join(self.fig_folder, '%s.png' % sim_name))

        if self.conductance_type == 'active':
            print '%s is spiking: %s' % (sim_name, is_spiking)

        print sim_name, np.std(cell.vmem), np.mean(cell.vmem)

        electrode = LFPy.RecExtElectrode(cell, **self.electrode_params)

        # np.save(join(self.folder, 'xstart.npy'), cell.xstart)
        # np.save(join(self.folder, 'ystart.npy'), cell.ystart)
        # np.save(join(self.folder, 'zstart.npy'), cell.zstart)
        # np.save(join(self.folder, 'xend.npy'), cell.xend)
        # np.save(join(self.folder, 'yend.npy'), cell.yend)
        # np.save(join(self.folder, 'zend.npy'), cell.zend)
        # np.save(join(self.folder, 'xmid.npy'), cell.xmid)
        # np.save(join(self.folder, 'ymid.npy'), cell.ymid)
        # np.save(join(self.folder, 'zmid.npy'), cell.zmid)
        # np.save(join(self.folder, 'diam.npy'), cell.diam)

        electrode.calc_lfp()
        # if not at_stallo:
        #     np.save(join(self.data_folder, 'imem_%s.npy' % sim_name), cell.imem)
        np.save(join(self.data_folder, 'somav_%s.npy' % sim_name), cell.somav)
        #     np.save(join(self.data_folder, 'vmem_%s.npy' % sim_name), cell.vmem)
        #     if self.conductance_type == 'active':
        #         plt.close('all')
        #         plt.subplot(121, aspect='equal', xlim=[-1000, 1000], ylim=[-1000, 1000])
        #         plt.scatter(cell.xmid, cell.ymid, edgecolor='none', s=2, c='r')
        #         plt.scatter(self.elec_x, self.elec_y, s=4, c='b')
        #
        #         plt.subplot(122, aspect='equal', xlim=[-1000, 1000], ylim=[-400, 1400])
        #         plt.scatter(cell.xmid, cell.zmid, edgecolor='none', s=2, c='r')
        #         plt.scatter(self.elec_x, self.elec_z, s=4, c='b')
        #         plt.savefig('cell_%05d.png' % cell_idx)
        #
        # elif cell_idx % 500 == 0:
        #     np.save(join(self.data_folder, 'imem_%s.npy' % sim_name), cell.imem)
        #     np.save(join(self.data_folder, 'somav_%s.npy' % sim_name), cell.somav)
        #     np.save(join(self.data_folder, 'vmem_%s.npy' % sim_name), cell.vmem)
        #
        #     if self.conductance_type == 'active':
        #         plt.close('all')
        #         plt.subplot(121, aspect='equal', xlim=[-1000, 1000], ylim=[-1000, 1000])
        #         plt.scatter(cell.xmid, cell.ymid, edgecolor='none', s=2, c='r')
        #         plt.scatter(self.elec_x, self.elec_y, s=4, c='b')
        #         plt.subplot(122, aspect='equal', xlim=[-1000, 1000], ylim=[-400, 1400])
        #         plt.scatter(cell.xmid, cell.zmid, edgecolor='none', s=2, c='r')
        #         plt.scatter(self.elec_x, self.elec_z, s=4, c='b')
        #         plt.savefig('cell_%05d.png' % cell_idx)

        np.save(join(self.data_folder, 'lfp_%s.npy' % sim_name), 1000*electrode.LFP)

    def sum_signals(self):
        total_signal = np.zeros((self.num_elecs, self.num_tsteps))
        session_name = join(self.data_folder, 'lfp_%s' % self.stem)
        for cell_idx in xrange(self.num_cells):
            total_signal += np.load('%s_cell_%d.npy' % (session_name, cell_idx))
        np.save('%s_total.npy' % session_name, total_signal)

    def set_input_spiketrain(self, cell, all_spike_trains, cell_input_idxs, spike_train_idxs):
        """ Makes synapses and feeds them predetermined spiketimes """
        for number, comp_idx in enumerate(cell_input_idxs):
            self.synapse_params.update({'idx': int(comp_idx)})
            s = LFPy.Synapse(cell, **self.synapse_params)
            spiketrain_idx = spike_train_idxs[number]
            s.set_spike_times(all_spike_trains[spiketrain_idx])

    def plot_LFP(self, conductance_list):

        fig = plt.figure()
        elec_axs = {}
        elec_ax_dict = {}
        for numb, elec_idx in enumerate(self.center_idxs):
            print self.elec_x[elec_idx],  self.elec_y[elec_idx],  self.elec_z[elec_idx]
            elec_axs[elec_idx] = fig.add_subplot(len(self.center_idxs), 1, numb + 1, **elec_ax_dict)

        for conductance_type in conductance_list:
            stem = 'lfp_%s_%s_%1.2f_%1.5f_total' % (self.sim_name, conductance_type, self.correlation, self.weight)
            session_name = join(self.data_folder, stem)
            lfp = np.load(session_name)
            for elec_idx in self.center_idxs:
                elec_axs[elec_idx].plot(self.tvec, lfp[elec_idx])
        fig.savefig(join(self.fig_folder, 'center_LFP_%d_%d.png' % (self.num_cells, self.population_radius)))


    def plot_compare_single_input_state(self):

        conductance_list = ['active', 'passive']

        simulation_idx = 0
        syn_strength = .1
        input_pos = 'homogeneous'
        correlation = 1.0

        xmid = np.load(join(folder, 'xmid.npy'))
        ymid = np.load(join(folder, 'ymid.npy'))
        zmid = np.load(join(folder, 'zmid.npy'))
        xstart = np.load(join(folder, 'xstart.npy'))
        ystart = np.load(join(folder, 'ystart.npy'))
        zstart = np.load(join(folder, 'zstart.npy'))
        xend = np.load(join(folder, 'xend.npy'))
        yend = np.load(join(folder, 'yend.npy'))
        zend = np.load(join(folder, 'zend.npy'))

        ncols = 6
        nrows = 3
        apic_idx = 475

        electrodes = np.array([1, 3, 5])
        elec_clr = lambda idx: plt.cm.rainbow(int(256. * idx/(len(electrodes) - 1.)))

        clr = lambda idx: plt.cm.jet(int(256. * idx/(len(conductance_list) - 1.)))

        v_min = -90
        v_max = 0

        for state in states:

            if state == '':
                state_name = '0'
            else:
                state_name = state[1:]
            plt.close('all')
            fig = plt.figure(figsize=[12,8])
            fig.suptitle('Synaptic input: %s, synaptic strength: %1.2f, Leak reversal shifted: %s mV'
                         % (input_pos, syn_strength, state_name))
            fig.subplots_adjust(wspace=0.5, hspace=0.5)

            ax_morph = fig.add_axes([0.02, 0.1, 0.15, 0.8], frameon=False, xticks=[], yticks=[])
            plot_cell_to_ax(ax_morph, xstart, xmid, xend, ystart, ymid, yend, elec_x,
                            elec_z, electrodes, apic_idx, elec_clr)

            # Somatic Vm
            ax_s1 = fig.add_axes([0.57, 0.1, 0.1, 0.2], ylim=[v_min, v_max], xlabel='ms', ylabel='Somatic Vm [mV]')
            ax_s2 = fig.add_axes([0.72, 0.1, 0.1, 0.2], xlabel='ms', ylabel='Shifted somatic Vm [mV]')
            ax_s3 = fig.add_axes([0.87, 0.1, 0.1, 0.2], xlim=[1, 500], xlabel='Hz',
                                 ylabel='Somatic Vm PSD [$mV^2$]')

            # Apic Vm
            ax_a1 = fig.add_axes([0.57, 0.4, 0.1, 0.2], ncols, 10, ylim=[v_min, v_max],
                                 xlabel='ms', ylabel='Apical Vm [mV]')
            ax_a2 = fig.add_axes([0.72, 0.4, 0.1, 0.2], 11, xlabel='ms', ylabel='Shifted apical Vm [mV]')
            ax_a3 = fig.add_axes([0.87, 0.4, 0.1, 0.2], xlim=[1, 500], xlabel='Hz', ylabel='Apical Vm PSD [$mV^2$]')

            ax_e1 = fig.add_axes([0.2, 0.1, 0.1, 0.2], xlabel='ms', ylabel='$\mu V$')
            ax_e2 = fig.add_axes([0.2, 0.4, 0.1, 0.2], xlabel='ms', ylabel='$\mu V$')
            ax_e3 = fig.add_axes([0.2, 0.7, 0.1, 0.2], xlabel='ms', ylabel='$\mu V$', title='Extracellular')

            ax_e1_psd = fig.add_axes([0.35, 0.1, 0.1, 0.2], xlim=[1, 500], xlabel='Hz', ylabel='$\mu V^2$')
            ax_e2_psd = fig.add_axes([0.35, 0.4, 0.1, 0.2], xlim=[1, 500], xlabel='Hz', ylabel='$\mu V^2$')
            ax_e3_psd = fig.add_axes([0.35, 0.7, 0.1, 0.2], xlim=[1, 500], xlabel='Hz', ylabel='$\mu V^2$',
                                     title='Extracellular PSD')

            e_ax = [ax_e1, ax_e2, ax_e3]
            e_ax_psd = [ax_e1_psd, ax_e2_psd, ax_e3_psd]

            axes = [ax_s1, ax_s2, ax_s3, ax_a1, ax_a2, ax_a3, ax_e1,
                    ax_e2, ax_e3, ax_e1_psd, ax_e2_psd, ax_e3_psd]
            for ax in axes:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

            clr_ax = [ax_e1, ax_e2, ax_e3]
            clr_ax_psd = [ax_e1_psd, ax_e2_psd, ax_e3_psd]
            for axnumb in xrange(len(clr_ax)):
                for spine in clr_ax[axnumb].spines.values():
                    spine.set_edgecolor(elec_clr(axnumb))
                for spine in clr_ax_psd[axnumb].spines.values():
                    spine.set_edgecolor(elec_clr(axnumb))

            ax_s3.grid(True)
            ax_a3.grid(True)

            ax_e1_psd.grid(True)
            ax_e2_psd.grid(True)
            ax_e3_psd.grid(True)

            #ax_s4.grid(True)
            lines = []
            line_names = []
            for idx, conductance_type in enumerate(conductance_list):

                identifier = '%s_%s_%1.2f_%1.3f%s_sim_%d.npy' %(conductance_type, input_pos, correlation,
                                                                syn_strength, state, simulation_idx)
                #somav = np.load(join(folder, 'somav_%s' %identifier))
                vmem = np.load(join(folder, 'vmem_%s' %identifier))
                somav = vmem[0,:]
                apicv = vmem[apic_idx,:]
                sig = np.load(join(folder, 'signal_%s' %(identifier)))[:,:]

                v_range = np.max(np.average(vmem, axis=1)) - np.min(np.average(vmem, axis=1))

                ax0 = fig.add_subplot(nrows, ncols + 3, idx + 5, frameon=False, xticks=[], yticks=[],
                                      title=conductance_type, aspect='equal')
                img = ax0.scatter(xmid, ymid, c=np.average(vmem, axis=1), edgecolor='none', s=1)
                cbar = plt.colorbar(img, ticks=[np.min(np.average(vmem, axis=1)), np.max(np.average(vmem, axis=1))], format='%1.1f')
                #cbar.set_label('Average Vm')
                somav_psd, freqs = aLFP.find_LFP_PSD(np.array([somav]), (1.0)/1000.)
                somav_psd_welch, freqs_welch = mlab.psd(somav, Fs=1000.,
                                                            NFFT=int(len(somav)/divide_into_welch),
                                                            noverlap=int(len(somav)/divide_into_welch/2),
                                                            window=plt.window_hanning, detrend=plt.detrend_mean)

                apicv_psd, freqs = aLFP.find_LFP_PSD(np.array([apicv]), (1.0)/1000.)
                apicv_psd_welch, freqs_welch = mlab.psd(apicv, Fs=1000.,
                                                            NFFT=int(len(somav)/divide_into_welch),
                                                            noverlap=int(len(somav)/divide_into_welch/2),
                                                            window=plt.window_hanning, detrend=plt.detrend_mean)

                for eidx, elec in enumerate(electrodes):
                    sig_psd_welch, freqs_welch = mlab.psd(sig[elec], Fs=1000.,
                                                            NFFT=int(len(somav)/divide_into_welch),
                                                            noverlap=int(len(somav)/divide_into_welch/2),
                                                            window=plt.window_hanning, detrend=plt.detrend_mean)
                    e_ax[eidx].plot(sig[elec] - sig[elec,0], color=clr(idx))
                    e_ax_psd[eidx].loglog(freqs_welch, sig_psd_welch, color=clr(idx))

                ax_s1.plot(somav, color=clr(idx))
                ax_s2.plot(somav - somav[0], color=clr(idx))
                ax_s3.loglog(freqs_welch, somav_psd_welch, color=clr(idx))
                ax_a1.plot(apicv, color=clr(idx))
                ax_a2.plot(apicv - apicv[0], color=clr(idx))
                l, = ax_a3.loglog(freqs_welch, apicv_psd_welch, color=clr(idx))

                line_names.append(conductance_type)
                lines.append(l)

            ax_s1.plot(ax_s1.get_xlim()[0], ax_s1.get_ylim()[1], 'yD', clip_on=False)
            ax_s2.plot(ax_s2.get_xlim()[0], ax_s2.get_ylim()[1], 'yD', clip_on=False)
            ax_s3.plot(ax_s3.get_xlim()[0], ax_s3.get_ylim()[1], 'yD', clip_on=False)
            ax_a1.plot(ax_a1.get_xlim()[0], ax_a1.get_ylim()[1], 'g*', clip_on=False, ms=10)
            ax_a2.plot(ax_a2.get_xlim()[0], ax_a2.get_ylim()[1], 'g*', clip_on=False, ms=10)
            ax_a3.plot(ax_a3.get_xlim()[0], ax_a3.get_ylim()[1], 'g*', clip_on=False, ms=10)

            ax_sparse = [ax_a1, ax_a2, ax_s1, ax_s2, ax_e1, ax_e2, ax_e3]
            for ax in ax_sparse:
                ax.set_xticks(ax.get_xticks()[::2])

            fig.legend(lines, line_names, frameon=False)

            fig.savefig(join('single_state_comparison_%s_%d_%1.2f%+s.png' %
                        (input_pos, simulation_idx, syn_strength, state)), dpi=150)
            plt.show()


def test_sim():

    # TODO: FIND WAY TO COMPARE CONDUCTANCE TYPES
    pop = Population(initialize=True)
    for correlation in [0.0]:
        for conductance in ['active', 'passive']:
            for cell_idx in xrange(pop.num_cells):
                pop = Population(conductance_type=conductance, correlation=correlation)
                pop.run_single_cell_simulation(cell_idx)

if __name__ == '__main__':
    test_sim()
    # pop = Population()