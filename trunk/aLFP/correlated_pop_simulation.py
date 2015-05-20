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

model = 'hay'
if at_stallo:
    neuron_model = join('/home', 'torbness', 'work', 'aLFP', 'neuron_models', model)
else:
    neuron_model = join('/home', 'torbjone', 'work', 'aLFP', 'neuron_models', model)
neuron.load_mechanisms(join(neuron_model, 'mod'))
neuron.load_mechanisms(join(neuron_model, '..'))
from hay_active_declarations import active_declarations as hay_active


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
            self.cut_off = 100

        if not os.path.isdir(self.data_folder):
            os.mkdir(self.data_folder)

        self.timeres = 2**-4
        self.num_cells = 5
        self.population_radius = 10.
        self.num_synapses = 1000
        self.correlation = correlation
        self.holding_potential = holding_potential
        self.conductance_type = conductance_type
        self.input_region = input_region
        self.weight = weight
        self.end_t = 100
        self.num_tsteps = round((self.end_t - 0) / self.timeres + 1)
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

        # if self.conductance_type == 'active':
        #     print '%s is spiking: %s' % (sim_name, is_spiking)

        #print sim_name, np.std(cell.vmem), np.mean(cell.vmem)

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

def distribute_cellsims_MPI():
    """ Run with
        openmpirun -np 4 python example_mpi.py
    """
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    if RANK == 0:
        print RANK, " initializing population"
        pop = Population(initialize=True)
    COMM.Barrier()
    sim_num = 0
    for correlation in [0.0]:
        for conductance in ['active', 'passive']:
            pop = Population(conductance_type=conductance, correlation=correlation)
            for cell_idx in xrange(pop.num_cells):
                if divmod(sim_num, SIZE)[1] == RANK:
                    print RANK, "simulating ", cell_idx
                    pop.run_single_cell_simulation(cell_idx)
                sim_num += 1
    COMM.Barrier()
    print RANK, "reached this point"
    sim_num = 0
    for correlation in [0.0, 1.0]:
        for conductance in ['active', 'passive']:
            pop = Population(conductance_type=conductance, correlation=correlation)
            if divmod(sim_num, SIZE)[1] == RANK:
                print RANK, "summing ", pop.stem
                pop.sum_signals()
                sim_num += 1


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
    distribute_cellsims_MPI()
    # pop = Population()