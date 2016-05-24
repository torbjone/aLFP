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
import neuron
import LFPy
#sys.path.append(join('..', '..', 'neuron_models', 'hay'))
from plotting_convention import *
nrn = neuron.h

class GenericStudy:

    def __init__(self, cell_name, input_type, conductance='generic', extended_electrode=True):

        self.mu_name_dict = {-0.5: 'Regenerative ($\mu^* =\ -0.5$)',
                             0: 'Passive ($\mu^* =\ 0$)',
                             2: 'Restorative ($\mu^* =\ 2$)'}

        self.cell_name = cell_name
        self.conductance = conductance
        self.input_type = input_type
        self.username = os.getenv('USER')
        self.root_folder = join('/home', self.username, 'work', 'aLFP')
        if at_stallo:
            self.figure_folder = join('/global', 'work', self.username, 'aLFP', 'generic_study')
            self.sim_folder = join('/global', 'work', self.username, 'aLFP', 'generic_study', cell_name)
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
        self.soma_idx = 0
        self.apic_idx = 605
        self.sec_clr_dict = {'soma': '0.3', 'dend': '0.5', 'apic': '0.7', 'axon': '0.1'}
        self._set_electrode_specific_properties(extended_electrode)
        self._set_input_specific_properties()
        self.num_tsteps = round(self.end_t/self.timeres_python + 1)
        self.divide_into_welch = 8.
        self.welch_dict = {'Fs': 1000 / self.timeres_python,
                           'NFFT': int(self.num_tsteps/self.divide_into_welch),
                           'noverlap': int(self.num_tsteps/self.divide_into_welch/2.),
                           'window': plt.window_hanning,
                           'detrend': plt.detrend_mean,
                           'scale_by_freq': True,
                           }

    def _set_input_specific_properties(self):
        if self.input_type == 'white_noise':
            print "white noise input"
            self.plot_psd = True
            self.single_neural_sim_function = self._run_single_wn_simulation
            self.timeres_NEURON = 2**-4
            self.timeres_python = 2**-4
            self.cut_off = 0
            self.repeats = 2
            self.end_t = 1000 * self.repeats
            self.max_freq = 500
            self.short_list_elecs = [1, 1 + 6, 1 + 6 * 2]
        elif self.input_type == 'distributed_synaptic':
            print "Distributed synaptic input"
            self.plot_psd = True
            self.single_neural_sim_function = self._run_distributed_synaptic_simulation
            self.timeres_NEURON = 2**-4
            self.timeres_python = 2**-4
            self.cut_off = 1000
            self.end_t = 20000
            self.repeats = None
            self.max_freq = 500
            self.short_list_elecs = [1, 1 + 6, 1 + 6 * 2]
        elif self.input_type == 'distributed_synaptic_cs':
            print "Distributed synaptic input with current synapses"
            self.plot_psd = True
            self.single_neural_sim_function = self._run_distributed_synaptic_simulation
            self.timeres_NEURON = 2**-4
            self.timeres_python = 2**-4
            self.cut_off = 100
            self.end_t = 20000
            self.repeats = None
            self.max_freq = 500
            self.short_list_elecs = [1, 1 + 6, 1 + 6 * 2]
        elif self.input_type == 'real_wn':
            print "REAL white noise input"
            self.plot_psd = True
            self.timeres_NEURON = 2**-4
            self.timeres_python = 2**-4
            self.single_neural_sim_function = self._run_single_wn_simulation
            self.cut_off = 0
            self.end_t = 5000
            self.max_freq = 500
            self.short_list_elecs = [1, 1 + 6, 1 + 6 * 2]

        else:
            raise RuntimeError("Unrecognized input type.")

    def _set_electrode_specific_properties(self, extended_electrode):
        if self.cell_name in ['hay', 'zuchkova']:
            self.zmax = 1100
            self.cell_plot_idxs = [self.apic_idx, self.soma_idx] # 455,

        self.distances = np.array([50, 100, 200, 400, 1600, 6400])

        elec_x, elec_z = np.meshgrid(self.distances, np.array([self.zmax, 500, 0]))
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
        neuron_models = join(self.root_folder, 'neuron_models')
        if not hasattr(neuron.h, "setdata_QA"):
            neuron.load_mechanisms(join(neuron_models))

        if self.cell_name == 'hay':

            if hasattr(self, 'w_bar_scaling_factor'):
                print "Scaling qausi-active conductance with %1.1f" % self.w_bar_scaling_factor
                fact = self.w_bar_scaling_factor
            else:
                fact = 1.

            sys.path.append(join(self.root_folder, 'neuron_models', 'hay'))
            from hay_active_declarations import active_declarations
            # neuron.load_mechanisms(join(neuron_models, 'hay', 'mod'))
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
                'custom_fun': [active_declarations],  # will execute this function
                'custom_fun_args': [{'conductance_type': conductance_type,
                                     'mu_factor': mu,
                                     'g_pas': 0.00005,#0.0002, # / 5,
                                     'distribution': distribution,
                                     'tau_w': tau_w,
                                     #'total_w_conductance': 6.23843378791,# / 5,
                                     'avrg_w_bar': 0.00005 * 2 * fact,
                                     'hold_potential': holding_potential}]
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
             'g_w_bar_QA': np.zeros(cell.totnsegs),
             }

        if not self.repeats is None:
            cut_off_idx = (len(cell.tvec) - 1) / self.repeats
            cell.tvec = cell.tvec[-cut_off_idx:] - cell.tvec[-cut_off_idx]
            cell.imem = cell.imem[:, -cut_off_idx:]
            cell.vmem = cell.vmem[:, -cut_off_idx:]

        tau = '%1.2f' % taum if type(taum) in [float, int] else taum
        dist_dict = self._get_distribution(dist_dict, cell)

        if type(input_idx) is int:
            # sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f_%1.4f' % (self.cell_name, self.input_type, input_idx, mu,
            #                                             self.holding_potential, distribution, tau, weight)
            sim_name = '%s_%s_%d_%1.1f_%+d_%s_%s' % (self.cell_name, self.input_type, input_idx, mu,
                                                        self.holding_potential, distribution, tau)
        elif type(input_idx) in [list, np.ndarray]:
            sim_name = '%s_%s_multiple_%1.2f_%1.1f_%+d_%s_%s_%1.4f' % (self.cell_name, self.input_type, weight, mu,
                                                                    self.holding_potential, distribution, tau, weight)
        elif type(input_idx) is str:
            sim_name = '%s_%s_%s_%1.1f_%+d_%s_%s_%1.4f' % (self.cell_name, self.input_type, input_idx, mu,
                                                        self.holding_potential, distribution, tau, weight)
        else:
            print input_idx, type(input_idx)
            raise RuntimeError("input_idx is not recognized!")

        if hasattr(self, 'w_bar_scaling_factor'):
            sim_name += '_%1.1f' % self.w_bar_scaling_factor

        np.save(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)), cell.tvec)
        np.save(join(self.sim_folder, 'dist_dict_%s.npy' % sim_name), dist_dict)
        np.save(join(self.sim_folder, 'sig_%s.npy' % sim_name), np.dot(electrode.electrodecoeff, cell.imem))
        #np.save(join(self.sim_folder, 'vmem_%s.npy' % sim_name), cell.vmem[self.cell_plot_idxs])
        #np.save(join(self.sim_folder, 'imem_%s.npy' % sim_name), cell.imem[self.cell_plot_idxs])
        np.save(join(self.sim_folder, 'vmem_%s.npy' % sim_name), cell.vmem)
        np.save(join(self.sim_folder, 'imem_%s.npy' % sim_name), cell.imem)
        np.save(join(self.sim_folder, 'synidx_%s.npy' % sim_name), cell.synidx)

        np.save(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name), electrode.x)
        np.save(join(self.sim_folder, 'elec_y_%s.npy' % self.cell_name), electrode.y)
        np.save(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name), electrode.z)

        np.save(join(self.sim_folder, 'xstart_%s.npy' % (self.cell_name)), cell.xstart)
        np.save(join(self.sim_folder, 'ystart_%s.npy' % (self.cell_name)), cell.ystart)
        np.save(join(self.sim_folder, 'zstart_%s.npy' % (self.cell_name)), cell.zstart)
        np.save(join(self.sim_folder, 'xend_%s.npy' % (self.cell_name)), cell.xend)
        np.save(join(self.sim_folder, 'yend_%s.npy' % (self.cell_name)), cell.yend)
        np.save(join(self.sim_folder, 'zend_%s.npy' % (self.cell_name)), cell.zend)
        np.save(join(self.sim_folder, 'xmid_%s.npy' % (self.cell_name)), cell.xmid)
        np.save(join(self.sim_folder, 'ymid_%s.npy' % (self.cell_name)), cell.ymid)
        np.save(join(self.sim_folder, 'zmid_%s.npy' % (self.cell_name)), cell.zmid)
        np.save(join(self.sim_folder, 'diam_%s.npy' % (self.cell_name)), cell.diam)

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

        if self.input_type == 'white_noise':
            input_scaling = 0.0005
            max_freq = 500
            plt.seed(1234)
            input_array = input_scaling * (self._make_WN_input(cell, max_freq))
            print 1000 * np.std(input_array)
        elif self.input_type == 'real_wn':
            tot_ntsteps = round((cell.tstopms - cell.tstartms)/cell.timeres_NEURON + 1)
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
                    print "Input inserted in ", sec.name()
                    syn = neuron.h.ISyn(seg.x, sec=sec)
                    # print "Dist: ", nrn.distance(seg.x)
                i += 1
        if syn is None:
            raise RuntimeError("Wrong stimuli index")
        syn.dur = 1E9
        syn.delay = 0
        noise_vec.play(syn._ref_amp, cell.timeres_NEURON)
        return cell, syn, noise_vec

    def _run_single_wn_simulation(self, mu, input_idx, distribution, tau_w):
        plt.seed(1234)
        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        # neuron.h('forall delete_section()')
        cell = self._return_cell(self.holding_potential, 'generic', mu, distribution, tau_w)
        # self._quickplot_setup(cell, electrode)
        cell, syn, noiseVec = self._make_white_noise_stimuli(cell, input_idx)
        print "Starting simulation ..."

        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)

        self.save_neural_sim_single_input_data(cell, electrode, input_idx, mu, distribution, tau_w)

    def _run_distributed_synaptic_simulation(self, mu, input_sec, distribution, tau_w, weight):
        plt.seed(1234)

        tau = '%1.2f' % tau_w if type(tau_w) in [int, float] else tau_w
        sim_name = '%s_%s_%s_%1.1f_%+d_%s_%s_%1.4f' % (self.cell_name, self.input_type, input_sec, mu,
                                                       self.holding_potential, distribution, tau, weight)

        # if os.path.isfile(join(self.sim_folder, 'sig_%s.npy' % sim_name)):
        #     print "Skipping ", mu, input_sec, distribution, tau_w, weight, 'sig_%s.npy' % sim_name
        #     return

        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        cell = self._return_cell(self.holding_potential, 'generic', mu, distribution, tau_w)
        cell, syn, noiseVec = self._make_distributed_synaptic_stimuli(cell, input_sec, weight)
        print "Starting simulation ..."
        #import ipdb; ipdb.set_trace()
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)

        # plt.plot(cell.tvec, cell.somav)
        # plt.show()

        self.save_neural_sim_single_input_data(cell, electrode, input_sec, mu, distribution, tau_w, weight)
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
        if input_sec == 'distal_tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 900
        elif input_sec == 'tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 600
        elif input_sec == 'homogeneous':
            input_pos = ['apic', 'dend']
            maxpos = 10000
            minpos = -10000
        else:
            # input_pos = input_sec
            # maxpos = 10000
            # minpos = -10000
            raise RuntimeError("Use other input section")
        num_synapses = 1000
        cell_input_idxs = cell.get_rand_idx_area_norm(section=input_pos, nidx=num_synapses,
                                                      z_min=minpos, z_max=maxpos)
        spike_trains = LFPy.inputgenerators.stationary_poisson(num_synapses, 5, cell.tstartms, cell.tstopms)
        synapses = self.set_input_spiketrain(cell, cell_input_idxs, spike_trains, synapse_params)
        #self.input_plot(cell, cell_input_idxs, spike_trains)

        return cell, synapses, None

    def _make_distributed_synaptic_stimuli_equal_density(self, cell, input_sec, weight=0.001, **kwargs):

        # Define synapse parameters
        synapse_params = {
            'e': 0.,                   # reversal potential
            'syntype': 'ExpSyn',       # synapse type
            'tau': 2.,                # syn. time constant
            'weight': weight,            # syn. weight
            'record_current': False,
        }
        if input_sec == 'distal_tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 900
        elif input_sec == 'tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 600
        elif input_sec == 'homogeneous':
            input_pos = ['apic', 'dend']
            maxpos = 10000
            minpos = -10000
        else:
            raise RuntimeError("Use other input section")
        num_synapses = 1000
        # cell_input_idxs = cell.get_rand_idx_area_norm(section=input_pos, nidx=num_synapses,  z_min=minpos, z_max=maxpos)
        homogeneous_idxs = cell.get_rand_idx_area_norm(section=['apic', 'dend'], nidx=num_synapses,  z_min=-10000, z_max=10000)
        in_range_list = []
        for inp in homogeneous_idxs:
            if minpos <= cell.zmid[inp] <= maxpos:
                in_range_list.append(inp)

        num_synapses = len(in_range_list)
        input_idxs = np.array(in_range_list)
        if input_sec == 'homogeneous':
            if not (input_idxs == homogeneous_idxs).all():
                raise RuntimeError("Wrong input indexes!")

        spike_trains = LFPy.inputgenerators.stationary_poisson(num_synapses, 5, cell.tstartms, cell.tstopms)
        synapses = self.set_input_spiketrain(cell, input_idxs, spike_trains, synapse_params)

        return cell, synapses, None

    def set_input_spiketrain(self, cell, cell_input_idxs, spike_trains, synapse_params):
        synapse_list = []
        for number, comp_idx in enumerate(cell_input_idxs):
            synapse_params.update({'idx': int(comp_idx)})
            s = LFPy.Synapse(cell, **synapse_params)
            s.set_spike_times(spike_trains[number])
            synapse_list.append(s)
        return synapse_list

    def distribute_cellsims_MPI(self):
        """ Run with
            openmpirun -np 4 python example_mpi.py
        """
        from mpi4py import MPI
        COMM = MPI.COMM_WORLD
        SIZE = COMM.Get_size()
        RANK = COMM.Get_rank()
        weights = np.array([0.0001])
        sim_num = 0
        for weight in weights:
            for input_sec in ['homogeneous', 'tuft', 'distal_tuft']:
                for mu in self.mus[1:]:
                    if divmod(sim_num, SIZE)[1] == RANK:
                        print RANK, "simulating ", weight, mu
                        self._run_distributed_synaptic_simulation(mu, input_sec, 'linear_increase', 'auto', weight)
                sim_num += 1
        COMM.Barrier()
        print RANK, "reached this point"
        # plot_num = 0
        # for weight in weights:
        #     if divmod(plot_num, SIZE)[1] == RANK:
        #         print RANK, "plotting ", weight
        #         self.LFP_with_distance_study(weight)
        #     plot_num += 1

        # if divmod(cellindex, SIZE)[1] == RANK:
        #     self.run_all_distributed_synaptic_input_simulations(float(sys.argv[1]), float(sys.argv[2]))


if __name__ == '__main__':

    gs = GenericStudy('hay', 'distributed_synaptic', conductance='generic')
    # gs.distribute_cellsims_MPI()
    # if len(sys.argv) == 3:
    gs._run_distributed_synaptic_simulation(float(sys.argv[1]), sys.argv[2], 'linear_increase', 'auto', 0.0001)
