__author__ = 'torbjone'

import sys
import os
import numpy as np
from os.path import join
from plotting_convention import *
import scipy.fftpack as ff
import aLFP
from matplotlib.collections import LineCollection
import neuron
import LFPy
import matplotlib.patches as mpatches
from scipy import stats
nrn = neuron.h

class NeuralSimulations():

    def __init__(self, PlotClass):
        self.frequency = PlotClass.frequency if hasattr(PlotClass, 'frequency') else None
        self.sim_folder = PlotClass.sim_folder
        self.cell_name = PlotClass.cell_name
        self.root_folder = PlotClass.root_folder
        self.timeres_NEURON = PlotClass.timeres_NEURON
        self.timeres_python = PlotClass.timeres_python
        self.cut_off = PlotClass.cut_off
        self.end_t = PlotClass.end_t
        self.repeats = PlotClass.repeats if hasattr(PlotClass, 'repeats') else None
        self.stimuli = PlotClass.stimuli
        if self.stimuli is 'white_noise':
            self.stimuli_function = self._make_white_noise_stimuli
        elif self.stimuli is 'synaptic':
            self.stimuli_function = self._make_synaptic_stimuli
        elif self.stimuli is 'single_sinusoid':
            self.stimuli_function = self._make_sinusoidal_stimuli
        else:
            raise RuntimeError("Unknown stimuli")

    def save_neural_sim_data(self, cell, electrode, input_idx,
                             conductance_type, holding_potential, weight):

        if not os.path.isdir(self.sim_folder): os.mkdir(self.sim_folder)
        sim_name = '%s_%d_%s_%+d_%1.4f' % (self.cell_name, input_idx, conductance_type, holding_potential, weight)

        if not self.repeats is None:
            cut_off_idx = (len(cell.tvec) - 1) / self.repeats
            cell.tvec = cell.tvec[-cut_off_idx:] - cell.tvec[-cut_off_idx]
            cell.imem = cell.imem[:, -cut_off_idx:]
            cell.vmem = cell.vmem[:, -cut_off_idx:]

        np.save(join(self.sim_folder, 'tvec_%s.npy' % self.cell_name), cell.tvec)
        LFP = np.dot(electrode.electrodecoeff, cell.imem)
        np.save(join(self.sim_folder, 'sig_%s.npy' % sim_name), LFP)
        np.save(join(self.sim_folder, 'vmem_%s.npy' % sim_name), cell.vmem)
        np.save(join(self.sim_folder, 'imem_%s.npy' % sim_name), cell.imem)

        if not self.frequency is None:
            v_max = np.max(cell.vmem[:, len(cell.tvec)/2:], axis=1)
            v_min = np.min(cell.vmem[:, len(cell.tvec)/2:], axis=1)
            s_max = np.max(LFP[:, len(cell.tvec)/2:], axis=1)
            s_min = np.min(LFP[:, len(cell.tvec)/2:], axis=1)
            i_max = np.max(cell.imem[:, len(cell.tvec)/2:], axis=1)
            i_min = np.min(cell.imem[:, len(cell.tvec)/2:], axis=1)
            np.save(join(self.sim_folder, 'v_max_%s_%dHz.npy' % (sim_name, self.frequency)), v_max)
            np.save(join(self.sim_folder, 'v_min_%s_%dHz.npy' % (sim_name, self.frequency)), v_min)

            np.save(join(self.sim_folder, 's_max_%s_%dHz.npy' % (sim_name, self.frequency)), s_max)
            np.save(join(self.sim_folder, 's_min_%s_%dHz.npy' % (sim_name, self.frequency)), s_min)

            np.save(join(self.sim_folder, 'i_max_%s_%dHz.npy' % (sim_name, self.frequency)), i_max)
            np.save(join(self.sim_folder, 'i_min_%s_%dHz.npy' % (sim_name, self.frequency)), i_min)

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

        neuron_models = join(self.root_folder, 'neuron_models')
        neuron.load_mechanisms(join(neuron_models))
        if self.cell_name == 'hay':
            from hay_active_declarations import active_declarations as hay_active
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
            from ca1_sub_declarations import active_declarations as ca1_active
            if conductance_type == 'active':
                use_channels = ['Ih', 'Im', 'INaP']
            elif conductance_type == 'passive':
                use_channels = []
            elif conductance_type == 'no_INaP':
                use_channels = ['Ih', 'Im']
            elif conductance_type == 'no_Im':
                use_channels = ['Ih', 'INaP']
            elif conductance_type == 'active_frozen':
                use_channels = ['Ih_frozen', 'Im_frozen', 'INaP_frozen']
            elif conductance_type == 'Im_frozen':
                use_channels = ['Im_frozen']
            elif conductance_type == 'Im':
                use_channels = ['Im']
            elif conductance_type == 'Ih':
                use_channels = ['Ih']
            elif conductance_type == 'INaP':
                use_channels = ['INaP']
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
        elif self.cell_name is 'shah':
            from shah_declarations import active_declarations as shah_declarations
            neuron.load_mechanisms(join(neuron_models, 'shah'))
            cell_params = {
                'morphology': join(neuron_models, 'shah', 'geo9068802.hoc'),
                'v_init': holding_potential,             # initial crossmembrane potential
                'passive': False,           # switch on passive mechs
                'nsegs_method': 'lambda_f',# method for setting number of segments,
                'lambda_f': 100,           # segments are isopotential at this frequency
                'timeres_NEURON': self.timeres_NEURON,   # dt of LFP and NEURON simulation.
                'timeres_python': self.timeres_python,
                'tstartms': -self.cut_off,          #start time, recorders start at t=0
                'tstopms': self.end_t,
                'custom_fun': [shah_declarations],
                'custom_fun_args': [{'hold_potential': holding_potential,
                                     'conductance_type': conductance_type}]
            }
        elif self.cell_name is 'infinite_neurite':
            from infinite_neurite_active_declarations import active_declarations
            mu_1, mu_2 = np.array(conductance_type.split('_'), float)
            print mu_1, mu_2
            args = [{'mu_factor_1': mu_1, 'mu_factor_2': mu_2}]
            cell_params = {
                'morphology': join(neuron_models, self.cell_name, 'infinite_neurite.hoc'),
                'v_init': holding_potential,
                'passive': False,
                'nsegs_method': 'lambda_f',
                'lambda_f': 500.,
                'timeres_NEURON': self.timeres_NEURON,   # dt of LFP and NEURON simulation.
                'timeres_python': self.timeres_python,
                'tstartms': -self.cut_off,          # start time, recorders start at t=0
                'tstopms': self.end_t,
                'custom_fun': [active_declarations],
                'custom_fun_args': args,
            }
        else:
            raise RuntimeError("Unrecognized cell name: %s" % self.cell_name)

        cell = LFPy.Cell(**cell_params)
        return cell

    def _quickplot_comp_numbs(self, cell):
        [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], lw=1.5, color='0.5', zorder=0, alpha=1)
            for idx in xrange(len(cell.xmid))]
        [plt.scatter(cell.xmid[idx], cell.zmid[idx], marker='$%d$' % idx, s=500) for idx in xrange(cell.totnsegs)]
        plt.axis('equal')
        plt.show()

    def do_single_neural_simulation(self, conductance_type, holding_potential, input_idx,
                                     elec_x, elec_y, elec_z, weight=0.0010):
        electrode_parameters = {
            'sigma': 0.3,
            'x': elec_x,
            'y': elec_y,
            'z': elec_z
        }
        electrode = LFPy.RecExtElectrode(**electrode_parameters)
        neuron.h('forall delete_section()')
        neuron.h.celsius = 32.
        cell = self._return_cell(holding_potential, conductance_type)
        # self._quickplot_comp_numbs(cell)

        if 0:
            [plt.plot([cell.xstart[i], cell.xend[i]], [cell.zstart[i], cell.zend[i]], 'k')
             for i in xrange(cell.totnsegs)]
            [plt.text(cell.xmid[i], cell.zend[i], '%1.2f' % i, color='r') for i in xrange(cell.totnsegs)]
            plt.axis('equal')
            plt.show()
        cell, syn, noiseVec = self.stimuli_function(cell, input_idx, weight)

        # print neuron.h.celsius
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)

        print np.std(cell.vmem[input_idx]), np.mean(cell.vmem[input_idx])

        plt.close('all')
        plt.plot(cell.tvec, cell.vmem[input_idx])
        plt.show()



        self.save_neural_sim_data(cell, electrode, input_idx, conductance_type, holding_potential, weight)

    def make_WN_input(self, cell, max_freq):
        """ White Noise input ala Linden 2010 is made """
        tot_ntsteps = round((cell.tstopms - cell.tstartms) / cell.timeres_NEURON + 1)
        I = np.zeros(tot_ntsteps)
        tvec = np.arange(tot_ntsteps) * cell.timeres_NEURON
        for freq in xrange(1, max_freq + 1):
            I += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())
        return I

    def _make_white_noise_stimuli(self, cell, input_idx, weight=0.0005):
        max_freq = 500
        plt.seed(1234)
        input_array = weight * self.make_WN_input(cell, max_freq)
        noiseVec = neuron.h.Vector(input_array)

        plt.close('all')
        plt.plot(input_array)
        plt.show()
        print 1000 * np.std(input_array)
        i = 0
        syn = None
        for sec in cell.allseclist:
            for seg in sec:
                if i == input_idx:
                    print "Input inserted in ", sec.name()
                    syn = neuron.h.ISyn(seg.x, sec=sec)
                    print "Dist: ", nrn.distance(seg.x)
                i += 1
        if syn is None:
            raise RuntimeError("Wrong stimuli index")
        syn.dur = 1E9
        syn.delay = 0 #cell.tstartms
        noiseVec.play(syn._ref_amp, cell.timeres_NEURON)
        return cell, syn, noiseVec

    def _make_sinusoidal_stimuli(self, cell, input_idx):
        input_scaling = 0.001

        plt.seed(1234)
        tot_ntsteps = round((cell.tstopms - cell.tstartms)/\
                      cell.timeres_NEURON + 1)
        tvec = np.arange(tot_ntsteps) * cell.timeres_NEURON / 1000.
        input_array = input_scaling * np.sin(2 * np.pi * self.frequency * tvec)
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
        syn.delay = 0#cell.tstartms
        noiseVec.play(syn._ref_amp, cell.timeres_NEURON)
        return cell, syn, noiseVec

    def _make_synaptic_stimuli(self, cell, input_idx, weight=None):

        # Define synapse parameters
        synapse_parameters = {
            'idx': input_idx,
            'e': 0.,                   # reversal potential
            'syntype': 'ExpSyn',       # synapse type
            'tau': 2.,                # syn. time constant
            'weight': 0.001 if weight is None else weight,            # syn. weight
            'record_current': True,
        }
        synapse = LFPy.Synapse(cell, **synapse_parameters)
        synapse.set_spike_times(np.array([5.]))
        return cell, synapse, None

    def _make_distributed_synaptic_stimuli(self, cell, input_idx, **kwargs):

        # Define synapse parameters
        synapse_params = {
            'idx': input_idx,
            'e': 0.,                   # reversal potential
            'syntype': 'ExpSyn',       # synapse type
            'tau': 2.,                # syn. time constant
            'weight': 0.001,            # syn. weight
            'record_current': False,
        }
        if kwargs['input_section'] == 'tuft':
            input_pos = ['apic']
            maxpos = 10000
            minpos = 200
        elif kwargs['input_section'] == 'homogeneous':
            input_pos = ['apic', 'dend']
            maxpos = 10000
            minpos = -10000
        else:
            input_pos = kwargs['input_section']
            maxpos = 10000
            minpos = -10000

        num_synapses = 1000
        cell_input_idxs = cell.get_rand_idx_area_norm(section=input_pos, nidx=num_synapses,
                                                      z_min=minpos, z_max=maxpos)
        spike_trains = LFPy.inputgenerators.stationary_poisson(num_synapses, 5, cell.tstartms, cell.tstopms)

        synapses = self.set_input_spiketrain(cell, cell_input_idxs, spike_trains, synapse_params)
        self.input_plot(cell, cell_input_idxs, spike_trains)

        return cell, synapses, None

    def input_plot(self, cell, cell_input_idxs, spike_trains):

        plt.close('all')
        plt.subplot(121)
        plt.plot(cell.xmid, cell.zmid, 'ko')
        plt.plot(cell.xmid[cell_input_idxs], cell.zmid[cell_input_idxs], 'rD')
        plt.axis('equal')
        plt.subplot(122)
        [plt.scatter(np.ones(spike_trains.shape[1]) * idx, spike_trains[idx]) for idx in spike_trains.shape[0]]
        plt.show()


    def set_input_spiketrain(self, cell, cell_input_idxs, spike_trains, synapse_params):
        synapse_list = []
        for number, comp_idx in enumerate(cell_input_idxs):
            synapse_params.update({'idx': int(comp_idx)})
            s = LFPy.Synapse(cell, **synapse_params)
            s.set_spike_times(spike_trains[number])
            synapse_list.append(s)
        return synapse_list


class PaperFigures():
    conductance_dict = {'active': 'Active',
                                 'Ih_linearized': r'Passive + linearized $I_{\rm h}$',
                                 'passive': 'Passive',
                                 'Ih_linearized_frozen': r'Passive+frozen $I_{\rm h}$',
                                 'Ih': r'Passive+$I_{\rm h}$',
                                 'Ih_frozen': r'Passive+frozen $I_{\rm h}$',
                                 'SKv3_1_Ih': 'Passive+$I_H$+SKv3_1',
                                 'SKv3_1': 'Passive+SKv3_1',
                                 'NaP': 'Persistent sodium',
                                 'NaP_linearized': 'Linearized persistent sodium',
                                 'NaP_frozen': 'Frozen Persistent sodium',
                                 'K': 'All posassium',
                                 'reduced': 'Reduced',}
    def __init__(self):
        self.conductance_clr = {'active': 'r', 'active_frozen': 'b', 'Ih_linearized': 'g', 'passive': 'k',
                                'Ih_linearized_frozen': 'b', 'no_INaP': 'orange', 'no_Im': 'y',
                                'Ih': 'b', 'Im': 'g', 'Ih_frozen': 'purple', 'SKv3_1_Ih': 'y',
                                'SKv3_1': 'orange', 'K': 'pink', 'Na': 'gray', 'NaP': 'pink', 'reduced': 'g',
                                'NaP_frozen': 'c', 'NaP_linearized': 'gray'}

        self.conductance_style = {'active': '-', 'active_frozen': '-', 'Ih_linearized': '-', 'passive': '-',
                                'Ih_linearized_frozen': '-', 'no_INaP': '-', 'no_Im': '-',
                                'INaP': '-', 'Ih': '--', 'Im': '-', 'Ih_frozen': ':', 'SKv3_1': '-.',
                                'NaP': '--', 'Na': '-', 'SKv3_1_Ih': '-.',
                                'K': '-', 'reduced': '-', 'NaP_frozen': '-', 'NaP_linearized': '-'}


        self.figure_folder = join('/home', 'torbjone', 'work', 'aLFP', 'paper_figures')
        self.root_folder = join('/home', 'torbjone', 'work', 'aLFP')

    def _draw_morph_to_axis(self, ax, input_pos, ic_comp=None, distribution=None):

        xstart = np.load(join(self.sim_folder, 'xstart_%s_%s.npy' % (self.cell_name, self.conductance)))
        zstart = np.load(join(self.sim_folder, 'zstart_%s_%s.npy' % (self.cell_name, self.conductance)))
        xend = np.load(join(self.sim_folder, 'xend_%s_%s.npy' % (self.cell_name, self.conductance)))
        zend = np.load(join(self.sim_folder, 'zend_%s_%s.npy' % (self.cell_name, self.conductance)))
        xmid = np.load(join(self.sim_folder, 'xmid_%s_%s.npy' % (self.cell_name, self.conductance)))
        zmid = np.load(join(self.sim_folder, 'zmid_%s_%s.npy' % (self.cell_name, self.conductance)))

        if type(input_pos) is int:
            ax.plot(xmid[input_pos], zmid[input_pos], 'y*', zorder=2, ms=10)
        elif type(input_pos) in [list, np.ndarray]:
            ax.plot(xmid[input_pos], zmid[input_pos], 'g.', zorder=2, ms=3)

        if distribution is None:
            clr_list = ['0.5' for idx in xrange(len(xmid))]
        elif distribution is 'linear_increase':
            color_at_pos = lambda d: plt.cm.hot(int(256./1500 * d))
            dist = np.sqrt(xmid**2 + zmid**2)
            clr_list = [color_at_pos(dist[idx]) for idx in xrange(len(xmid))]
            if not ic_comp is None:
                ax.scatter(xmid[ic_comp], zmid[ic_comp], c='orange', edgecolor='none', s=50)
        elif distribution is 'uniform':
            clr_list = [plt.cm.hot(int(256./2)) for idx in xrange(len(xmid))]
            if not ic_comp is None:
                ax.scatter(xmid[ic_comp], zmid[ic_comp], c='orange', edgecolor='none', s=50)
        else:
            raise NotImplementedError("Implement other dists to use this")
        [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1.5, color=clr_list[idx], zorder=0, alpha=1)
         for idx in xrange(len(xmid))]


    def _draw_set_up_to_axis(self, ax, input_pos, elec_x, elec_z, ic_plot=True):

        xstart = np.load(join(self.sim_folder, 'xstart_%s.npy' % self.cell_name))
        zstart = np.load(join(self.sim_folder, 'zstart_%s.npy' % self.cell_name))
        xend = np.load(join(self.sim_folder, 'xend_%s.npy' % self.cell_name))
        zend = np.load(join(self.sim_folder, 'zend_%s.npy' % self.cell_name))
        xmid = np.load(join(self.sim_folder, 'xmid_%s.npy' % self.cell_name))
        zmid = np.load(join(self.sim_folder, 'zmid_%s.npy' % self.cell_name))

        # if type(input_pos) is int:
        ax.plot(xmid[input_pos], zmid[input_pos], 'y*', zorder=2, ms=10)

        [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1.5, color='0.7', zorder=0, alpha=1)
         for idx in xrange(len(xmid))]

        ec_arrow_dict = {'width': 2, 'lw': 1, 'clip_on': False, 'color': 'c', 'zorder': 0}
        ic_arrow_dict = {'width': 2, 'lw': 1, 'clip_on': False, 'color': 'orange', 'zorder': 0}
        arrow_dx = 60
        arrow_dz = 20

        ax.scatter(elec_x[self.elec_apic_idx], elec_z[self.elec_apic_idx], edgecolor='none', s=70, c='c')
        ax.scatter(elec_x[self.elec_soma_idx], elec_z[self.elec_soma_idx], edgecolor='none', s=70, c='c')
        ax.arrow(elec_x[self.elec_apic_idx] + 110, elec_z[self.elec_apic_idx] + 10,
                       arrow_dx, arrow_dz, **ec_arrow_dict)
        ax.arrow(elec_x[self.elec_soma_idx] + 110, elec_z[self.elec_soma_idx] - 10,
                       arrow_dx, -arrow_dz, **ec_arrow_dict)
        if ic_plot:
            ax.scatter(xmid[self.apic_idx], zmid[self.apic_idx], edgecolor='none', s=70, c='orange')
            ax.scatter(xmid[self.soma_idx], zmid[self.soma_idx], edgecolor='none', s=70, c='orange')
            ax.arrow(xmid[self.apic_idx] - 110, zmid[self.apic_idx] + 10,
                     -arrow_dx, arrow_dz, **ic_arrow_dict)
            ax.arrow(xmid[self.soma_idx] - 110, zmid[self.soma_idx] - 10,
                     -arrow_dx, -arrow_dz, **ic_arrow_dict)


    def _draw_simplified_morph_to_axis(self, ax, input_pos=None, grading=None):

        xstart = np.load(join(self.sim_folder, 'xstart_%s_%s.npy' % (self.cell_name, self.conductance)))
        zstart = np.load(join(self.sim_folder, 'zstart_%s_%s.npy' % (self.cell_name, self.conductance)))
        xend = np.load(join(self.sim_folder, 'xend_%s_%s.npy' % (self.cell_name, self.conductance)))
        zend = np.load(join(self.sim_folder, 'zend_%s_%s.npy' % (self.cell_name, self.conductance)))
        xmid = np.load(join(self.sim_folder, 'xmid_%s_%s.npy' % (self.cell_name, self.conductance)))
        zmid = np.load(join(self.sim_folder, 'zmid_%s_%s.npy' % (self.cell_name, self.conductance)))

        dist = np.sqrt(xmid**2 + zmid**2)

        if grading in [None, 'uniform']:
            color_at_pos = lambda d: plt.cm.hot(int(256./1500 * 750))
        elif grading is 'linear_increase':
            color_at_pos = lambda d: plt.cm.hot(int(256./1500 * d))
        elif grading is 'linear_decrease':
            color_at_pos = lambda d: plt.cm.hot(int(256./1500 * (np.max(dist) - d)))
        else:
            raise RuntimeError("Wrong with distribution!")
        [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1, zorder=0, c=color_at_pos(dist[idx]))
             for idx in xrange(len(xstart))]

        # [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], **lineprops) for idx in xrange(len(xstart))]
        # ax.plot(0, 0, '^', c=lineprops['color'], ms=20, mec=lineprops['color'])

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

    # def _draw_simplified_morph_to_axis(self, ax, input_pos=None, grading=None):
    #
    #     xstart = [-150, 150, 0, 0, 0]
    #     xend = [0, 0, 0, -100, 100]
    #     zstart = [-150, -150, 0, 1000, 1000]
    #     zend = [0, 0, 1000, 1100, 1100]
    #     lineprops = {'lw': 5, 'color': 'gray'}
    #
    #     color_at_pos = lambda dist: plt.cm.gray(int(256./1500 * dist))
    #     if grading is None:
    #         [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], **lineprops) for idx in xrange(len(xstart))]
    #         ax.plot(0, 0, '^', c=lineprops['color'], ms=20, mec=lineprops['color'])
    #     else:
    #         num_interpts = 10
    #         x = np.array([np.linspace(xstart[i], xend[i], num_interpts)
    #                           for i in xrange(len(xstart))]).flatten()
    #         z = np.array([np.linspace(zstart[i], zend[i], num_interpts)
    #                           for i in xrange(len(zend))]).flatten()
    #         dist = np.sqrt(x**2 + z**2)
    #         points = np.array([x, z]).T.reshape(-1, 1, 2)
    #         segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #         lc = LineCollection(segments, cmap=plt.get_cmap('gray'),
    #                     norm=plt.Normalize(0, 1500), zorder=0)
    #         if grading == 'uniform':
    #             lc.set_array(np.ones(len(z)) * 500)
    #             soma_clr = color_at_pos(500)
    #         elif grading == 'linear_increase':
    #             lc.set_array(dist)
    #             soma_clr = color_at_pos(0)
    #         elif grading == 'linear_decrease':
    #             lc.set_array(np.max(dist) - dist)
    #             soma_clr = color_at_pos(np.max(dist))
    #         else:
    #             raise RuntimeError("Unknown grading!")
    #         lc.set_linewidth(3)
    #         ax.plot(0, 0, '^', c=soma_clr, ms=10, mec=soma_clr)
    #         ax.add_collection(lc)
    #     if not input_pos is None:
    #         synapse_clr = 'g'
    #         if input_pos is 'soma':
    #             synapse_coor = [-90, 20]
    #             axon_start = [-175, 50]
    #             marker = (3, 0, 55)
    #         else:
    #             synapse_coor = [-100, 1010]
    #             axon_start = [-200, 900]
    #             marker = (3, 0, 15)
    #
    #         ax.plot([axon_start[0], synapse_coor[0]], [axon_start[1], synapse_coor[1]], lw=5, c=synapse_clr)
    #         ax.plot(synapse_coor[0], synapse_coor[1], c=synapse_clr, mec=synapse_clr, ms=15, marker=marker)
    #     ax.axis('equal')

    @staticmethod
    def _find_closest_idx_to_pos(pos, elec_x, elec_y, elec_z):
        return np.argmin(np.sum(np.array([(elec_x - pos[0])**2, (elec_y - pos[1])**2, (elec_z - pos[2])**2]), axis=0))

    @classmethod
    def _return_idxs_from_positions(cls, position_list, elec_x, elec_y, elec_z):
        return [cls._find_closest_idx_to_pos(pos, elec_x, elec_y, elec_z) for pos in position_list]

    def LFP_arrow_to_axis(self, pos, ax_origin, ax_target, c='orange'):

        upper_pixel_coor = ax_target.transAxes.transform(([0.5, .9]))
        upper_coor = ax_origin.transData.inverted().transform(upper_pixel_coor)
        upper_line_x = [pos[0], upper_coor[0]]
        upper_line_y = [pos[1], upper_coor[1]]
        ax_origin.plot(upper_line_x, upper_line_y, lw=2, c=c, clip_on=False, alpha=1.)

class IntroFigures():
    np.random.seed(0)
    conductance_clr = {'active': 'r', 'active_frozen': 'b', 'Ih_linearized': 'g', 'passive': 'k',
                       'Ih_linearized_frozen': 'c', 'NaP': 'pink', 'regenerative': 'c'}
    def __init__(self, cell_name, figure_name, weight=0.0005, do_simulations=False):
        
        self.cell_name = cell_name
        self.figure_name = figure_name
        self.figure_folder = join('/home', 'torbjone', 'work', 'aLFP', 'paper_figures')
        self.root_folder = join('/home', 'torbjone', 'work', 'aLFP')

        if figure_name == 'figure_1':
            self.stimuli = 'synaptic'
            self._set_synaptic_properties()
        elif figure_name == 'figure_2':
            self.stimuli = 'white_noise'
            self._set_white_noise_properties()
        else:
            raise RuntimeError("Unknown figure")

        self.type_name = '%s' % self.cell_name
        self.timeres_NEURON = 2**-4
        self.timeres_python = 2**-4
        self.holding_potentials = [-80, -60]
        self.weight = weight
        self._set_cell_specific_properties()
        if do_simulations:
            self._do_all_simulations()

    def _do_all_simulations(self):
        neural_sim = NeuralSimulations(self)
        for holding_potential in self.holding_potentials[1:]:
            for input_idx in [self.apic_idx, self.soma_idx]:
                for conductance_type in self.conductance_types:
                    neural_sim.do_single_neural_simulation(conductance_type, holding_potential, input_idx,
                                                           self.elec_x, self.elec_y, self.elec_z, self.weight)

    def _set_white_noise_properties(self):
        self.repeats = 2
        self.cut_off = 0
        self.end_t = 1000 * self.repeats
        self.start_t = 0
        self.sim_folder = join(self.root_folder, 'paper_simulations', 'intro_fig_white_noise')
        self.ec_ax_dict = {'frameon': True,
                           'xticks': [1e0, 1e1, 1e2],
                           'xticklabels': [],
                           'yticks': [1e-7, 1e-5, 1e-3],
                           'yticklabels': [], 'xscale': 'log', 'yscale': 'log',
                           'ylim': [1e-7, 1e-2],
                           'xlim': [1, 450]}
        self.clip_on = True

    def _set_synaptic_properties(self):

        self.start_t = 0
        self.end_t = 30
        self.cut_off = 1000
        self.sim_folder = join(self.root_folder, 'paper_simulations', 'intro_fig_synapse')
        self.stimuli = 'synaptic'
        self.ec_ax_dict = {'frameon': False,
                           'xticks': [],
                           'xticklabels': [],
                           'yticks': [],
                           'yticklabels': [],
                           'ylim': [0, 0.005],
                           'xlim': [0, 30]}
        self.clip_on = False

    def _set_cell_specific_properties(self):

        elec_x, elec_z = np.meshgrid(np.linspace(-200, 200, 7),
                                     np.linspace(-200, 1200, 15))
        self.elec_x = elec_x.flatten()
        self.elec_z = elec_z.flatten()
        self.elec_y = np.zeros(len(self.elec_x))
        self.plot_positions = np.array([self.elec_x, self.elec_y, self.elec_z]).T
        self.conductance_types = ['active', 'passive']
        self.conductance_name_dict = {'active': 'Active',
                                      'passive': 'Passive',
                                      'NaP': 'Sodium',
                                      'regenerative': 'Regenerative'}
        self.soma_idx = 0
        self.apic_idx = 852
        self.use_elec_idxs = [8, 36, 26, 67, 85]
        self.ax_dict = {'ylim': [-300, 1300], 'xlim': [-400, 400]}

    def _draw_morph_to_axis(self, ax, input_pos):
        xstart = np.load(join(self.sim_folder, 'xstart_%s.npy' % self.type_name))
        zstart = np.load(join(self.sim_folder, 'zstart_%s.npy' % self.type_name))
        xend = np.load(join(self.sim_folder, 'xend_%s.npy' % self.type_name))
        zend = np.load(join(self.sim_folder, 'zend_%s.npy' % self.type_name))
        xmid = np.load(join(self.sim_folder, 'xmid_%s.npy' % self.type_name))
        zmid = np.load(join(self.sim_folder, 'zmid_%s.npy' % self.type_name))
        if type(input_pos) is int:
            ax.plot(xmid[input_pos], zmid[input_pos], 'y*', zorder=1, ms=10)

        [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1, color='0.8', zorder=0, alpha=1)
         for idx in xrange(len(xmid))]

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
        if self.figure_name == 'figure_1':
            ax_.plot(x_vec, signal[idx, :] - signal[idx, 0], **ec_plot_dict)
        else:
            ax_.plot(x_vec, signal[idx, :], **ec_plot_dict)
        if self.figure_name == 'figure_2':
            ax_.grid(True)
            if conductance_type == 'passive' and idx == 8 and ax is fig.axes[0]:
                # ax_.set_xticks([10, 100])
                ax_.set_xticklabels(['$10^0$', '$10^1$', '$10^2$'], size=10)
                ax_.set_yticklabels(['$10^{-7}$', '', '$10^{-2}$'], size=10)
                ax_.set_xlabel('Hz', size=10)
                ax_.set_ylabel('PSD', size=7)

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
        sim_name = '%s_%d_%s_%+d_%1.4f' % (self.cell_name, input_idx, conductance_type, holding_potential, self.weight)
        LFP = 1e3*np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))
        # vmem = np.load(join(self.sim_folder, 'vmem_%s.npy' % sim_name))
        # print vmem
        # print np.max(vmem), np.min(vmem), np.std(vmem)
        # print input_idx
        # f2 = plt.figure()
        # ax2 = f2.add_subplot(111)
        # ax2.plot(tvec, vmem[input_idx])
        # ax2.plot(tvec, vmem[0])
        # plt.show()
        # plt.close(f2)
        ax.scatter(elec_x[self.use_elec_idxs], elec_z[self.use_elec_idxs], c='cyan', s=15, edgecolor='none')
        if self.figure_name == 'figure_1':
            x_vec, y_vec = tvec, LFP
        elif self.figure_name == 'figure_2':
            x_vec, y_vec = aLFP.return_freq_and_psd(tvec, LFP)

        else:
            raise RuntimeError("Unknown figure name: %s" % self.figure_name)

        for idx in self.use_elec_idxs:
            if conductance_type == 'active' and input_idx == self.apic_idx:
                print conductance_type, input_idx, holding_potential, elec_x[idx], elec_z[idx], x_vec[np.argmax(y_vec[idx, 1:]) + 1], 'Hz'
            self._plot_EC_signal_to_ax(fig, ax, idx, y_vec, elec_x[idx], elec_z[idx], x_vec, conductance_type)

    def make_figure(self):
        # self.conductance_types = ['regenerative', 'NaP', 'active', 'passive']
        tvec = np.load(join(self.sim_folder, 'tvec_%s.npy' % self.type_name))
        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.type_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.type_name))

        if not elec_x.shape == self.plot_positions[:, 0].shape:
            raise RuntimeError('Loaded elec_x shape: %s, Current elec_x shape: %s' %
                               (elec_x.shape, self.plot_positions[:, 0].shape))

        start_idx = 0#np.argmin(np.abs(tvec - self.start_t))
        end_idx = len(tvec)# np.argmin(np.abs(tvec - self.end_t))
        #tvec = tvec[start_idx:end_idx]

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
        fig.text(0.25, .95, '%d mV' % self.holding_potentials[0], rotation='horizontal', ha='center', size=20)
        fig.text(0.75, .95, '%d mV' % self.holding_potentials[1], rotation='horizontal', ha='center', size=20)

        ax1.plot([100, 300], [-300, -300], lw=3, clip_on=False, c='k')
        ax1.text(200, -350, '200 $\mu m$', va='top', ha='center')

        ax1.arrow(450, -100, 70, 0, lw=1, head_width=20, head_length=17, color='k', clip_on=False)
        ax1.arrow(450, -100, 0, 100, lw=1, head_width=15, head_length=25, color='k', clip_on=False)
        ax1.text(400, -50, 'z', size=10, ha='center', va='center', clip_on=False)
        ax1.text(480, -150, 'x', size=10, ha='center', va='center', clip_on=False)


        ax_numb = 0
        for input_idx in [self.apic_idx, self.soma_idx]:
            for holding_potential in self.holding_potentials:
                for conductance_type in self.conductance_types:
                    # name = 'Soma' if input_idx == 0 else 'Apic'
                    # ax_list[ax_numb].text(0, self.ax_dict['ylim'][1], '%s %d mV' % (name, holding_potential),
                    #                       horizontalalignment='center')
                    # if not ax_numb in [1,3]:
                    #     continue
                    self._plot_one_signal(input_idx, conductance_type, holding_potential, start_idx, end_idx,
                                         ax_list[ax_numb], fig, elec_x, elec_z, tvec)
                ax_numb += 1

        lines = []
        line_names = []
        for conductance_type in self.conductance_types:
            l, = plt.plot(0, 0, color=self.conductance_clr[conductance_type], lw=2)
            lines.append(l)
            line_names.append(self.conductance_name_dict[conductance_type])

        if self.figure_name is 'figure_1':
            bar_ax = fig.add_axes(self.return_ax_coors(fig, ax4, (-500, -300)), **self.ec_ax_dict)
            bar_ax.axis('off')
            bar_ax.plot([0, 0], [0, bar_ax.axis()[3]*2], lw=3, color='k', clip_on=False)
            bar_ax.plot(bar_ax.axis()[:2], [0, 0], lw=3, color='k', clip_on=False)
            bar_ax.text(2, bar_ax.axis()[2] + (bar_ax.axis()[3] - bar_ax.axis()[2])/2, '%1.2f $\mu V$'
                                            % (bar_ax.axis()[3] * 2), verticalalignment='bottom')
            bar_ax.text(2, bar_ax.axis()[2] - 0.0005, '%d $ms$' % (bar_ax.axis()[1] - bar_ax.axis()[0]),
                        verticalalignment='top')

        mark_subplots(ax_list, xpos=0.12, ypos=0.9)
        [ax.axis('off') for ax in ax_list]

        fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=5)
        fig.savefig(join(self.figure_folder, '%s.png' % self.figure_name), dpi=200)
        fig.savefig(join(self.figure_folder, '%s.pdf' % self.figure_name), dpi=200)

class NewIntroFigure(PaperFigures):
    np.random.seed(0)
    conductance_clr = {'active': 'r', 'active_frozen': 'b', 'Ih_linearized': 'g', 'passive': 'k',
                       'Ih_linearized_frozen': 'c', 'NaP': 'pink', 'regenerative': 'c'}
    def __init__(self, cell_name, figure_name, weight=0.0005, do_simulations=False):

        PaperFigures.__init__(self)
        self.cell_name = cell_name
        self.figure_name = figure_name
        self.figure_folder = join('/home', 'torbjone', 'work', 'aLFP', 'paper_figures')
        self.root_folder = join('/home', 'torbjone', 'work', 'aLFP')

        self.type_name = '%s' % self.cell_name
        self.timeres_NEURON = 2**-4
        self.timeres_python = 2**-4
        self.holding_potentials = [-80, -60]
        self.weight = weight
        self._set_cell_specific_properties()

    def _set_white_noise_properties(self):
        self.repeats = 2
        self.cut_off = 0
        self.end_t = 1000 * self.repeats
        self.start_t = 0
        self.sim_folder = join(self.root_folder, 'paper_simulations', 'intro_fig_white_noise')
        self.ec_ax_dict = {'frameon': True,
                           'xticks': [1e0, 1e1, 1e2],
                           'xticklabels': [],
                           'yticks': [1e-7, 1e-5, 1e-3],
                           'yticklabels': [], 'xscale': 'log', 'yscale': 'log',
                           'ylim': [1e-7, 1e-2],
                           'xlim': [1, 450]}
        self.clip_on = True

    def _set_synaptic_properties(self):

        self.start_t = 0
        self.end_t = 30
        self.cut_off = 1000
        self.sim_folder = join(self.root_folder, 'paper_simulations', 'intro_fig_synapse')
        self.stimuli = 'synaptic'
        self.ec_ax_dict = {'frameon': False,
                           'xticks': [],
                           'xticklabels': [],
                           'yticks': [],
                           'yticklabels': [],
                           'ylim': [0, 0.005],
                           'xlim': [0, 30]}
        self.clip_on = False

    def _set_cell_specific_properties(self):

        elec_x, elec_z = np.meshgrid(np.linspace(-200, 200, 7),
                                     np.linspace(-200, 1200, 15))
        self.elec_x = elec_x.flatten()
        self.elec_z = elec_z.flatten()
        self.elec_y = np.zeros(len(self.elec_x))
        self.plot_positions = np.array([self.elec_x, self.elec_y, self.elec_z]).T
        self.conductance_types = ['active', 'passive']
        self.conductance_name_dict = {'active': 'Active',
                                      'passive': 'Passive',
                                      'NaP': 'Sodium',
                                      'regenerative': 'Regenerative'}
        self.soma_idx = 0
        self.apic_idx = 852
        self.use_elec_idxs = [8, 36, 26, 67, 85]
        self.ax_dict = {'ylim': [-300, 1300], 'xlim': [-400, 400]}

    def _draw_morph_to_axis(self, ax, input_pos):
        xstart = np.load(join(self.sim_folder, 'xstart_%s.npy' % self.type_name))
        zstart = np.load(join(self.sim_folder, 'zstart_%s.npy' % self.type_name))
        xend = np.load(join(self.sim_folder, 'xend_%s.npy' % self.type_name))
        zend = np.load(join(self.sim_folder, 'zend_%s.npy' % self.type_name))
        xmid = np.load(join(self.sim_folder, 'xmid_%s.npy' % self.type_name))
        zmid = np.load(join(self.sim_folder, 'zmid_%s.npy' % self.type_name))
        if type(input_pos) is int:
            ax.plot(xmid[input_pos], zmid[input_pos], 'y*', zorder=1, ms=10)

        [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=1, color='0.7', zorder=0, alpha=1)
         for idx in xrange(len(xmid))]

    @staticmethod
    def _find_closest_idx_to_pos(pos, elec_x, elec_y, elec_z):
        return np.argmin(np.sum(np.array([(elec_x - pos[0])**2, (elec_y - pos[1])**2, (elec_z - pos[2])**2]), axis=0))

    @classmethod
    def _return_idxs_from_positions(cls, position_list, elec_x, elec_y, elec_z):
        return [cls._find_closest_idx_to_pos(pos, elec_x, elec_y, elec_z) for pos in position_list]

    def _plot_EC_signal_to_ax(self, fig, ax, idx, signal, elec_x_pos, elec_z_pos, x_vec, conductance_type, input_type):

        ec_plot_dict = {'color': self.conductance_clr[conductance_type],
                        'lw': 1,
                        'clip_on': self.clip_on}
        ax_ = fig.add_axes(self.return_ax_coors(fig, ax, (elec_x_pos, elec_z_pos), input_type), **self.ec_ax_dict)

        ax_.minorticks_off()
        simplify_axes(ax_)
        ax_.patch.set_alpha(0.0)
        if input_type is 'synaptic':
            ax_.plot(x_vec, signal[idx, :] - signal[idx, 0], **ec_plot_dict)
        else:
            ax_.plot(x_vec, signal[idx, :], **ec_plot_dict)
        if input_type is 'white_noise':
            ax_.grid(True)
            if conductance_type == 'passive' and idx == 8 and len(fig.axes) == 39: # Ugly hack to only get these labels once
                # ax_.set_xticks([10, 100])
                ax_.set_xticklabels(['$10^0$', '$10^1$', '$10^2$'], size=7)
                ax_.set_yticklabels(['$10^{-7}$', '', '$10^{-2}$'], size=7)
                ax_.set_xlabel('Hz', size=7)
                ax_.set_ylabel('PSD [$\mu$V$^2$/Hz]', size=7)

    def return_ax_coors(self, fig, mother_ax, pos, input_type):
        if input_type is "synaptic":
            ax_w = 0.1 / 3
            ax_h = 0.03
        else:
            ax_w = 0.12 / 3
            ax_h = 0.06
        xstart, ystart = fig.transFigure.inverted().transform(mother_ax.transData.transform(pos))
        return [xstart, ystart, ax_w, ax_h]

    def _plot_one_signal(self, input_idx, conductance_type, holding_potential, start_idx, end_idx,
                        ax, fig, elec_x, elec_z, tvec, input_type):
        sim_name = '%s_%d_%s_%+d_%1.4f' % (self.cell_name, input_idx, conductance_type, holding_potential, self.weight)
        LFP = 1e3*np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))

        ax.scatter(elec_x[self.use_elec_idxs], elec_z[self.use_elec_idxs], c='cyan', s=25, edgecolor='none')
        if input_type is 'synaptic':
            x_vec, y_vec = tvec, LFP
        elif input_type is 'white_noise':
            x_vec, y_vec = aLFP.return_freq_and_psd(tvec, LFP)
        else:
            raise RuntimeError("Unknown figure name: %s" % self.figure_name)

        for idx in self.use_elec_idxs:
            self._plot_EC_signal_to_ax(fig, ax, idx, y_vec, elec_x[idx], elec_z[idx], x_vec, conductance_type, input_type)

    def make_figure(self):

        plt.close('all')
        self.fig = plt.figure(figsize=[10, 5])
        self.fig.subplots_adjust(hspace=0.15, wspace=0.15, top=0.9, bottom=0.1, left=0.0, right=0.95)

        # ax1.text(190, -300, '200 $\mu$m', verticalalignment='center', horizontalalignment='right')
        self.fig.text(0.01, 0.75, 'Apical input', rotation='vertical', va='center', size=15)
        self.fig.text(0.01, 0.25, 'Somatic input', rotation='vertical', va='center', size=15)
        self.fig.text(0.08, .95, '%d mV' % self.holding_potentials[0], rotation='horizontal', ha='center', size=15)
        self.fig.text(0.23, .95, '%d mV' % self.holding_potentials[1], rotation='horizontal', ha='center', size=15)

        self.fig.text(0.37, 0.75, 'Apical input', rotation='vertical', va='center', size=15)
        self.fig.text(0.37, 0.25, 'Somatic input', rotation='vertical', va='center', size=15)
        self.fig.text(0.43, .95, '%d mV' % self.holding_potentials[0], rotation='horizontal', ha='center', size=15)
        self.fig.text(0.58, .95, '%d mV' % self.holding_potentials[1], rotation='horizontal', ha='center', size=15)


        self.make_panel_AB('synaptic')
        self.make_panel_AB('white_noise')
        self.make_panel_C()

        # self.fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=5)
        self.fig.savefig(join(self.figure_folder, '%s.png' % self.figure_name), dpi=200)
        self.fig.savefig(join(self.figure_folder, '%s.pdf' % self.figure_name), dpi=200)

    def make_panel_C(self):
        self.sim_folder = join(self.root_folder, 'paper_simulations', self.cell_name)
        self.conductance_types = ['active', 'passive', 'Ih', 'Ih_frozen']

        elec_x, elec_z = np.meshgrid(np.linspace(-200, 200, 7), np.linspace(-200, 1200, 15))
        self.elec_x = elec_x.flatten()
        self.elec_z = elec_z.flatten()
        self.elec_y = np.zeros(len(self.elec_x))
        self.plot_positions = np.array([self.elec_x, self.elec_y, self.elec_z]).T
        self.soma_idx = 0
        self.apic_idx = 852
        self.elec_apic_idx = 88
        self.elec_soma_idx = 18


        input_idx = self.apic_idx
        holding_potential = -80
        weight = self.weight
        tvec = np.load(join(self.sim_folder, 'tvec_%s.npy' % self.cell_name))
        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name))
        ax_dict = {'xlim': [1e0, 450], 'xlabel': 'Hz',}
        # print input_idx, holding_potential

        ax_morph = plt.axes([0.69, 0.05, 0.12, 0.6])
        ax_morph.axis('off')
        self.fig.text(0.7, 0.67, '%d mV' % holding_potential)

        self._draw_set_up_to_axis(ax_morph, input_idx, elec_x, elec_z, ic_plot=False)

        ec_ax_a = self.fig.add_axes([0.88, 0.45, 0.1, 0.2], ylim=[1e-5, 1e-1], **ax_dict)
        ec_ax_s = self.fig.add_axes([0.88, 0.05, 0.1, 0.2], ylim=[1e-7, 1e-3], **ax_dict)
        ec_ax_a.set_ylabel('PSD [$\mu$V$^2$/Hz]', size=9)
        ec_ax_s.set_ylabel('PSD [$\mu$V$^2$/Hz]', size=9)
        sig_ax_list = [ec_ax_s, ec_ax_a]
        [ax.grid(True) for ax in sig_ax_list]

        mark_subplots(ax_morph, 'C', xpos=0, ypos=1.1)

        for conductance_type in self.conductance_types:
            self._plot_sigs_C(input_idx, conductance_type, holding_potential,
                            sig_ax_list, tvec, elec_x, elec_z, weight)
        simplify_axes(self.fig.axes)
        lines = []
        line_names = []
        for conductance_type in self.conductance_types:
            l, = plt.plot(0, 0, color=self.conductance_clr[conductance_type], lw=2,
                          linestyle=self.conductance_style[conductance_type])
            lines.append(l)
            line_names.append(self.conductance_dict[conductance_type])
        [ax.set_yticks([ax.get_yticks()[1], ax.get_yticks()[-2]]) for ax in sig_ax_list]
        self.fig.legend(lines, line_names, frameon=False, loc='upper right', ncol=1, fontsize=12)

        #self.fig2.savefig(join(self.figure_folder, 'test_LFP_%d_%d.png' % (input_idx, holding_potential)), dpi=300)

    def _plot_sigs_C(self, input_idx, conductance_type, holding_potential, axes, tvec, elec_x, elec_z, weight):

        LFP = 1000 * np.load(join(self.sim_folder, 'sig_%s_%d_%s_%+d_%1.4f.npy' %
                                  (self.cell_name, input_idx, conductance_type, holding_potential, weight)))

        line_dict = {'c': self.conductance_clr[conductance_type],
                     'linestyle': self.conductance_style[conductance_type],
                     'lw': 2}

        if 0:
            num_rows = 15
            num_cols = 7
            for numb in xrange(len(elec_x)):
                ax = self.fig2.add_subplot(num_rows, num_cols, numb + 1, title='x=%d, z=%d' % (elec_x[numb], elec_z[numb]),
                                      xlim=[1e0, 5e2], ylim=[1e-5, 1e-2])

                simplify_axes(ax)
                freqs, LFP_psd = aLFP.return_freq_and_psd(tvec, LFP[numb, :])
                ax.loglog(freqs, LFP_psd[0], **line_dict)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        freqs, LFP_psd_soma = aLFP.return_freq_and_psd(tvec, LFP[self.elec_soma_idx, :])
        freqs, LFP_psd_apic = aLFP.return_freq_and_psd(tvec, LFP[self.elec_apic_idx, :])

        axes[0].loglog(freqs, LFP_psd_soma[0], **line_dict)
        axes[1].loglog(freqs, LFP_psd_apic[0], **line_dict)

    def make_panel_AB(self, input_type):

        if input_type is 'synaptic':
            ax_shift = 0
            self._set_synaptic_properties()
            ax_name = 'A'
        else:
            ax_shift = 0.35
            self._set_white_noise_properties()
            ax_name = 'B'

        tvec = np.load(join(self.sim_folder, 'tvec_%s.npy' % self.type_name))
        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.type_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.type_name))

        if not elec_x.shape == self.plot_positions[:, 0].shape:
            raise RuntimeError('Loaded elec_x shape: %s, Current elec_x shape: %s' %
                               (elec_x.shape, self.plot_positions[:, 0].shape))

        start_idx = 0#np.argmin(np.abs(tvec - self.start_t))
        end_idx = len(tvec)# np.argmin(np.abs(tvec - self.end_t))
        #tvec = tvec[start_idx:end_idx]


        ax1 = plt.axes([0.01 + ax_shift, 0.55, 0.15, 0.37], **self.ax_dict)
        ax2 = plt.axes([0.15 + ax_shift, 0.55, 0.15, 0.37], **self.ax_dict)
        ax3 = plt.axes([0.01 + ax_shift, 0.05, 0.15, 0.37], **self.ax_dict)
        ax4 = plt.axes([0.15 + ax_shift, 0.05, 0.15, 0.37], **self.ax_dict)

        # ax1 = plt.subplot(2, 6, 1 + ax_shift, **self.ax_dict)
        # ax2 = plt.subplot(2, 6, 2 + ax_shift, **self.ax_dict)
        # ax3 = plt.subplot(2, 6, 7 + ax_shift, **self.ax_dict)
        # ax4 = plt.subplot(2, 6, 8 + ax_shift, **self.ax_dict)

        mark_subplots(ax1, ax_name, xpos=0.1)

        ax_list = [ax1, ax2, ax3, ax4]

        [self._draw_morph_to_axis(ax, self.apic_idx) for ax in [ax1, ax2]]
        [self._draw_morph_to_axis(ax, self.soma_idx) for ax in [ax3, ax4]]


        ax_numb = 0
        for input_idx in [self.apic_idx, self.soma_idx]:
            for holding_potential in self.holding_potentials:
                for conductance_type in self.conductance_types:
                    # name = 'Soma' if input_idx == 0 else 'Apic'
                    # ax_list[ax_numb].text(0, self.ax_dict['ylim'][1], '%s %d mV' % (name, holding_potential),
                    #                       horizontalalignment='center')
                    # if not ax_numb in [1,3]:
                    #     continue
                    self._plot_one_signal(input_idx, conductance_type, holding_potential, start_idx, end_idx,
                                         ax_list[ax_numb], self.fig, elec_x, elec_z, tvec, input_type)
                ax_numb += 1

        lines = []
        line_names = []
        for conductance_type in self.conductance_types:
            l, = plt.plot(0, 0, color=self.conductance_clr[conductance_type], lw=2)
            lines.append(l)
            line_names.append(self.conductance_name_dict[conductance_type])

        if input_type is 'synaptic':
            ax1.plot([100, 300], [-300, -300], lw=2, clip_on=False, c='k')
            ax1.text(200, -350, '200 $\mu$m', va='top', ha='center', size=10)

            ax1.arrow(450, -100, 70, 0, lw=1, head_width=20, head_length=17, color='k', clip_on=False)
            ax1.arrow(450, -100, 0, 100, lw=1, head_width=15, head_length=25, color='k', clip_on=False)
            ax1.text(400, -50, 'z', size=10, ha='center', va='center', clip_on=False)
            ax1.text(480, -150, 'x', size=10, ha='center', va='center', clip_on=False)

            bar_ax = self.fig.add_axes(self.return_ax_coors(self.fig, ax4, (-600, -300), input_type), **self.ec_ax_dict)
            bar_ax.axis('off')
            bar_ax.plot([0, 0], [0, bar_ax.axis()[3]*2], lw=2, color='k', clip_on=False)
            bar_ax.plot(bar_ax.axis()[:2], [0, 0], lw=2, color='k', clip_on=False)
            bar_ax.text(2, bar_ax.axis()[2] + (bar_ax.axis()[3] - bar_ax.axis()[2])/2, '%1.2f $\mu$V'
                                            % (bar_ax.axis()[3] * 2), verticalalignment='bottom', size=10)
            bar_ax.text(2, bar_ax.axis()[2] - 0.0007, '%d ms' % (bar_ax.axis()[1] - bar_ax.axis()[0]),
                        verticalalignment='top', size=10)

        # mark_subplots(ax_list, xpos=0.12, ypos=0.9)
        [ax.axis('off') for ax in ax_list]

class Figure3(PaperFigures):

    def __init__(self, weight, do_simulations=False):
        PaperFigures.__init__(self)
        self.cell_name = 'hay'
        self.figure_name = 'figure_3'
        self.sim_folder = join(self.root_folder, 'paper_simulations', self.cell_name)
        # self.timeres = 2**-5
        self.conductance_dict = {'active': 'Active',
                                 'Ih_linearized': 'Passive + linearized $I_H$',
                                 'passive': 'Passive',
                                 'Ih_linearized_frozen': 'Passive+frozen $I_H$',
                                 'Ih': 'Passive+$I_H$',
                                 'Ih_frozen': 'Passive+frozen $I_H$',
                                 'SKv3_1_Ih': 'Passive+$I_H$+SKv3_1',
                                 'SKv3_1': 'Passive+SKv3_1',
                                 'NaP': 'Persistent sodium',
                                 'NaP_linearized': 'Linearized persistent sodium',
                                 'NaP_frozen': 'Frozen Persistent sodium',
                                 'K': 'All posassium',
                                 'reduced': 'Reduced',}
        self.holding_potentials = [-80, -60]
        self.timeres_NEURON = 2**-4
        self.timeres_python = 2**-4
        self.cut_off = 0
        self.repeats = 2
        self.end_t = 1000 * self.repeats
        self.stimuli = 'white_noise'
        self.conductance_types = ['active', 'passive', 'Ih', 'Ih_frozen', 'SKv3_1_Ih']
        if self.cell_name == 'hay':
            elec_x, elec_z = np.meshgrid(np.linspace(-200, 200, 7), np.linspace(-200, 1200, 15))
            self.elec_x = elec_x.flatten()
            self.elec_z = elec_z.flatten()
            self.elec_y = np.zeros(len(self.elec_x))
            self.plot_positions = np.array([self.elec_x, self.elec_y, self.elec_z]).T
            self.soma_idx = 0
            self.apic_idx = 852
            self.elec_apic_idx = 88
            self.elec_soma_idx = 18
        elif self.cell_name == 'n120':
            elec_x, elec_z = np.meshgrid(np.linspace(-150, 150, 7), np.linspace(-150, 800, 15))
            self.elec_x = elec_x.flatten()
            self.elec_z = elec_z.flatten()
            self.elec_y = np.zeros(len(self.elec_x))
            self.plot_positions = np.array([self.elec_x, self.elec_y, self.elec_z]).T
            self.soma_idx = 0
            self.apic_idx = 685
            self.use_elec_idxs = [33, 2, 78, 61, 22]
            self.ax_dict = {'ylim': [-200, 700], 'xlim': [-250, 250]}
            self.elec_apic_idx = 65
            self.elec_soma_idx = 18
        elif self.cell_name == 'c12861':
            elec_x, elec_z = np.meshgrid(np.linspace(-150, 150, 7), np.linspace(-150, 800, 15))
            self.elec_x = elec_x.flatten()
            self.elec_z = elec_z.flatten()
            self.elec_y = np.zeros(len(self.elec_x))
            self.plot_positions = np.array([self.elec_x, self.elec_y, self.elec_z]).T
            self.soma_idx = 0
            self.apic_idx = 963
            self.use_elec_idxs = [26, 2, 77, 68, 28]#[19, 8, 23, 71, 52]
            self.ax_dict = {'ylim': [-200, 750], 'xlim': [-300, 300]}
            self.elec_apic_idx = 75
            self.elec_soma_idx = 18
        elif self.cell_name == 'shah':
            elec_x, elec_z = np.meshgrid(np.linspace(-150, 150, 7), np.linspace(-150, 800, 15))
            self.elec_x = elec_x.flatten()
            self.elec_z = elec_z.flatten()
            self.elec_y = np.zeros(len(self.elec_x))
            self.plot_positions = np.array([self.elec_x, self.elec_y, self.elec_z]).T
            self.soma_idx = 0
            self.apic_idx = 443
            self.use_elec_idxs = [26, 2, 77, 68, 28]#[19, 8, 23, 71, 52]
            self.ax_dict = {'ylim': [-200, 750], 'xlim': [-300, 300]}
            self.elec_apic_idx = 75
            self.elec_soma_idx = 18
        else:
            raise ValueError("Unknown cell_name")

        if do_simulations:
            self._do_all_simulations(weight)
        self.conductance_types = ['active', 'passive', 'Ih', 'Ih_frozen']
        #self.make_figure(self.apic_idx, -60, weight)
        #self.make_figure(self.soma_idx, -60, weight)
        self.make_figure(self.apic_idx, -80, weight)
        # self.make_figure(self.soma_idx, -80)

    def _do_all_simulations(self, weight):
        neural_sim = NeuralSimulations(self)
        for holding_potential in [-80.]:
            for input_idx in [self.apic_idx]:#, self.apic_idx]:
                for conductance_type in self.conductance_types:
                    neural_sim.do_single_neural_simulation(conductance_type, holding_potential, input_idx,
                                                           self.elec_x, self.elec_y, self.elec_z, weight)

    def make_figure(self, input_idx, holding_potential, weight):
        tvec = np.load(join(self.sim_folder, 'tvec_%s.npy' % self.cell_name))
        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name))
        ax_dict = {'xlim': [1e0, 450], 'xlabel': 'Hz',}
        # print input_idx, holding_potential
        plt.close('all')
        #self.fig2 = plt.figure(figsize=[12, 16])
        fig = plt.figure(figsize=[3, 5])
        fig.subplots_adjust(hspace=0.55, wspace=0.55, bottom=0.2)
        ax_morph = plt.subplot(121, aspect=1)
        ax_morph.axis('off')
        fig.text(0.58, 0.93, '%d mV' % holding_potential)
        # fig.text(0.005, 0.8, 'Apical', rotation='vertical')
        # fig.text(0.005, 0.4, 'Somatic', rotation='vertical')
        self._draw_set_up_to_axis(ax_morph, input_idx, elec_x, elec_z, ic_plot=False)

        # vm_ax_a = fig.add_subplot(2, 4, 1, ylim=[1e-2, 1e4], title='$V_m$', ylabel='PSD',  **ax_dict)
        # vm_ax_s = fig.add_subplot(2, 4, 5, ylim=[1e-5, 1e1], title='$V_m$', **ax_dict)
        # im_ax_a = fig.add_subplot(2, 4, 2, ylim=[1e-5, 1e-1], title='$I_m$', **ax_dict)
        # im_ax_s = fig.add_subplot(2, 4, 6, ylim=[1e-10, 1e-6], title='$I_m$', **ax_dict)
        ec_ax_a = fig.add_subplot(2, 2, 2, ylim=[1e-5, 1e-1], title='$\Phi$', **ax_dict)
        ec_ax_s = fig.add_subplot(2, 2, 4, ylim=[1e-7, 1e-3], title='$\Phi$', **ax_dict)

        # vm_ax_a = fig.add_subplot(2, 4, 1, title='$V_m$', ylabel='mV',  **ax_dict)
        # vm_ax_s = fig.add_subplot(2, 4, 5, title='$V_m$', ylabel='mV', **ax_dict)
        # im_ax_a = fig.add_subplot(2, 4, 2, title='$I_m$', ylabel='nA', **ax_dict)
        # im_ax_s = fig.add_subplot(2, 4, 6, title='$I_m$', ylabel='nA', **ax_dict)
        # ec_ax_a = fig.add_subplot(2, 4, 4, title='$\Phi$', ylabel='$\mu$V', **ax_dict)
        # ec_ax_s = fig.add_subplot(2, 4, 8, title='$\Phi$', ylabel='$\mu$V', **ax_dict)

        sig_ax_list = [ec_ax_s, ec_ax_a]
        [ax.grid(True) for ax in sig_ax_list]
        # mark_subplots([vm_ax_a, im_ax_a, vm_ax_s, im_ax_s])
        # mark_subplots([ec_ax_a, ec_ax_s], 'FG')
        mark_subplots(ax_morph, 'C', xpos=0, ypos=1)

        for conductance_type in self.conductance_types:
            self._plot_sigs(input_idx, conductance_type, holding_potential,
                            sig_ax_list, tvec, elec_x, elec_z, weight)
        simplify_axes(fig.axes)
        lines = []
        line_names = []
        for conductance_type in self.conductance_types:
            l, = plt.plot(0, 0, color=self.conductance_clr[conductance_type], lw=2,
                          linestyle=self.conductance_style[conductance_type])
            lines.append(l)
            line_names.append(self.conductance_dict[conductance_type])
        [ax.set_yticks([ax.get_yticks()[1], ax.get_yticks()[-2]]) for ax in fig.axes]
        fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=5)
        #self.fig2.legend(lines, line_names, frameon=False, loc='lower center', ncol=5)
        fig.savefig(join(self.figure_folder, '%s.png' % (self.figure_name)), dpi=150)
        fig.savefig(join(self.figure_folder, '%s.pdf' % (self.figure_name)), dpi=150)
        #self.fig2.savefig(join(self.figure_folder, 'test_LFP_%d_%d.png' % (input_idx, holding_potential)), dpi=300)

    def _plot_sigs(self, input_idx, conductance_type, holding_potential, axes, tvec, elec_x, elec_z, weight):
        # vmem = np.load(join(self.sim_folder, 'vmem_%s_%d_%s_%+d_%1.4f.npy' %
        #                     (self.cell_name, input_idx, conductance_type, holding_potential, weight)))
        # imem = np.load(join(self.sim_folder, 'imem_%s_%d_%s_%+d_%1.4f.npy' %
        #                     (self.cell_name, input_idx, conductance_type, holding_potential, weight)))
        LFP = 1000 * np.load(join(self.sim_folder, 'sig_%s_%d_%s_%+d_%1.4f.npy' %
                                  (self.cell_name, input_idx, conductance_type, holding_potential, weight)))

        # cut_off_idx = 0#(vmem.shape[1] - 1) / self.repeats

        # plt.close('all')
        # plt.plot(tvec, imem[self.soma_idx, :])
        # plt.show()
        line_dict = {'c': self.conductance_clr[conductance_type],
                     'linestyle': self.conductance_style[conductance_type],
                     'lw': 2}

        if 0:
            num_rows = 15
            num_cols = 7
            for numb in xrange(len(elec_x)):
                ax = self.fig2.add_subplot(num_rows, num_cols, numb + 1, title='x=%d, z=%d' % (elec_x[numb], elec_z[numb]),
                                      xlim=[1e0, 5e2], ylim=[1e-5, 1e-2])
                #ax.axis('off')
                #ax.grid(True)
                simplify_axes(ax)
                freqs, LFP_psd = aLFP.return_freq_and_psd(tvec, LFP[numb, :])
                ax.loglog(freqs, LFP_psd[0], **line_dict)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        # axes[0].set_title('Vm %1.2f (%1.2f)' % (np.mean(vmem[self.soma_idx, -cut_off_idx:]),
        #                                                 np.std(vmem[self.soma_idx, -cut_off_idx:])))
        # axes[1].set_title('Vm %1.2f (%1.2f)' % (np.mean(vmem[self.apic_idx, -cut_off_idx:]),
        #                                                 np.std(vmem[self.apic_idx, -cut_off_idx:])))
        # freqs, vmem_psd_soma = aLFP.return_freq_and_psd(tvec, vmem[self.soma_idx, :])
        # freqs, vmem_psd_apic = aLFP.return_freq_and_psd(tvec, vmem[self.apic_idx, :])
        # freqs, imem_psd_soma = aLFP.return_freq_and_psd(tvec, imem[self.soma_idx, :])
        # freqs, imem_psd_apic = aLFP.return_freq_and_psd(tvec, imem[self.apic_idx, :])
        freqs, LFP_psd_soma = aLFP.return_freq_and_psd(tvec, LFP[self.elec_soma_idx, :])
        freqs, LFP_psd_apic = aLFP.return_freq_and_psd(tvec, LFP[self.elec_apic_idx, :])


        # axes[0].loglog(freqs, vmem_psd_soma[0], **line_dict)
        # axes[1].loglog(freqs, vmem_psd_apic[0], **line_dict)
        # axes[2].loglog(freqs, imem_psd_soma[0], **line_dict)
        # axes[3].loglog(freqs, imem_psd_apic[0], **line_dict)
        # axes[2].loglog(freqs, imem_psd_soma[0], **line_dict)
        # axes[3].loglog(freqs, imem_psd_apic[0], **line_dict)
        axes[0].loglog(freqs, LFP_psd_soma[0], **line_dict)
        axes[1].loglog(freqs, LFP_psd_apic[0], **line_dict)


class Figure4(PaperFigures):

    def __init__(self, weight, do_simulations=False):
        PaperFigures.__init__(self)
        self.cell_name = 'hay'
        self.figure_name = 'figure_4'
        self.sim_folder = join(self.root_folder, 'paper_simulations', self.cell_name)
        # self.timeres = 2**-5
        self.conductance_dict = {'Ih_linearized': r'Passive+linearized $I_{\rm h}$',
                                 'Ih': r'Passive+$I_{\rm h}$',
                                 'passive': 'Passive',
                                 'NaP': r'Passive+Na$_{\rm P}$',
                                 'NaP_linearized': r'Passive+linearized Na$_{\rm P}$',
                                 }
        self.holding_potentials = [-80, -60]
        self.elec_apic_idx = 88
        self.elec_soma_idx = 18
        self.timeres_NEURON = 2**-4
        self.timeres_python = 2**-4
        self.cut_off = 0
        self.repeats = 2
        self.end_t = 1000 * self.repeats
        self.stimuli = 'white_noise'
        self.conductance_types_1 = ['Ih_linearized', 'passive', 'Ih']#'Ih',
        self.conductance_types_2 = ['NaP_linearized', 'passive', 'NaP', 'active']
        # self.holding_potential = -80
        if self.cell_name == 'hay':
            elec_x, elec_z = np.meshgrid(np.linspace(-200, 200, 7),
                                         np.linspace(-200, 1200, 15))
            self.elec_x = elec_x.flatten()
            self.elec_z = elec_z.flatten()
            self.elec_y = np.zeros(len(self.elec_x))
            self.plot_positions = np.array([self.elec_x, self.elec_y, self.elec_z]).T
            self.soma_idx = 0
            self.apic_idx = 852
        elif self.cell_name == 'n120':
            elec_x, elec_z = np.meshgrid(np.linspace(-150, 150, 7),
                                         np.linspace(-150, 800, 15))
            self.elec_x = elec_x.flatten()
            self.elec_z = elec_z.flatten()
            self.elec_y = np.zeros(len(self.elec_x))
            self.plot_positions = np.array([self.elec_x, self.elec_y, self.elec_z]).T
            self.soma_idx = 0
            self.apic_idx = 685
            self.use_elec_idxs = [33, 2, 78, 61, 22]
            self.ax_dict = {'ylim': [-200, 700], 'xlim': [-250, 250]}
            self.elec_apic_idx = 65
            self.elec_soma_idx = 18
        elif self.cell_name == 'c12861':
            elec_x, elec_z = np.meshgrid(np.linspace(-150, 150, 7), np.linspace(-150, 800, 15))
            self.elec_x = elec_x.flatten()
            self.elec_z = elec_z.flatten()
            self.elec_y = np.zeros(len(self.elec_x))
            self.plot_positions = np.array([self.elec_x, self.elec_y, self.elec_z]).T
            self.soma_idx = 0
            self.apic_idx = 963
            self.use_elec_idxs = [26, 2, 77, 68, 28]#[19, 8, 23, 71, 52]
            self.ax_dict = {'ylim': [-200, 750], 'xlim': [-300, 300]}
            self.elec_apic_idx = 75
            self.elec_soma_idx = 18
        elif self.cell_name == 'shah':
            elec_x, elec_z = np.meshgrid(np.linspace(-150, 150, 7), np.linspace(-150, 800, 15))
            self.elec_x = elec_x.flatten()
            self.elec_z = elec_z.flatten()
            self.elec_y = np.zeros(len(self.elec_x))
            self.plot_positions = np.array([self.elec_x, self.elec_y, self.elec_z]).T
            self.soma_idx = 0
            self.apic_idx = 443
            self.use_elec_idxs = [26, 2, 77, 68, 28]#[19, 8, 23, 71, 52]
            self.ax_dict = {'ylim': [-200, 750], 'xlim': [-300, 300]}
            self.elec_apic_idx = 75
            self.elec_soma_idx = 18
        else:
            raise ValueError("Unknown cell_name")

        if do_simulations:
            self._do_all_simulations(weight)

        self.make_figure(weight)

    def _do_all_simulations(self, weight):
        neural_sim = NeuralSimulations(self)
        # for holding_potential in [-80.]:
        #     for input_idx in [self.apic_idx]:
        #         for conductance_type in self.conductance_types_1:
        #             neural_sim.do_single_neural_simulation(conductance_type, holding_potential, input_idx,
        #                                                     self.elec_x, self.elec_y, self.elec_z, weight)

        for holding_potential in [-60.]:
            for input_idx in [self.soma_idx]:
                for conductance_type in self.conductance_types_2[-1:]:
                    neural_sim.do_single_neural_simulation(conductance_type, holding_potential, input_idx,
                                                            self.elec_x, self.elec_y, self.elec_z, weight)

    def make_figure(self, weight):
        tvec = np.load(join(self.sim_folder, 'tvec_%s.npy' % self.cell_name))
        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name))

        ax_dict = {'xlim': [1e0, 450]}
        plt.close('all')
        fig = plt.figure(figsize=[8, 5])
        fig.subplots_adjust(hspace=0.55, wspace=0.55, bottom=0.2, left=0, right=0.96)
        ax_morph_1 = plt.subplot(141)
        ax_morph_2 = plt.subplot(143)
        ax_morph_1.axis('off')
        ax_morph_2.axis('off')

        fig.text(0.05, 0.9, '-80 mV')
        fig.text(0.57, 0.9, '-60 mV')
        # fig.text(0.005, 0.8, 'Apical', rotation='vertical')
        # fig.text(0.005, 0.4, 'Somatic', rotation='vertical')

        self._draw_set_up_to_axis(ax_morph_1, self.apic_idx, elec_x, elec_z, ic_plot=False)
        self._draw_set_up_to_axis(ax_morph_2, self.soma_idx, elec_x, elec_z, ic_plot=False)

        ec_ax_a_1 = fig.add_subplot(2, 4, 2, ylim=[1e-5, 1e-1], xlabel='Hz', **ax_dict)
        ec_ax_s_1 = fig.add_subplot(2, 4, 6, ylim=[1e-7, 1e-3], **ax_dict)
        ec_ax_a_2 = fig.add_subplot(2, 4, 4, ylim=[1e-6, 1e-2], **ax_dict)
        ec_ax_s_2 = fig.add_subplot(2, 4, 8, ylim=[1e-5, 1e-1], **ax_dict)

        ec_ax_a_1.set_ylabel('PSD [$\mu$V$^2$/Hz]', fontsize=10)

        sig_ax_list = [ec_ax_a_1, ec_ax_s_1, ec_ax_a_2, ec_ax_s_2]
        [ax.grid(True) for ax in sig_ax_list]
        mark_subplots(ax_morph_1, 'A', ypos=1, xpos=0.1)
        mark_subplots(ax_morph_2, 'D', ypos=1, xpos=0.)
        mark_subplots(sig_ax_list, 'BCEF')

        for conductance_type in self.conductance_types_1:
            self._plot_sigs(self.apic_idx, conductance_type, -80, [ec_ax_a_1, ec_ax_s_1], tvec, weight)

        for conductance_type in self.conductance_types_2:
            self._plot_sigs(self.soma_idx, conductance_type, -60, [ec_ax_a_2, ec_ax_s_2], tvec, weight)

        simplify_axes(fig.axes)
        lines = []
        line_names = []
        conductance_types = ['passive', 'NaP', 'Ih', 'NaP_linearized', 'Ih_linearized']

        for conductance_type in conductance_types:
            l, = plt.plot(0, 0, color=self.conductance_clr[conductance_type], lw=2,
                          linestyle=self.conductance_style[conductance_type])
            lines.append(l)
            line_names.append(self.conductance_dict[conductance_type])
        fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=3)
        fig.savefig(join(self.figure_folder, '%s.png' % (self.figure_name)), dpi=150)
        fig.savefig(join(self.figure_folder, '%s.pdf' % (self.figure_name)), dpi=150)

    def _plot_sigs(self, input_idx, conductance_type, holding_potential, axes, tvec, weight):

        LFP = 1000 * np.load(join(self.sim_folder, 'sig_%s_%d_%s_%+d_%1.4f.npy' %
                            (self.cell_name, input_idx, conductance_type, holding_potential, weight)))
        freqs, LFP_psd_soma = aLFP.return_freq_and_psd(tvec, LFP[self.elec_soma_idx, :])
        freqs, LFP_psd_apic = aLFP.return_freq_and_psd(tvec, LFP[self.elec_apic_idx, :])

        line_dict = {'c': self.conductance_clr[conductance_type],
                     'linestyle': self.conductance_style[conductance_type],
                     'lw': 2}

        axes[0].loglog(freqs, LFP_psd_apic[0], **line_dict)
        axes[1].loglog(freqs, LFP_psd_soma[0], **line_dict)


class FigureSystematic(PaperFigures):

    def __init__(self):
        PaperFigures.__init__(self)
        self.cell_name = 'hay'
        self.figure_name = 'figure_5_3'
        self.conductance = 'generic'
        # self.sim_folder = join(self.root_folder, 'paper_simulations', 'generic_study')
        self.sim_folder = join(self.root_folder, 'generic_study', 'hay')
        self.timeres = 2**-4
        self.holding_potential = -80
        self.elec_apic_idx = 0
        self.elec_soma_idx = 12
        self.soma_idx = 0
        self.tau_w = 'auto'
        self.numrows = 6
        self.numcols = 4
        self.apic_idx = 605
        self.axis_w_shift = 0.25
        self.mus = [-0.5, 0, 2]
        self.mu_name_dict = {-0.5: 'Regenerative ($\mu^* =\ -0.5$)',
                             0: 'Passive ($\mu^* =\ 0$)',
                             2: 'Restorative ($\mu^* =\ 2$)'}
        self.input_type = 'wn'
        self.input_idxs = [self.apic_idx, self.soma_idx]
        self.elec_idxs = [self.elec_apic_idx, self.elec_soma_idx]
        self.distributions = ['uniform', 'linear_increase', 'linear_decrease']
        # self.mu_clr = lambda mu: plt.cm.jet(int(256. * (mu - np.min(self.mus))/
        #                                           (np.max(self.mus) - np.min(self.mus))))
        self.mu_clr = {-0.5: 'r',
                       0: 'k',
                       2: 'b'}

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

        # dummy_ax = self.fig.add_subplot(311)
        # dummy_ax.axis('off')
        mark_subplots(self.ax_morph_apic, 'D', xpos=0.05, ypos=1.1)
        mark_subplots(self.ax_morph_soma, 'k', xpos=0.05, ypos=1.1)
        # mark_subplots([self.ax_morph_apic, self.ax_morph_soma], 'BC')
        self._draw_morph_to_axis(self.ax_morph_apic, self.apic_idx)
        self._draw_morph_to_axis(self.ax_morph_soma, self.soma_idx)
        self.ax_morph_apic.scatter(elec_x[[self.elec_apic_idx, self.elec_soma_idx]],
                              elec_z[[self.elec_apic_idx, self.elec_soma_idx]], c='cyan', edgecolor='none', s=50)
        #[plt.plot(elec_x[idx], elec_z[idx], marker='$%d$' % idx) for idx in xrange(len(elec_x))]
        self.ax_morph_soma.scatter(elec_x[[self.elec_apic_idx, self.elec_soma_idx]],
                              elec_z[[self.elec_apic_idx, self.elec_soma_idx]], c='cyan', edgecolor='none', s=50)
        self._make_ax_dict()

        arrow_dict = {'width': 5, 'lw': 1, 'clip_on': False, 'color': 'c', 'zorder': 0}
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
            tau = '%1.2f' % self.tau_w if type(self.tau_w) in [int, float] else self.tau_w
            sim_name = '%s_%s_%d_%1.1f_%+d_%s_%s' % (self.cell_name, self.input_type, 0, 0,
                                                        self.holding_potential, distribution, tau)
            dist_dict = np.load(join(self.sim_folder, 'dist_dict_%s.npy' % sim_name)).item()
            ax_line = self.fig.add_axes([0.25 + self.axis_w_shift*dist_num, 0.83, 0.15, 0.1],
                                      xticks=[], yticks=[], ylim=[0, 0.0004])
            mark_subplots(ax_line, 'ABC'[dist_num])
            ax_morph = self.fig.add_axes([0.17 + self.axis_w_shift*dist_num, 0.83, 0.1, 0.1])
            ax_morph.axis('off')
            self._draw_simplified_morph_to_axis(ax_morph, grading=distribution)
            if dist_num == 1:
                ax_line.set_xlabel('Distance from soma')
            simplify_axes(ax_line)
            argsort = np.argsort(dist_dict['dist'])
            dist = dist_dict['dist'][argsort]
            g = dist_dict['g_w_bar_QA'][argsort]
            x = [dist[0], dist[-1]]
            y = [g[0], g[-1]]
            ax_line.plot(x, y, lw=2, c='gray')

    def _finitialize_figure(self):

        lines = []
        line_names = []
        textsize = 20
        for mu in self.mus:
            l, = plt.plot(0, 0, color=self.mu_clr[mu], lw=2)
            lines.append(l)
            line_names.append(self.mu_name_dict[mu])
        self.fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=5)
        self.fig.text(0.5, 0.97, 'Channel distributions', ha='center', size=textsize)
        self.fig.text(0.5, 0.754, 'Apical input', ha='center', size=textsize)
        self.fig.text(0.5, 0.4, 'Somatic input', ha='center', size=textsize)

        self.fig.text(0.28, 0.93, 'Uniform', rotation='horizontal')
        self.fig.text(0.52, 0.93, 'Linear increase', rotation='horizontal')
        self.fig.text(0.75, 0.93, 'Linear decrease', rotation='horizontal')

        mark_subplots(self.ax_dict.values(), 'MPEGNIQFOJLH')
        simplify_axes(self.ax_dict.values())

        for [input_idx, elec, distribution], ax in self.ax_dict.items():

            if input_idx == 605 and elec == 12 and distribution == 'uniform':
                ax.set_yticks([1e-6, 1e-4, 1e-2])
                ax.set_yticklabels(['10$^{-6}$', '', '10$^{-2}$'])
                ax.set_xticks([1e0, 1e1, 1e2])
                ax.set_xticklabels(['10$^{0}$', '', '10$^{-2}$'])
            else:
                ax.set_yticks([1e-6, 1e-4, 1e-2])
                ax.set_yticklabels(['', '', ''])
                ax.set_xticks([1e0, 1e1, 1e2])
                ax.set_xticklabels(['', '', ''])


        self.fig.savefig(join(self.figure_folder, '%s_%s.png' % (self.figure_name, self.cell_name)), dpi=150)
        self.fig.savefig(join(self.figure_folder, '%s_%s.pdf' % (self.figure_name, self.cell_name)), dpi=150)

    def _make_ax_dict(self):
        self.ax_dict = {}
        ax_w = 0.15
        ax_h = 0.1
        ax_props = {'xlim': [1, 450], 'ylim': [1e-6, 2e-2], 'xscale': 'log',
                    'yscale': 'log'}

        for input_num, input_idx in enumerate(self.input_idxs):
            for elec_num, elec in enumerate(self.elec_idxs):
                for dist_num, distribution in enumerate(self.distributions):
                    ax_x_start = 0.25 + self.axis_w_shift * dist_num
                    ax_y_start = 0.63 - input_num * 0.37 - elec_num * ax_h*1.5
                    ax = self.fig.add_axes([ax_x_start, ax_y_start, ax_w, ax_h], **ax_props)

                    # ax.set_title('%d %d %s' % (input_idx, elec, distribution))
                    ax.grid(True)
                    self.ax_dict[input_idx, elec, distribution] = ax

                    if input_num == 0 and elec_num == 1 and dist_num == 0:
                        ax.set_xlabel('Hz')
                        ax.set_ylabel('PSD')

    def make_figure(self):
        for [input_idx, elec, distribution], ax in self.ax_dict.items():
            for mu in self.mus:
                self._plot_sigs(input_idx, ax, distribution, mu, elec)

    def _plot_sigs(self, input_idx, ax, distribution, mu, elec):
        tau = '%1.2f' % self.tau_w if type(self.tau_w) in [int, float] else self.tau_w
        sim_name = '%s_%s_%d_%1.1f_%+d_%s_%s' % (self.cell_name, self.input_type, input_idx, mu,
                                                    self.holding_potential, distribution, tau)
        LFP = 1000 * np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))
        # tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))
        freqs, LFP_psd = aLFP.return_freq_and_psd(self.timeres / 1000., LFP[elec, :])
        ax.loglog(freqs, LFP_psd[0], c=self.mu_clr[mu], lw=2)


class FigureNeurite(PaperFigures):

    def __init__(self, do_simulations=False):
        PaperFigures.__init__(self)
        self.cell_name = 'infinite_neurite'
        self.figure_name = 'infinite_neurite'
        self.sim_folder = join(self.root_folder, 'paper_simulations', 'infinite_neurite')
        self.holding_potential = -80
        self.timeres_NEURON = 2**-4
        self.timeres_python = 2**-4
        self.repeats = 2
        self.cut_off = 0
        self.end_t = 1000 * self.repeats
        self.weight = 0.0001
        self.stimuli = 'white_noise'
        self.conductance_types_1 = ['2.0_2.0', '0.0_0.0', '-0.5_-0.5']
        self.conductance_types_2 = ['0.0_2.0', '0.0_0.0', '0.0_-0.5']
        self.conductance_types_3 = ['2.0_0.0', '0.0_0.0', '-0.5_0.0']

        self.mu_name_dict = {'-0.5': 'Regenerative ($\mu_{factor} = -0.5$)',
                             '0.0': 'Passive ($\mu_{factor} = 0$)',
                             '2.0': 'Restorative ($\mu_{factor} = 2$)'}
        self.mu_clr = {'-0.5': 'r',
                       '0.0': 'k',
                       '2.0': 'b'}
        # elec_x, elec_z = np.meshgrid(np.linspace(0, 1000, 4), np.ones(1) * 20)
        elec_x, elec_z = np.meshgrid(np.array([0., 70.796, 106.194, 973.451]), np.ones(1) * 20)
        self.elec_x = elec_x.flatten()
        self.elec_z = elec_z.flatten()
        self.elec_y = np.zeros(len(self.elec_x))
        self.input_idx = 0

        if do_simulations:
             self._do_all_simulations()

        self.xstart = np.load(join(self.sim_folder, 'xstart_%s.npy' % self.cell_name))
        self.zstart = np.load(join(self.sim_folder, 'zstart_%s.npy' % self.cell_name))
        self.xend = np.load(join(self.sim_folder, 'xend_%s.npy' % self.cell_name))
        self.zend = np.load(join(self.sim_folder, 'zend_%s.npy' % self.cell_name))
        self.xmid = np.load(join(self.sim_folder, 'xmid_%s.npy' % self.cell_name))
        self.zmid = np.load(join(self.sim_folder, 'zmid_%s.npy' % self.cell_name))
        self.xlim = [self.xmid[self.input_idx] - 50, self.xmid[self.input_idx] + 1050]
        self.plot_comps = self.input_idx + np.array([0, 4, 6, 55])

        self.num_cols = len(self.plot_comps)

    def _do_all_simulations(self):
        neural_sim = NeuralSimulations(self)
        for conductance_type in self.conductance_types_1 + self.conductance_types_2 + self.conductance_types_3:
            neural_sim.do_single_neural_simulation(conductance_type, self.holding_potential, self.input_idx,
                                                            self.elec_x, self.elec_y, self.elec_z, self.weight)

    def _draw_neurite(self):

        ax_dict = {'frameon': False,
                   'xlim': self.xlim,
                   'ylim': [-25, 25],
                   'yticks': [],
                   'xticks': []}
        self.ax_neur_1 = self.fig.add_subplot(12, 1, 1, **ax_dict)
        self.ax_neur_2 = self.fig.add_subplot(12, 1, 5, **ax_dict)
        self.ax_neur_3 = self.fig.add_subplot(12, 1, 9, **ax_dict)

        width = 10
        dx = self.xend[-1] - self.xstart[0]
        cond_length = 100
        neurite_1 = mpatches.Rectangle([self.xstart[0], self.zstart[0] - width / 2], dx, width, color='0.7', ec='k')
        neurite_2a = mpatches.Rectangle([self.xstart[0], self.zstart[0] - width / 2], cond_length, width, color='w', ec='k')
        neurite_2b = mpatches.Rectangle([self.xstart[0] + cond_length, self.zstart[0] - width / 2], dx - cond_length, width, color='0.7', ec='k')
        neurite_3a = mpatches.Rectangle([self.xstart[0], self.zstart[0] - width / 2], cond_length, width, color='0.7', ec='k')
        neurite_3b = mpatches.Rectangle([self.xstart[0] + cond_length, self.zstart[0] - width / 2], dx - cond_length, width, color='w', ec='k')

        self.ax_neur_1.add_patch(neurite_1)
        self.ax_neur_2.add_patch(neurite_2a)
        self.ax_neur_2.add_patch(neurite_2b)
        self.ax_neur_3.add_patch(neurite_3a)
        self.ax_neur_3.add_patch(neurite_3b)

        self.ax_neur_1.plot([-5, 95], [17, 17], lw=3, c='k')
        self.ax_neur_1.text(25, 25, '100 $\mu m$', ha='center')

        self.ax_neur_1.plot(self.xmid[self.input_idx], self.zmid[self.input_idx], 'y*', ms=15)
        self.ax_neur_2.plot(self.xmid[self.input_idx], self.zmid[self.input_idx], 'y*', ms=15)
        self.ax_neur_3.plot(self.xmid[self.input_idx], self.zmid[self.input_idx], 'y*', ms=15)

        [self.ax_neur_1.plot(self.xmid[idx], self.zmid[idx], marker='o', color='orange', ms=5) for idx in self.plot_comps]
        [self.ax_neur_2.plot(self.xmid[idx], self.zmid[idx], marker='o', color='orange', ms=5) for idx in self.plot_comps]
        [self.ax_neur_3.plot(self.xmid[idx], self.zmid[idx], marker='o', color='orange', ms=5) for idx in self.plot_comps]

        [self.ax_neur_1.plot(self.elec_x[idx], self.elec_z[idx], marker='o', color='c', ms=5) for idx in xrange(len(self.elec_x))]
        [self.ax_neur_2.plot(self.elec_x[idx], self.elec_z[idx], marker='o', color='c', ms=5) for idx in xrange(len(self.elec_x))]
        [self.ax_neur_3.plot(self.elec_x[idx], self.elec_z[idx], marker='o', color='c', ms=5) for idx in xrange(len(self.elec_x))]

        mark_subplots([self.ax_neur_1, self.ax_neur_2, self.ax_neur_3])


    def _initialize_figure(self):
        plt.close('all')
        self.fig = plt.figure(figsize=[5, 10])
        self.fig.subplots_adjust(hspace=0.8, wspace=0.55, bottom=0.15, top=0.95, left=0.17, right=0.96)
        self._draw_neurite()

        ax_dict = {'xlim': [1, 450], 'xscale': 'log', 'yscale': 'log'}

        # i_ylim_list = [[1e-5, 1e-4], [1e-7, 1e-6], [1e-7, 1e-6], [1e-9, 1e-8]]
        i_ylim_list = [[1e-5, 1e-3], [1e-8, 1e-6], [1e-8, 1e-6], [1e-9, 1e-7]]
        lfp_ylim_list = [[1e-3, 1e-1], [1e-5, 1e-3], [1e-5, 1e-3], [1e-6, 1e-4]]

        self.vmem_ax_list_1 = [self.fig.add_subplot(12, self.num_cols,  1 + idx + self.num_cols * 1, **ax_dict)
                               for idx in xrange(self.num_cols)]
        self.imem_ax_list_1 = [self.fig.add_subplot(12, self.num_cols,  1 + idx + self.num_cols * 2, ylim=i_ylim_list[idx], **ax_dict)
                               for idx in xrange(self.num_cols)]
        self.LFP_ax_list_1 = [self.fig.add_subplot(12, self.num_cols,  1 + idx + self.num_cols * 3, ylim=lfp_ylim_list[idx], **ax_dict)
                               for idx in xrange(self.num_cols)]

        self.vmem_ax_list_2 = [self.fig.add_subplot(12, self.num_cols,  1 + idx + self.num_cols * 5, **ax_dict)
                               for idx in xrange(self.num_cols)]
        self.imem_ax_list_2 = [self.fig.add_subplot(12, self.num_cols,  1 + idx + self.num_cols * 6, ylim=i_ylim_list[idx], **ax_dict)
                               for idx in xrange(self.num_cols)]
        self.LFP_ax_list_2 = [self.fig.add_subplot(12, self.num_cols,  1 + idx + self.num_cols * 7, ylim=lfp_ylim_list[idx], **ax_dict)
                               for idx in xrange(self.num_cols)]

        self.vmem_ax_list_3 = [self.fig.add_subplot(12, self.num_cols,  1 + idx + self.num_cols * 9, **ax_dict)
                               for idx in xrange(self.num_cols)]
        self.imem_ax_list_3 = [self.fig.add_subplot(12, self.num_cols,  1 + idx + self.num_cols * 10, ylim=i_ylim_list[idx], **ax_dict)
                               for idx in xrange(self.num_cols)]
        self.LFP_ax_list_3 = [self.fig.add_subplot(12, self.num_cols,  1 + idx + self.num_cols * 11, ylim=lfp_ylim_list[idx], **ax_dict)
                               for idx in xrange(self.num_cols)]

        [ax.set_ylabel('nA') for ax in [self.imem_ax_list_1[0], self.imem_ax_list_2[0], self.imem_ax_list_3[0]]]
        [ax.set_xlabel('Hz') for ax in [self.imem_ax_list_1[0]]]
        [ax.set_xticklabels(['$10^0$', '', '$10^2$']) for ax in [self.imem_ax_list_1[0]]]
        [ax.set_ylabel('mV') for ax in [self.vmem_ax_list_1[0], self.vmem_ax_list_2[0], self.vmem_ax_list_3[0]]]

        for numb, idx in enumerate(self.plot_comps):
            pos = [self.xmid[idx], self.zmid[idx]]
            self.LFP_arrow_to_axis(pos, self.ax_neur_1, self.vmem_ax_list_1[numb])
            self.LFP_arrow_to_axis(pos, self.ax_neur_2, self.vmem_ax_list_2[numb])
            self.LFP_arrow_to_axis(pos, self.ax_neur_3, self.vmem_ax_list_3[numb])

    def make_figure(self):

        self._initialize_figure()
        self._plot_sigs()
        simplify_axes(self.fig.axes)

        i_ax_list = (self.imem_ax_list_1 + self.imem_ax_list_2 + self.imem_ax_list_3)
        for ax in i_ax_list:
            ax.set_yticks(ax.get_ylim())
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels(['', '', ''])

        v_ax_list = (self.vmem_ax_list_1 + self.vmem_ax_list_2 + self.vmem_ax_list_3)
        for ax in v_ax_list:
            # max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()[1:]) for l in ax.get_lines()])))
            ax.set_ylim([1e-2, 1e1])
            ax.set_yticks([1e-2, 1e1])
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels(['', '', ''])

        lines = []
        line_names = []
        for conductance_type in self.mu_clr.keys():
            l, = plt.plot(0, 0, color=self.mu_clr[conductance_type], lw=2)
            lines.append(l)
            line_names.append(self.mu_name_dict[conductance_type])
        self.fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=1)
        self.fig.savefig(join(self.figure_folder, '%s.png' % self.figure_name), dpi=150)

    def _plot_sigs(self):
        tvec = np.load(join(self.sim_folder, 'tvec_%s.npy' % self.cell_name))
        for conductance_type in self.conductance_types_1:
            self._single_plot(tvec, conductance_type, self.vmem_ax_list_1, self.imem_ax_list_1, self.LFP_ax_list_1)
        for conductance_type in self.conductance_types_2:
            self._single_plot(tvec, conductance_type, self.vmem_ax_list_2, self.imem_ax_list_2, self.LFP_ax_list_2)
        for conductance_type in self.conductance_types_3:
            self._single_plot(tvec, conductance_type, self.vmem_ax_list_3, self.imem_ax_list_3, self.LFP_ax_list_3)

    def _single_plot(self, tvec, conductance_type, v_ax_list, i_ax_list, lfp_ax_list):
        vmem = np.load(join(self.sim_folder, 'vmem_%s_%d_%s_%+d_%1.4f.npy' %
                            (self.cell_name, self.input_idx, conductance_type, self.holding_potential, self.weight)))
        imem = np.load(join(self.sim_folder, 'imem_%s_%d_%s_%+d_%1.4f.npy' %
                            (self.cell_name, self.input_idx, conductance_type, self.holding_potential, self.weight)))
        lfp = 1000 * np.load(join(self.sim_folder, 'sig_%s_%d_%s_%+d_%1.4f.npy' %
                          (self.cell_name, self.input_idx, conductance_type, self.holding_potential, self.weight)))
        freqs, vmem_psd = aLFP.return_freq_and_psd(tvec, vmem[self.plot_comps, :-1])
        freqs, imem_psd = aLFP.return_freq_and_psd(tvec, imem[self.plot_comps, :-1])
        freqs, lfp_psd = aLFP.return_freq_and_psd(tvec, lfp[:, :-1])
        if '2.0' in conductance_type:
            conductance_name = '2.0'
        elif '-0.5' in conductance_type:
            conductance_name = '-0.5'
        else:
            conductance_name = '0.0'
        clr = self.mu_clr[conductance_name]
        [i_ax_list[idx].loglog(freqs, imem_psd[idx], c=clr) for idx in xrange(len(self.plot_comps))]
        [v_ax_list[idx].loglog(freqs, vmem_psd[idx], c=clr) for idx in xrange(len(self.plot_comps))]
        [lfp_ax_list[idx].loglog(freqs, lfp_psd[idx], c=clr) for idx in xrange(len(self.elec_x))]


class FigureNeurite2(PaperFigures):

    def __init__(self, do_simulations=False):
        PaperFigures.__init__(self)
        self.cell_name = 'infinite_neurite'
        self.figure_name = 'infinite_neurite'
        self.sim_folder = join(self.root_folder, 'paper_simulations', 'infinite_neurite')
        self.holding_potential = -80
        self.timeres_NEURON = 2**-4
        self.timeres_python = 2**-4
        self.repeats = 2
        self.cut_off = 0
        self.end_t = 1000 * self.repeats
        self.weight = 0.0005
        self.stimuli = 'white_noise'
        self.conductance_types_1 = ['2.0_2.0', '0.0_0.0', '-0.5_-0.5']

        self.mu_name_dict = {'-0.5': 'Regenerative ($\mu^* =\ -0.5$)',
                             '0.0': 'Passive ($\mu^* =\ 0$)',
                             '2.0': 'Restorative ($\mu^* =\ 2$)',
                             '-1.0': 'Regenerative ($\mu^* =\ -1$)',
                             '4.0': 'Restorative ($\mu^* =\ 4$)'}
        self.mu_clr = {'-0.5': 'r',
                       '0.0': 'k',
                       '2.0': 'b',}
        # elec_x, elec_z = np.meshgrid(np.linspace(0, 1000, 4), np.ones(1) * 20)
        elec_x, elec_z = np.meshgrid(np.array([0., 176.991, 530.973, 973.451]), np.array([-20, -300]))


        self.elec_x = elec_x.flatten()
        self.elec_z = elec_z.flatten()
        self.elec_y = np.zeros(len(self.elec_x))
        self.input_idx = 0

        if do_simulations:
             self._do_all_simulations()

        self.xstart = np.load(join(self.sim_folder, 'xstart_%s.npy' % self.cell_name))
        self.zstart = np.load(join(self.sim_folder, 'zstart_%s.npy' % self.cell_name))
        self.xend = np.load(join(self.sim_folder, 'xend_%s.npy' % self.cell_name))
        self.zend = np.load(join(self.sim_folder, 'zend_%s.npy' % self.cell_name))
        self.xmid = np.load(join(self.sim_folder, 'xmid_%s.npy' % self.cell_name))
        self.zmid = np.load(join(self.sim_folder, 'zmid_%s.npy' % self.cell_name))
        self.xlim = [self.xmid[self.input_idx] - 50, self.xmid[self.input_idx] + 1050]
        self.plot_comps = self.input_idx + np.array([0, 10, 30, 55])
        # print self.xmid[self.plot_comps]

        self.num_cols = len(self.plot_comps)
        self.num_LFP_rows = 2

    def _do_all_simulations(self):
        neural_sim = NeuralSimulations(self)
        for conductance_type in self.conductance_types_1:
            neural_sim.do_single_neural_simulation(conductance_type, self.holding_potential, self.input_idx,
                                                            self.elec_x, self.elec_y, self.elec_z, self.weight)

    def _draw_neurite(self):

        ax_dict = {'frameon': False,
                   'xlim': self.xlim,
                   'ylim': [-52, 52],
                   'yticks': [],
                   'xticks': [],}
        self.ax_neur_1 = self.fig.add_subplot(5, 1, 3, **ax_dict)

        width = 10
        dx = self.xend[-1] - self.xstart[0]
        cond_length = 100
        neurite_1 = mpatches.Rectangle([self.xstart[0], self.zstart[0] - width / 2], dx, width, color='0.7', ec='k')

        self.ax_neur_1.add_patch(neurite_1)

        self.ax_neur_1.plot([350, 450], [-20, -20], lw=3, c='k', clip_on=False)
        self.ax_neur_1.text(400, -55, '100 $\mu m$', ha='center')

        self.ax_neur_1.arrow(-100, 0, 40, 0, lw=1, head_width=10, color='k', clip_on=False)
        self.ax_neur_1.arrow(-100, 0, 0, 30, lw=1, head_width=14, head_length=12, color='k', clip_on=False)
        self.ax_neur_1.text(-120, 20, 'z', size=10, ha='center', va='center', clip_on=False)
        self.ax_neur_1.text(-80, -12, 'x', size=10, ha='center', va='center', clip_on=False)


        [self.ax_neur_1.plot(self.xmid[idx], self.zmid[idx], marker='o', color='orange', ms=7, mec='orange') for idx in self.plot_comps]
        [self.ax_neur_1.plot(self.elec_x[idx], self.elec_z[idx], marker='o', color='cyan', ms=7, mec='cyan', clip_on=False) for idx in xrange(len(self.elec_x))]
        self.ax_neur_1.plot(self.xmid[self.input_idx], self.zmid[self.input_idx], 'y*', ms=15)
        #mark_subplots([self.ax_neur_1])

    def _initialize_figure(self):
        plt.close('all')
        self.fig = plt.figure(figsize=[5, 7])
        self.fig.subplots_adjust(hspace=0.9, wspace=0.55, bottom=0.23, top=0.95, left=0.19, right=0.98)
        self._draw_neurite()

        ax_dict = {'xlim': [1, 450], 'xscale': 'log', 'yscale': 'log'}

        # i_ylim_list = [[1e-5, 1e-4], [1e-7, 1e-6], [1e-7, 1e-6], [1e-9, 1e-8]]
        i_ylim_list = [[1e-4, 1e-2], [1e-7, 1e-5], [1e-8, 1e-6], [1e-8, 1e-6]]
        lfp_ylim_list = [[1e-2, 1e0], [1e-4, 1e-2], [1e-4, 1e-2], [1e-4, 1e-2], [1e-5, 1e-3], [1e-5, 1e-3], [1e-5, 1e-3], [1e-5, 1e-3]]

        self.vmem_ax_list_1 = [self.fig.add_subplot(5, self.num_cols,  1 + idx + self.num_cols * 0, **ax_dict)
                               for idx in xrange(self.num_cols)]
        self.imem_ax_list_1 = [self.fig.add_subplot(5, self.num_cols,  1 + idx + self.num_cols * 1, ylim=i_ylim_list[idx], **ax_dict)
                               for idx in xrange(self.num_cols)]
        self.LFP_ax_list_1 = [self.fig.add_subplot(5, self.num_cols,  1 + idx + self.num_cols * 3, ylim=lfp_ylim_list[idx], **ax_dict)
                               for idx in xrange(len(self.elec_x))]


        mark_subplots(self.ax_neur_1, 'I')
        mark_subplots(self.vmem_ax_list_1, 'ABCD', ypos=1.2, xpos=0.1)
        mark_subplots(self.imem_ax_list_1, 'EFGH', ypos=1.2, xpos=0.1)
        mark_subplots(self.LFP_ax_list_1, 'JKLMNOPQ', ypos=1.2, xpos=0.1)


        [ax.set_ylabel('$I_m$\n[nA$^2$/Hz]') for ax in [self.imem_ax_list_1[0]]]
        [ax.set_ylabel('$V_m$\n[mV$^2$/Hz]') for ax in [self.vmem_ax_list_1[0]]]
        self.LFP_ax_list_1[0].set_ylabel('LFP\n[$\mu$V$^2$/Hz]')
        self.LFP_ax_list_1[4].set_ylabel('LFP\n[$\mu$V$^2$/Hz]')
        # [ax.set_ylabel('LFP [$\mu$V$^2$/Hz]') for ax in [self.LFP_ax_list_1[0], self.LFP_ax_list_1[4]]]


        [ax.set_xlabel('Hz') for ax in [self.imem_ax_list_1[0]]]
        [ax.set_xticklabels(['$10^0$', '', '$10^2$']) for ax in [self.imem_ax_list_1[0]]]

        for numb, idx in enumerate(self.plot_comps):
            pos = [self.xmid[idx], self.zmid[idx]]
            self.LFP_arrow_to_axis(pos, self.ax_neur_1, self.imem_ax_list_1[numb])

        for idx in xrange(len(self.LFP_ax_list_1)):
            pos = [self.elec_x[idx], self.elec_z[idx]]
            self.LFP_arrow_to_axis(pos, self.ax_neur_1, self.LFP_ax_list_1[idx], c='cyan')

        # for idx in xrange(len(self.LFP_ax_list_2)):
        #     pos = [self.elec_x[idx], self.elec_z[idx]]
        #     self.LFP_arrow_to_axis(pos, self.ax_neur_1, self.LFP_ax_list_1[idx], c='cyan')

    def make_figure(self):

        self._initialize_figure()
        self._plot_sigs()
        simplify_axes(self.fig.axes)

        i_ax_list = (self.imem_ax_list_1)
        for ax in i_ax_list:
            ax.set_yticks(ax.get_ylim())
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels(['$10^0$', '', '$10^2$'])

        v_ax_list = (self.vmem_ax_list_1)
        for ax in v_ax_list:
            # max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()[1:]) for l in ax.get_lines()])))
            ax.set_ylim([1e0, 1e3])
            ax.set_yticks([1e0, 1e3])
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels(['$10^0$', '', '$10^2$'])

        for ax in self.LFP_ax_list_1:
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels(['$10^0$', '', '$10^2$'])

        lines = []
        line_names = []
        for conductance_type in self.mu_clr.keys():
            l, = plt.plot(0, 0, color=self.mu_clr[conductance_type], lw=2)
            lines.append(l)
            line_names.append(self.mu_name_dict[conductance_type])
        self.fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=1)
        self.fig.savefig(join(self.figure_folder, '%s.png' % self.figure_name), dpi=150)
        self.fig.savefig(join(self.figure_folder, '%s.pdf' % self.figure_name), dpi=150)

    def _plot_sigs(self):
        tvec = np.load(join(self.sim_folder, 'tvec_%s.npy' % self.cell_name))
        for conductance_type in self.conductance_types_1:
            self._single_plot(tvec, conductance_type, self.vmem_ax_list_1, self.imem_ax_list_1, self.LFP_ax_list_1)


    def _single_plot(self, tvec, conductance_type, v_ax_list, i_ax_list, lfp_ax_list):
        vmem = np.load(join(self.sim_folder, 'vmem_%s_%d_%s_%+d_%1.4f.npy' %
                            (self.cell_name, self.input_idx, conductance_type, self.holding_potential, self.weight)))
        imem = np.load(join(self.sim_folder, 'imem_%s_%d_%s_%+d_%1.4f.npy' %
                            (self.cell_name, self.input_idx, conductance_type, self.holding_potential, self.weight)))
        lfp = 1000 * np.load(join(self.sim_folder, 'sig_%s_%d_%s_%+d_%1.4f.npy' %
                          (self.cell_name, self.input_idx, conductance_type, self.holding_potential, self.weight)))

        freqs, vmem_psd = aLFP.return_freq_and_psd(tvec, vmem[self.plot_comps, :-1])
        freqs, imem_psd = aLFP.return_freq_and_psd(tvec, imem[self.plot_comps, :-1])
        freqs, lfp_psd = aLFP.return_freq_and_psd(tvec, lfp[:, :-1])
        if '2.0' in conductance_type:
            conductance_name = '2.0'
        elif '-0.5' in conductance_type:
            conductance_name = '-0.5'
        else:
            conductance_name = '0.0'
        clr = self.mu_clr[conductance_name]
        [i_ax_list[idx].loglog(freqs, imem_psd[idx], c=clr) for idx in xrange(len(self.plot_comps))]
        [v_ax_list[idx].loglog(freqs, vmem_psd[idx], c=clr) for idx in xrange(len(self.plot_comps))]
        [lfp_ax_list[idx].loglog(freqs, lfp_psd[idx], c=clr) for idx in xrange(len(self.elec_x))]
        # [lfp_ax_list_2[idx].loglog(freqs, lfp_psd[idx], c=clr) for idx in xrange(len(self.elec_x)/2)]
        # plt.close('all')
        # plt.plot(tvec, lfp[0] / max(np.abs(lfp[0])), lw=2)
        # plt.plot(tvec, lfp[2] / max(np.abs(lfp[2])), lw=2)
        # plt.plot([tvec[0], tvec[-1]], [0, 0], '--', lw=1, c='k')
        # plt.xlim([230, 270])
        # plt.show()


class FigureTimeConstant(PaperFigures):

    def __init__(self):
        PaperFigures.__init__(self)
        self.cell_name = 'hay'
        self.figure_name = 'figure_time_constant_2'
        self.conductance = 'generic'
        self.sim_folder = join(self.root_folder, 'generic_study', 'hay')
        self.timeres = 2**-4
        self.holding_potential = -80
        self.soma_idx = 0
        self.apic_idx = 605
        self.input_idx = self.apic_idx
        self.elec_idx = 13
        self.distribution = 'linear_increase'
        self.mus = [-0.5, 0, 2]
        # self.mu_name_dict = {-0.5: 'Regenerative ($\mu_{factor} = -0.5$)',
        #                      0: 'Passive ($\mu_{factor} = 0$)',
        #                      2: 'Restorative ($\mu_{factor} = 2$)'}
        self.input_type = 'wn'
        self.tau_ws = ['auto0.1', 'auto', 'auto10']

        # self.mu_clr = {-0.5: 'r',
        #                0: 'k',
        #                2: 'b'}

        self.mu_name_dict = {-0.5: 'Regenerative ($\mu^* =\ -0.5$)',
                             0: 'Passive ($\mu^* =\ 0$)',
                             2: 'Restorative ($\mu^* =\ 2$)'}
        # self.mu_clr = {'-0.5': 'r',
        #                '0.0': 'k',
        #                '2.0': 'b'}
        self.mu_clr = {-0.5: 'r',
                       0: 'k',
                       2: 'b'}

        self._initialize_figure()
        self.make_figure()
        self._finitialize_figure()

    def _initialize_figure(self):
        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name))

        plt.close('all')
        self.fig = plt.figure(figsize=[10, 6])
        self.fig.subplots_adjust(hspace=1.2, wspace=0.45, bottom=0.15, left=0)
        self.ax_morph = self.fig.add_axes([0.01, 0.05, 0.25, 0.85], aspect='equal')
        self.ax_morph.axis('off')
        mark_subplots(self.ax_morph, xpos=0, ypos=.85)

        self._draw_morph_to_axis(self.ax_morph, self.apic_idx, ic_comp=0, distribution=self.distribution)
        self.ax_morph.scatter(elec_x[self.elec_idx], elec_z[self.elec_idx], c='cyan', edgecolor='none', s=50)


    def _finitialize_figure(self):
        lines = []
        line_names = []
        for mu in self.mus:
            l, = plt.plot(0, 0, color=self.mu_clr[mu], lw=2)
            lines.append(l)
            line_names.append(self.mu_name_dict[mu])
        self.fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=5)
        simplify_axes(self.fig.axes)
        textsize = 25
        mark_subplots(self.fig.axes, ypos=1.2)
        self.fig.text(0.25, 0.82, r'$\tau_m / 10$:', ha='center', size=textsize)
        self.fig.text(0.25, 0.52, r'$\tau_m$:', ha='center', size=textsize)
        self.fig.text(0.25, 0.2, r'$\tau_m \times 10$:', ha='center', size=textsize)
        self.fig.axes[1].set_xlabel('Hz')
        self.fig.axes[1].set_ylabel('PSD')

        [ax.set_yticks([ax.get_yticks()[1], ax.get_yticks()[-2]]) for ax in self.fig.axes]

        self.fig.savefig(join(self.figure_folder, '%s_%s.pdf' % (self.figure_name, self.cell_name)), dpi=150)
        self.fig.savefig(join(self.figure_folder, '%s_%s.png' % (self.figure_name, self.cell_name)), dpi=150)

    def make_figure(self):
        for numb, tau_w in enumerate(self.tau_ws):
            for mu in self.mus:
                self._plot_sigs(numb, tau_w, mu)

    def _plot_sigs(self, numb, tau_w, mu):
        tau = '%1.2f' % tau_w if type(tau_w) in [int, float] else tau_w
        sim_name = '%s_%s_%d_%1.1f_%+d_%s_%s' % (self.cell_name, self.input_type, self.input_idx, mu,
                                                 self.holding_potential, self.distribution, tau)
        LFP = 1000 * np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))[self.elec_idx, :]
        imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))[self.soma_idx, :]
        vmem = np.load(join(self.sim_folder, 'vmem_%s.npy' % sim_name))[self.soma_idx, :]

        freqs, LFP_psd = aLFP.return_freq_and_psd(self.timeres / 1000., LFP)
        freqs, imem_psd = aLFP.return_freq_and_psd(self.timeres / 1000., imem)
        freqs, vmem_psd = aLFP.return_freq_and_psd(self.timeres / 1000., vmem)

        ax_dict = {'xlim': [1, 450]}
        ax2 = self.fig.add_subplot(3, 5, numb * 5 + 3, ylim=[1e-5, 1e-2], **ax_dict)
        ax1 = self.fig.add_subplot(3, 5, numb * 5 + 4, ylim=[1e-10, 1.1e-7], **ax_dict)
        ax0 = self.fig.add_subplot(3, 5, numb * 5 + 5, ylim=[1e-6, 1e-4], **ax_dict)

        ax0.set_title('$\Phi$')
        ax1.set_title('$I_m$')
        ax2.set_title('$V_m$')

        ax0.loglog(freqs, LFP_psd[0], c=self.mu_clr[mu], lw=2)
        ax1.loglog(freqs, imem_psd[0], c=self.mu_clr[mu], lw=2)
        ax2.loglog(freqs, vmem_psd[0], c=self.mu_clr[mu], lw=2)


class FigureDistributedSynaptic(PaperFigures):

    def __init__(self):
        PaperFigures.__init__(self)
        self.cell_name = 'hay'
        self.figure_name = 'figure_distributed_synaptic'
        self.conductance = 'generic'
        self.sim_folder = join(self.root_folder, 'generic_study', 'hay')
        self.timeres = 2**-4
        self.holding_potential = -80
        self.soma_idx = 0
        self.apic_idx = 605
        self.input_idx = self.apic_idx
        self.elec_idx = 13
        self.distribution = 'linear_increase'
        self.input_secs = ['distal_tuft', 'tuft', 'homogeneous']
        self.mus = [-0.5, 0, 2]
        self.mu_name_dict = {-0.5: 'Regenerative ($\mu^* =\ -0.5$)',
                             0: 'Passive ($\mu^*\ = 0$)',
                             2: 'Restorative ($\mu^* =\ 2$)',
                             -1: 'Regenerative ($\mu^* =\ -1$)',
                             4: 'Restorative ($\mu^* =\ 4$)',
                             }

        self.input_type = 'distributed_synaptic'
        self.tau_w = 'auto'
        self.weight = 0.0001


        self.mu_clr = {-0.5: 'r',
                       0: 'k',
                       2: 'b',
                       -1: 'r',
                       4: 'b',
        }
        self._initialize_figure()
        self.make_figure()
        self._finitialize_figure()

    def _initialize_figure(self):
        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name))

        plt.close('all')
        self.fig = plt.figure(figsize=[10, 4])
        self.fig.subplots_adjust(hspace=0.3, wspace=0.9, bottom=0.3, left=0.17, right=0.98)
        self.ax_morph_1 = self.fig.add_axes([0., 0.2, 0.15, 0.8], aspect='equal')
        self.ax_morph_2 = self.fig.add_axes([0.32, 0.2, 0.15, 0.8], aspect='equal')
        self.ax_morph_3 = self.fig.add_axes([0.64, 0.2, 0.15, 0.8], aspect='equal')
        self.ax_morph_1.axis('off')
        self.ax_morph_2.axis('off')
        self.ax_morph_3.axis('off')
        mark_subplots(self.fig.axes, 'ABC', xpos=0.3, ypos=.95)

        sim_name = '%s_%s_%s_%1.1f_%+d_%s_%s_%1.4f' % (self.cell_name, self.input_type, 'distal_tuft', 2.0,
                                                           self.holding_potential, self.distribution, 'auto', self.weight)
        input_1 = np.load(join(self.sim_folder, 'synidx_%s.npy' % sim_name))
        sim_name = '%s_%s_%s_%1.1f_%+d_%s_%s_%1.4f' % (self.cell_name, self.input_type, 'tuft', 2.0,
                                                           self.holding_potential, self.distribution, 'auto', self.weight)
        input_2 = np.load(join(self.sim_folder, 'synidx_%s.npy' % sim_name))


        sim_name = '%s_%s_%s_%1.1f_%+d_%s_%s_%1.4f' % (self.cell_name, self.input_type, 'homogeneous', 2.0,
                                                           self.holding_potential, self.distribution, 'auto', self.weight)
        input_3 = np.load(join(self.sim_folder, 'synidx_%s.npy' % sim_name))
        self._draw_morph_to_axis(self.ax_morph_1, input_1, distribution=self.distribution)
        self._draw_morph_to_axis(self.ax_morph_2, input_2, distribution=self.distribution)
        self._draw_morph_to_axis(self.ax_morph_3, input_3, distribution=self.distribution)
        self.ax_morph_1.scatter(elec_x[self.elec_idx], elec_z[self.elec_idx], c='cyan', edgecolor='none', s=50)
        self.ax_morph_2.scatter(elec_x[self.elec_idx], elec_z[self.elec_idx], c='cyan', edgecolor='none', s=50)
        self.ax_morph_3.scatter(elec_x[self.elec_idx], elec_z[self.elec_idx], c='cyan', edgecolor='none', s=50)

    def _finitialize_figure(self):
        lines = []
        line_names = []
        for mu in self.mus:
            l, = plt.plot(0, 0, color=self.mu_clr[mu], lw=2)
            lines.append(l)
            line_names.append(self.mu_name_dict[mu])
        self.fig.legend(lines, line_names, frameon=False, loc='lower center', ncol=5)
        simplify_axes(self.fig.axes)
        self.fig.savefig(join(self.figure_folder, '%s_%s.png' % (self.figure_name, self.cell_name)), dpi=150)
        self.fig.savefig(join(self.figure_folder, '%s_%s.pdf' % (self.figure_name, self.cell_name)), dpi=150)

    def make_figure(self):
        for numb, input_sec in enumerate(self.input_secs):
            self._plot_sigs(numb, input_sec)

    def _plot_sigs(self, numb, input_sec):

        tau = '%1.2f' % self.tau_w if type(self.tau_w) in [int, float] else self.tau_w
        freqs = None
        LFP_dict = {}
        for mu in self.mus:
            sim_name = '%s_%s_%s_%1.1f_%+d_%s_%s_%1.4f' % (self.cell_name, self.input_type, input_sec, mu,
                                                           self.holding_potential, self.distribution, tau, self.weight)
            LFP = 1000 * np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))[self.elec_idx, :-1]
            num_tsteps = len(LFP)
            divide_into_welch = 16.
            welch_dict = {'Fs': 1000 / self.timeres,
                               'NFFT': int(num_tsteps/divide_into_welch),
                               'noverlap': int(num_tsteps/divide_into_welch/2.),
                               'window': plt.window_hanning,
                               'detrend': plt.detrend_mean,
                               'scale_by_freq': True,
                               }
            freqs, LFP_psd = aLFP.return_freq_and_psd_welch(LFP, welch_dict)
            LFP_dict[mu] = LFP_psd

        normalize = np.max([np.max(sig) for key, sig in LFP_dict.items()])
        ax_dict = {'xlim': [1, 450], 'xlabel': 'Hz', 'ylabel': 'Norm. LFP PSD'}
        ax0 = self.fig.add_subplot(1, 3, numb + 1, ylim=[1e-2, 1e0], **ax_dict)
        for key, sig in LFP_dict.items():
            ax0.loglog(freqs, sig[0] / normalize, c=self.mu_clr[key], lw=2)


class FigureDistanceStudy(PaperFigures):
    def __init__(self):
        PaperFigures.__init__(self)
        self.cell_name = 'hay'
        self.figure_name = 'figure_distance_study'
        self.conductance = 'generic'
        self.sim_folder = join(self.root_folder, 'generic_study', 'hay')
        self.timeres = 2**-4
        self.holding_potential = -80
        self.soma_idx = 0
        self.apic_idx = 605
        self.input_idx = self.apic_idx
        self.distribution = 'linear_increase'
        self.input_type = 'wn'
        self.tau_w = 'auto'
        self.weight = 0.0001
        self.mu = 2.0

        self._initialize_figure()
        self.make_figure()
        self._finitialize_figure()

    def return_ax_coors(self, mother_ax, pos, x_shift=0):
        ax_w = 0.12
        ax_h = 0.06
        xstart, ystart = self.fig.transFigure.inverted().transform(mother_ax.transData.transform(pos))
        return [xstart + x_shift, ystart, ax_w, ax_h]

    def _initialize_figure(self):
        plt.close('all')
        self.fig = plt.figure(figsize=[10, 8])
        self.fig.subplots_adjust(hspace=0.5, wspace=0.4, top=0.94, bottom=0.05, left=0.03, right=0.96)
        ax_dict = {'frame_on': False, 'xticks': [], 'yticks': []}
        lfp_ax_dict = {'xlim': [1, 450], 'xscale': 'log', 'yscale': 'log', 'xlabel': 'Hz',
                       'ylabel': 'LFP PSD [$\mu$V$^2$/Hz]'}
        self.amp_ax = self.fig.add_subplot(321, **ax_dict)#[0., 0.3, 0.8, 0.6]
        self.sig_ax = self.fig.add_subplot(323, xlim=[240, 290], xlabel='ms', xticks=[])
        self.freq_ax = self.fig.add_subplot(325, **ax_dict)
        self.lfp_ax = self.fig.add_subplot(322, ylim=[1e-7, 1e-4], **lfp_ax_dict)
        self.lfp2_ax = self.fig.add_subplot(324, ylim=[1e-9, 1e-5], **lfp_ax_dict)
        self.q_ax = self.fig.add_subplot(326, **ax_dict)
        self.lfp_ax.grid('on')
        self.lfp2_ax.grid('on')
        self.sig_ax.axis('off')

        self.sig_ax.plot([280, 290], [-1, -1], 'k', lw=4, clip_on=False)
        self.sig_ax.text(282, -1.3, '10 ms')

        #self.sig_ax

        mark_subplots(self.fig.axes, xpos=0.)

    def _finitialize_figure(self):

        simplify_axes(self.fig.axes)
        self.fig.savefig(join(self.figure_folder, '%s.png' % self.figure_name), dpi=150)
        self.fig.savefig(join(self.figure_folder, '%s.pdf' % self.figure_name), dpi=150)
        plt.close('all')

    def make_figure(self):

        sim_name = '%s_%s_%s_%1.1f_%+d_%s_%s' % (self.cell_name, self.input_type, str(self.input_idx), self.mu,
                                                 self.holding_potential, self.distribution, self.tau_w)
        distances = np.linspace(-2500, 2500, 100)
        heights = np.linspace(1850, -650, 50)
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

        lfp_trace_positions = np.array([[200, 0], [200, 500], [200, 1000], [200, 1500]])
        lfp_trace_elec_idxs = [np.argmin((elec_x - dist)**2 + (elec_z - height)**2) for dist, height in lfp_trace_positions]

        lfp2_trace_positions = np.array([[600, 0], [600, 500], [600, 1000], [600, 1500]])
        lfp2_trace_elec_idxs = [np.argmin((elec_x - dist)**2 + (elec_z - height)**2) for dist, height in lfp2_trace_positions]

        if 0:
            print "Recalculating extracellular potential"
            cell = self._return_cell(self.holding_potential, 'generic', 2.0, self.distribution, self.tau_w)
            cell.tstartms = 0
            cell.tstopms = 1
            cell.simulate(rec_imem=True)
            cell.imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))
            cell.tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))
            electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
            print "Calculating"
            electrode.calc_lfp()
            LFP = 1000 * electrode.LFP
            np.save(join(self.sim_folder, 'LFP_%s.npy' % sim_name), LFP)
            print "Saved LFP: ", 'LFP_%s.npy' % sim_name
            print "Calculated"
            if self.input_type is 'distributed_synaptic':
                print "Starting PSD calc"
                freqs, LFP_psd = aLFP.return_freq_and_psd_welch(LFP, self.welch_dict)
            else:
                freqs, LFP_psd = aLFP.return_freq_and_psd(self.timeres_python/1000., LFP)
            np.save(join(self.sim_folder, 'LFP_psd_%s.npy' % sim_name), LFP_psd)
            np.save(join(self.sim_folder, 'LFP_freq_%s.npy' % sim_name), freqs)
        else:
            LFP = np.load(join(self.sim_folder, 'LFP_%s.npy' % sim_name))
            LFP_psd = np.load(join(self.sim_folder, 'LFP_psd_%s.npy' % sim_name))
            freqs = np.load(join(self.sim_folder, 'LFP_freq_%s.npy' % sim_name))

        upper_idx_limit = np.argmin(np.abs(freqs - 500))

        num_elec_cols = len(set(elec_x))
        num_elec_rows = len(set(elec_z))
        dc = np.zeros((num_elec_rows, num_elec_cols))
        res_amp = np.zeros((num_elec_rows, num_elec_cols))
        max_value = np.zeros((num_elec_rows, num_elec_cols))
        # rms_value = np.zeros((num_elec_rows, num_elec_cols))
        freq_at_max = np.zeros((num_elec_rows, num_elec_cols))

        num_elec_cols = len(set(elec_x))
        num_elec_rows = len(set(elec_z))
        elec_idxs = np.arange(len(elec_x)).reshape(num_elec_rows, num_elec_cols)

        for elec in xrange(len(elec_z)):
            row, col = np.array(np.where(elec_idxs == elec))[:, 0]
            dc[row, col] = LFP_psd[elec, 1]
            max_value[row, col] = np.max(LFP_psd[elec, 1:upper_idx_limit])
            # rms_value[row, col] = np.sqrt(np.average(LFP_psd[elec, 1:upper_idx_limit])**2)
            freq_at_max[row, col] = freqs[1 + np.argmax(LFP_psd[elec, 1:upper_idx_limit])]

        res_freq = stats.mode(freq_at_max, axis=None)[0][0]
        res_idx = np.argmin(np.abs(freqs - res_freq))
        for elec in xrange(len(elec_z)):
            row, col = np.array(np.where(elec_idxs == elec))[:, 0]
            res_amp[row, col] = LFP_psd[elec, res_idx]

        # self.amp_ax.set_title('Max PSD')
        self.freq_ax.set_title('Frequency at max PSD')
        self.q_ax.set_title('PSD Q-value')

        q = max_value / dc

        max_q_value = 20.
        vmin = 1e-7
        vmax = 1e-4

        imshow_dict = {'extent': [np.min(distances), np.max(distances), np.min(heights), np.max(heights)],
                       'interpolation': 'nearest',
                       'aspect': 1,
                       #'cmap': plt.cm.#plt.cm.gist_yarg
                       }

        img_amp = self.amp_ax.imshow(max_value,  norm=LogNorm(), vmin=vmin, vmax=vmax, cmap=plt.cm.gist_yarg, **imshow_dict)
        img_freq = self.freq_ax.imshow(freq_at_max,  vmin=0., vmax=40., cmap=plt.cm.jet, **imshow_dict)
        img_q = self.q_ax.imshow(q, vmin=1, vmax=max_q_value, cmap=plt.cm.jet, **imshow_dict)
        # img_q = self.q_ax.contourf(q, cmap=plt.cm.gist_yarg, aspect=1, origin='upper', levels=[1, 3, 5, 7, 9],
        #                            extent=[np.min(distances), np.max(distances), np.min(heights), np.max(heights)], extend='max')

        self.amp_ax.contour(max_value, extent=[np.min(distances), np.max(distances), np.min(heights), np.max(heights)],
                      aspect=1, norm=LogNorm(), origin='upper', levels=[1e-7], colors=['g'])
        self.freq_ax.contour(max_value, extent=[np.min(distances), np.max(distances), np.min(heights), np.max(heights)],
                      aspect=1, norm=LogNorm(), origin='upper', levels=[1e-7], colors=['g'])
        self.q_ax.contour(max_value, extent=[np.min(distances), np.max(distances), np.min(heights), np.max(heights)],
                      aspect=1, norm=LogNorm(), origin='upper', levels=[1e-7], colors=['g'])

        # lfp_clr = ['lime', 'chocolate', 'yellow', 'purple']
        # lfp_clr = ['0.0', '0.25', '0.5', '.75']

        lfp_clr = lambda d: plt.cm.Reds(int(256./(len(lfp_trace_elec_idxs) + 0.5) * (d + 1)))
        lfp2_clr = lambda d: plt.cm.Blues(int(256./(len(lfp_trace_elec_idxs) + 0.5) * (d + 1)))
        tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))

        for numb, idx in enumerate(lfp_trace_elec_idxs):
            self.amp_ax.plot(elec_x[idx], elec_z[idx], 'o', c=lfp_clr(numb), ms=10)
            self.lfp_ax.loglog(freqs, LFP_psd[idx], lw=2, c=lfp_clr(numb))

        for numb, idx in enumerate(lfp2_trace_elec_idxs):
            self.amp_ax.plot(elec_x[idx], elec_z[idx], 'o', c=lfp2_clr(numb), ms=10)
            self.lfp2_ax.loglog(freqs, LFP_psd[idx], lw=2, c=lfp2_clr(numb))
            # self.LFP_arrow_to_axis([elec_x[idx], elec_z[idx]], self.amp_ax, ax_, c=lfp_clr[numb])

        self.sig_ax.plot(tvec, LFP[lfp_trace_elec_idxs[1]] / np.max(np.abs(LFP[lfp_trace_elec_idxs[1]])), lw=2, c=lfp_clr(1))
        self.sig_ax.plot(tvec, LFP[lfp_trace_elec_idxs[3]] / np.max(np.abs(LFP[lfp_trace_elec_idxs[3]])), lw=2, c=lfp_clr(3))
        self.sig_ax.plot([0, 1000], [0, 0], '--', lw=1, c='k')
            #self.sig_ax.plot(tvec, LFP[idx], lw=2, c=lfp_clr(numb))

        xstart = np.load(join(self.sim_folder, 'xstart_%s_%s.npy' % (self.cell_name, self.conductance)))
        xmid = np.load(join(self.sim_folder, 'xmid_%s_%s.npy' % (self.cell_name, self.conductance)))
        xend = np.load(join(self.sim_folder, 'xend_%s_%s.npy' % (self.cell_name, self.conductance)))
        zstart = np.load(join(self.sim_folder, 'zstart_%s_%s.npy' % (self.cell_name, self.conductance)))
        zend = np.load(join(self.sim_folder, 'zend_%s_%s.npy' % (self.cell_name, self.conductance)))
        zmid = np.load(join(self.sim_folder, 'zmid_%s_%s.npy' % (self.cell_name, self.conductance)))

        for ax in [self.freq_ax, self.q_ax, self.amp_ax]:
            [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2, color='w', zorder=2)
             for idx in xrange(len(xstart))]
        self.amp_ax.plot(xmid[self.input_idx], zmid[self.input_idx], 'y*', ms=10)
        self.amp_ax.plot([np.min(distances), np.min(distances)], [100, 600], lw=5, c='k', clip_on=False)
        self.amp_ax.text(np.min(distances) + 170, 400, '500 $\mu$m')

        clbar_args = {'orientation': 'vertical', 'shrink': 0.7,}

        cax_1 = self.fig.add_axes([0.45, 0.73, 0.01, 0.2])
        cax_2 = self.fig.add_axes([0.45, 0.06, 0.01, 0.2])
        cax_3 = self.fig.add_axes([0.97, 0.06, 0.01, 0.2])

        cl1 = plt.colorbar(img_amp, cax=cax_1, label='Max LFP PSD')
        cl2 = plt.colorbar(img_freq, cax=cax_2, label='Hz', extend='max', ticks=[0, 10, 20., 30., 40])
        cl3 = plt.colorbar(img_q, cax=cax_3, ticks=[1, 10, 20], extend='max')


class FigureDistanceStudyInfiniteNeurite(PaperFigures):
    def __init__(self):
        PaperFigures.__init__(self)
        self.cell_name = 'infinite_neurite'
        self.figure_name = 'inf_neur_distance_study'
        self.conductance = '-0.5_-0.5'
        self.sim_folder = join(self.root_folder, 'paper_simulations', 'infinite_neurite')
        self.timeres = 2**-4
        self.holding_potential = -80

        self.input_idx = 0
        self.distribution = 'linear_increase'
        self.input_type = 'white_noise'
        self.tau_w = 'auto'
        self.weight = 0.0001
        self.mu = 2.0

        self.timeres_NEURON = 2**-4
        self.timeres_python = 2**-4
        self.repeats = 2
        self.cut_off = 0
        self.end_t = 1000 * self.repeats
        self.stimuli = 'white_noise'

        self._initialize_figure()
        self.make_figure()
        self._finitialize_figure()

    def return_ax_coors(self, mother_ax, pos, x_shift=0):
        ax_w = 0.12
        ax_h = 0.06
        xstart, ystart = self.fig.transFigure.inverted().transform(mother_ax.transData.transform(pos))
        return [xstart + x_shift, ystart, ax_w, ax_h]

    def _initialize_figure(self):
        plt.close('all')
        self.fig = plt.figure(figsize=[10, 10])
        self.fig.subplots_adjust(hspace=0.5, wspace=0.4, top=0.94, bottom=0.05, left=0.03, right=0.96)
        ax_dict = {'frame_on': False, 'xticks': [], 'yticks': []}
        lfp_ax_dict = {'xlim': [1, 450], 'xscale': 'log', 'yscale': 'log', 'xlabel': 'Hz',
                       'ylabel': 'LFP'}
        self.amp_ax = self.fig.add_subplot(321, **ax_dict)#[0., 0.3, 0.8, 0.6]
        self.sig_ax = self.fig.add_subplot(322, xlim=[240, 290], xlabel='ms', xticks=[])
        self.freq_ax = self.fig.add_subplot(323, **ax_dict)
        self.lfp_ax = self.fig.add_subplot(324, ylim=[1e-6, 1e-2], **lfp_ax_dict)
        self.q_ax = self.fig.add_subplot(325, **ax_dict)
        self.lfp2_ax = self.fig.add_subplot(326, ylim=[1e-7, 1e-3], **lfp_ax_dict)
        self.lfp_ax.grid('on')
        self.lfp2_ax.grid('on')
        self.sig_ax.axis('off')

        self.sig_ax.plot([280, 290], [-1, -1], 'k', lw=4, clip_on=False)
        self.sig_ax.text(282, -1.2, '10 $ms$')

        #self.sig_ax

        mark_subplots(self.fig.axes, xpos=0.)

    def _finitialize_figure(self):

        simplify_axes(self.fig.axes)
        self.fig.savefig(join(self.figure_folder, '%s_%s.png' % (self.figure_name, self.conductance)), dpi=150)
        plt.close('all')

    def make_figure(self):

        sim_name = '%s_%s_%s_%+d_%0.4f' % (self.cell_name, str(self.input_idx), self.conductance,
                                                 self.holding_potential, self.weight)
        distances = np.linspace(-1500, 3500, 100)
        heights = np.linspace(1250, -1250, 50)
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

        lfp_trace_positions = np.array([[0, -20], [176.991, -20], [530.973, -20], [973.451, -20]])
        lfp_trace_elec_idxs = [np.argmin((elec_x - dist)**2 + (elec_z - height)**2) for dist, height in lfp_trace_positions]

        lfp2_trace_positions = np.array([[0, -300], [176.991, -300], [530.973, -300], [973.451, -300]])
        lfp2_trace_elec_idxs = [np.argmin((elec_x - dist)**2 + (elec_z - height)**2) for dist, height in lfp2_trace_positions]

        if 1:
            print "Recalculating extracellular potential"
            neural_sim = NeuralSimulations(self)
            cell = neural_sim._return_cell(self.holding_potential, self.conductance)
            cell.tstartms = 0
            cell.tstopms = 1
            cell.simulate(rec_imem=True)
            cell.imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))
            cell.tvec = np.load(join(self.sim_folder, 'tvec_%s.npy' % (self.cell_name)))
            electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
            print "Calculating"
            electrode.calc_lfp()
            LFP = 1000 * electrode.LFP
            np.save(join(self.sim_folder, 'LFP_%s.npy' % sim_name), LFP)
            print "Saved LFP: ", 'LFP_%s.npy' % sim_name
            print "Calculated"
            if self.input_type is 'distributed_synaptic':
                print "Starting PSD calc"
                freqs, LFP_psd = aLFP.return_freq_and_psd_welch(LFP, self.welch_dict)
            else:
                freqs, LFP_psd = aLFP.return_freq_and_psd(self.timeres_python/1000., LFP)
            np.save(join(self.sim_folder, 'LFP_psd_%s.npy' % sim_name), LFP_psd)
            np.save(join(self.sim_folder, 'LFP_freq_%s.npy' % sim_name), freqs)
        else:
            LFP = np.load(join(self.sim_folder, 'LFP_%s.npy' % sim_name))
            LFP_psd = np.load(join(self.sim_folder, 'LFP_psd_%s.npy' % sim_name))
            freqs = np.load(join(self.sim_folder, 'LFP_freq_%s.npy' % sim_name))

        upper_idx_limit = np.argmin(np.abs(freqs - 500))

        num_elec_cols = len(set(elec_x))
        num_elec_rows = len(set(elec_z))
        dc = np.zeros((num_elec_rows, num_elec_cols))
        res_amp = np.zeros((num_elec_rows, num_elec_cols))
        max_value = np.zeros((num_elec_rows, num_elec_cols))
        # rms_value = np.zeros((num_elec_rows, num_elec_cols))
        freq_at_max = np.zeros((num_elec_rows, num_elec_cols))

        num_elec_cols = len(set(elec_x))
        num_elec_rows = len(set(elec_z))
        elec_idxs = np.arange(len(elec_x)).reshape(num_elec_rows, num_elec_cols)

        for elec in xrange(len(elec_z)):
            row, col = np.array(np.where(elec_idxs == elec))[:, 0]
            dc[row, col] = LFP_psd[elec, 1]
            max_value[row, col] = np.max(LFP_psd[elec, 1:upper_idx_limit])
            # rms_value[row, col] = np.sqrt(np.average(LFP_psd[elec, 1:upper_idx_limit])**2)
            freq_at_max[row, col] = freqs[1 + np.argmax(LFP_psd[elec, 1:upper_idx_limit])]

        res_freq = stats.mode(freq_at_max, axis=None)[0][0]
        res_idx = np.argmin(np.abs(freqs - res_freq))
        for elec in xrange(len(elec_z)):
            row, col = np.array(np.where(elec_idxs == elec))[:, 0]
            res_amp[row, col] = LFP_psd[elec, res_idx]

        self.amp_ax.set_title('Max amplitude')
        self.freq_ax.set_title('Frequency at max amplitude\n(Mode: %1.1f Hz)' % res_freq)
        self.q_ax.set_title('q-value\nMax amp / amp[%1.1f Hz]' % freqs[1])

        q = max_value / dc

        max_q_value = 9.
        vmin = 1e-7
        vmax = 1e-4

        imshow_dict = {'extent': [np.min(distances), np.max(distances), np.min(heights), np.max(heights)],
                       'interpolation': 'none',
                       'aspect': 1,
                       'cmap': plt.cm.gist_yarg
                       }

        img_amp = self.amp_ax.imshow(max_value,  norm=LogNorm(), vmin=vmin, vmax=vmax, **imshow_dict)
        img_freq = self.freq_ax.imshow(freq_at_max,  vmin=0, vmax=30., **imshow_dict)

        img_q = self.q_ax.imshow(q, vmin=1, vmax=max_q_value, **imshow_dict)
        # img_q = self.q_ax.contourf(q, cmap=plt.cm.gist_yarg, aspect=1, origin='upper', levels=[1, 3, 5, 7, 9],
        #                            extent=[np.min(distances), np.max(distances), np.min(heights), np.max(heights)], extend='max')

        self.amp_ax.contour(max_value, extent=[np.min(distances), np.max(distances), np.min(heights), np.max(heights)],
                      aspect=1, norm=LogNorm(), origin='upper', levels=[1e-7], colors=['b'])
        self.freq_ax.contour(max_value, extent=[np.min(distances), np.max(distances), np.min(heights), np.max(heights)],
                      aspect=1, norm=LogNorm(), origin='upper', levels=[1e-7], colors=['b'])
        self.q_ax.contour(max_value, extent=[np.min(distances), np.max(distances), np.min(heights), np.max(heights)],
                      aspect=1, norm=LogNorm(), origin='upper', levels=[1e-7], colors=['b'])

        # lfp_clr = ['lime', 'chocolate', 'yellow', 'purple']
        # lfp_clr = ['0.0', '0.25', '0.5', '.75']

        lfp_clr = lambda d: plt.cm.Reds(int(256./(len(lfp_trace_elec_idxs)) * (d + 1)))
        lfp2_clr = lambda d: plt.cm.Greens(int(256./(len(lfp_trace_elec_idxs)) * (d + 1)))
        tvec = np.load(join(self.sim_folder, 'tvec_%s.npy' % (self.cell_name)))

        for numb, idx in enumerate(lfp_trace_elec_idxs):
            self.amp_ax.plot(elec_x[idx], elec_z[idx], 'o', c=lfp_clr(numb), ms=10)
            self.lfp_ax.loglog(freqs, LFP_psd[idx], lw=2, c=lfp_clr(numb))

        for numb, idx in enumerate(lfp2_trace_elec_idxs):
            self.amp_ax.plot(elec_x[idx], elec_z[idx], 'o', c=lfp2_clr(numb), ms=10)
            self.lfp2_ax.loglog(freqs, LFP_psd[idx], lw=2, c=lfp2_clr(numb))
            # self.LFP_arrow_to_axis([elec_x[idx], elec_z[idx]], self.amp_ax, ax_, c=lfp_clr[numb])

        self.sig_ax.plot(tvec, LFP[lfp_trace_elec_idxs[0]] / np.max(np.abs(LFP[lfp_trace_elec_idxs[0]])), lw=2, c=lfp_clr(0))
        self.sig_ax.plot(tvec, LFP[lfp_trace_elec_idxs[2]] / np.max(np.abs(LFP[lfp_trace_elec_idxs[2]])), lw=2, c=lfp_clr(2))
        self.sig_ax.plot([0, 1000], [0, 0], '--', lw=1, c='k')
            #self.sig_ax.plot(tvec, LFP[idx], lw=2, c=lfp_clr(numb))

        xstart = np.load(join(self.sim_folder, 'xstart_%s.npy' % self.cell_name))
        xmid = np.load(join(self.sim_folder, 'xmid_%s.npy' % self.cell_name))
        xend = np.load(join(self.sim_folder, 'xend_%s.npy' % self.cell_name))
        zstart = np.load(join(self.sim_folder, 'zstart_%s.npy' % self.cell_name))
        zend = np.load(join(self.sim_folder, 'zend_%s.npy' % self.cell_name))
        zmid = np.load(join(self.sim_folder, 'zmid_%s.npy' % self.cell_name))

        for ax in [self.freq_ax, self.q_ax, self.amp_ax]:
            [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2, color='y', zorder=2)
             for idx in xrange(len(xstart))]
        self.amp_ax.plot(xmid[self.input_idx], zmid[self.input_idx], 'y*', ms=10)
        self.amp_ax.plot([np.min(distances), np.min(distances)], [100, 600], lw=5, c='k', clip_on=False)
        self.amp_ax.text(np.min(distances) + 170, 400, '500 $\mu m$')

        clbar_args = {'orientation': 'vertical', 'shrink': 0.7,}

        cax_1 = self.fig.add_axes([0.45, 0.73, 0.01, 0.2])
        cax_2 = self.fig.add_axes([0.45, 0.4, 0.01, 0.2])
        cax_3 = self.fig.add_axes([0.45, 0.06, 0.01, 0.2])

        cl1 = plt.colorbar(img_amp, cax=cax_1, label='PSD')
        cl2 = plt.colorbar(img_freq, cax=cax_2, label='Hz', extend='max')
        cl3 = plt.colorbar(img_q, cax=cax_3, ticks=[1, 5, 9], extend='max')


if __name__ == '__main__':

    NewIntroFigure('hay', 'figure_1', 0.001, False).make_figure()
    # IntroFigures('hay', 'figure_2', 0.0005, True).make_figure()
    # Figure3(0.0005, False)
    # Figure4(0.0005, False)

    # FigureSystematic()
    # FigureTimeConstant()
    # FigureNeurite2(do_simulations=False).make_figure()
    # FigureDistributedSynaptic()
    # FigureDistanceStudy()
    # FigureDistanceStudyInfiniteNeurite()