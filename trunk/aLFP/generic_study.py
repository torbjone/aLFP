__author__ = 'torbjone'

import sys
import os
from os.path import join
import numpy as np
import pylab as plt
from plotting_convention import *
import neuron
nrn = neuron.h
import LFPy
import aLFP

class GenericStudy:

    def __init__(self, cell_name):

        self.cell_name = cell_name
        self.root_folder = join('/home', 'torbjone', 'work', 'aLFP')
        self.figure_folder = join('/home', 'torbjone', 'work', 'aLFP', 'generic_study')
        self.sim_folder = join('/home', 'torbjone', 'work', 'aLFP', 'generic_study', cell_name)
        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
        if not os.path.isdir(self.sim_folder):
            os.mkdir(self.sim_folder)

        self.timeres = 2**-4
        self.holding_potentials = [-80, -70, -60]
        self.holding_potential = -80
        self.phis = [0.5, 0, -2]
        self.phi_clr = lambda phi: plt.cm.Dark2(int(256. * (phi + 2)/2.5 ))
        self.cut_off = 0
        self.end_t = 100
        self.sec_clr_dict = {'soma': '0.3', 'dend': '0.5', 'apic': '0.7', 'axon': '0.1'}
        self._set_cell_specific_properties()


    def _set_cell_specific_properties(self):
        if self.cell_name == 'hay':
            self.elec_x = np.ones(3) * 100
            self.elec_y = np.zeros(3)
            self.elec_z = np.linspace(0, 1000, 3)
            self.elec_markers = ['o', 'D', 's']
            self.cell_plot_idxs = [0, 305, 400]
            self.comp_markers = ['o', 'D', 's']

    def _return_cell(self, holding_potential, conductance_type, phi, distribution, taum):
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
                                     'phi': phi,
                                     'distribution': distribution,
                                     'taum': taum,
                                     'total_conductance': 0.623843378791,
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


    def save_neural_sim_data(self, cell, electrode, input_idx,
                             phi, distribution, taum):

        if not os.path.isdir(self.sim_folder):
            os.mkdir(self.sim_folder)

        dist_dict = {'dist': np.zeros(cell.totnsegs),
             'sec_clrs': np.zeros(cell.totnsegs, dtype='|S3'),
             'gm_QA': np.zeros(cell.totnsegs),
             'em_QA': np.zeros(cell.totnsegs),
             'v': np.zeros(cell.totnsegs),
             'phi_QA': np.zeros(cell.totnsegs),
             'taum_QA': np.zeros(cell.totnsegs),
             }
        dist_dict = self._get_distribution(dist_dict, cell)

        sim_name = '%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, input_idx, phi, self.holding_potential,
                                           distribution, taum)
        np.save(join(self.sim_folder, 'tvec_%s.npy' % self.cell_name), cell.tvec)
        np.save(join(self.sim_folder, 'dist_dict_%s.npy' % sim_name), dist_dict)
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


    def plot_parameter_distributions(self, fig, input_idx, distribution, taum):

        ax0 = fig.add_subplot(351, title='$g_{m}$', xlabel='$\mu m$', ylim=[0, 0.0001],
                              xticks=[0, 400, 800, 1200], ylabel='$S/cm^2$')
        ax1 = fig.add_subplot(356, title='$\phi$',
                              xlabel='$\mu m$', xticks=[0, 400, 800, 1200])
        ax2 = fig.add_subplot(3, 5, 11, title=r'$\tau_m$', ylim=[0, taum*2],
                              xlabel='$\mu m$', xticks=[0, 400, 800, 1200])
        mark_subplots([ax0, ax1, ax2], 'abc')
        simplify_axes([ax0, ax1, ax2])

        for phi in self.phis:
            sim_name = '%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, input_idx, phi,
                                               self.holding_potential, distribution, taum)
            dist_dict = np.load(join(self.sim_folder, 'dist_dict_%s.npy' % sim_name)).item()
            ax0.scatter(dist_dict['dist'], dist_dict['gm_QA'], c=dist_dict['sec_clrs'],
                        edgecolor='none')
            ax1.scatter(dist_dict['dist'], dist_dict['phi_QA'], c=dist_dict['sec_clrs'],
                        edgecolor='none')
            ax2.scatter(dist_dict['dist'], dist_dict['taum_QA'], c=dist_dict['sec_clrs'],
                        edgecolor='none')
        return [ax0, ax1, ax2]

    def _plot_sig_to_axes(self, ax_list, sig, tvec, phi):
        if not len(ax_list) == len(sig):
            raise RuntimeError("Something wrong with number of electrodes!")
        for idx, ax in enumerate(ax_list):
            ax.plot(tvec, sig[idx], color=self.phi_clr(phi), lw=2)

    def _plot_signals(self, fig, input_idx, distribution, taum):
        ax_vmem_1 = fig.add_subplot(3, 5, 3,
                                    ylim=[self.holding_potential - 1, self.holding_potential + 10])
        ax_vmem_2 = fig.add_subplot(3, 5, 8,
                                    ylim=[self.holding_potential - 1, self.holding_potential + 10])
        ax_vmem_3 = fig.add_subplot(3, 5, 13,
                                    ylim=[self.holding_potential - 1, self.holding_potential + 10])

        ax_imem_1 = fig.add_subplot(3, 5, 4)
        ax_imem_2 = fig.add_subplot(3, 5, 9)
        ax_imem_3 = fig.add_subplot(3, 5, 14)

        ax_sig_1 = fig.add_subplot(3, 5, 5)
        ax_sig_2 = fig.add_subplot(3, 5, 10)
        ax_sig_3 = fig.add_subplot(3, 5, 15)

        ax_imem_1.set_title('Transmembrane\ncurrents', color='b')
        ax_vmem_1.set_title('Membrane\npotential', color='b')
        ax_sig_1.set_title('Extracellular\npotential', color='g')
        [ax.set_ylabel('$nA$', color='b') for ax in [ax_imem_3, ax_imem_2, ax_imem_1]]
        [ax.set_ylabel('$\mu V$', color='g') for ax in [ax_sig_3, ax_sig_2, ax_sig_1]]
        [ax.set_ylabel('$mV$', color='b') for ax in [ax_vmem_3, ax_vmem_2, ax_vmem_1]]
        [ax.set_xlabel('$ms$', color='b') for ax in [ax_vmem_3, ax_vmem_2, ax_vmem_1]]
        [ax.set_xlabel('$ms$', color='b') for ax in [ax_imem_3, ax_imem_2, ax_imem_1]]
        [ax.set_xlabel('$ms$', color='g') for ax in [ax_sig_3, ax_sig_2, ax_sig_1]]

        tvec = np.load(join(self.sim_folder, 'tvec_%s.npy' % self.cell_name))

        lines = []
        line_names = []

        for phi in self.phis:
            sim_name = '%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, input_idx, phi,
                                               self.holding_potential, distribution, taum)
            LFP = np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))
            vmem = np.load(join(self.sim_folder, 'vmem_%s.npy' % sim_name))
            imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))
            lines.append(plt.plot(0, 0, color=self.phi_clr(phi), lw=2)[0])
            line_names.append('$\phi = %1.1f$' % phi)
            self._plot_sig_to_axes([ax_sig_3, ax_sig_2, ax_sig_1], LFP, tvec, phi)
            self._plot_sig_to_axes([ax_vmem_3, ax_vmem_2, ax_vmem_1], vmem[self.cell_plot_idxs], tvec, phi)
            self._plot_sig_to_axes([ax_imem_3, ax_imem_2, ax_imem_1], imem[self.cell_plot_idxs], tvec, phi)

        ax_list = [ax_vmem_1, ax_vmem_2, ax_vmem_3, ax_imem_1, ax_imem_2, ax_imem_3,
                   ax_sig_1, ax_sig_2, ax_sig_3]

        [ax.set_yticks(ax.get_yticks()[::3]) for ax in ax_list]
        color_axes([ax_sig_3, ax_sig_2, ax_sig_1], 'g')
        color_axes([ax_imem_3, ax_imem_2, ax_imem_1, ax_vmem_3, ax_vmem_2, ax_vmem_1], 'b')
        simplify_axes(ax_list)
        mark_subplots(ax_list, 'efghijklmn')
        fig.legend(lines, line_names, frameon=False, ncol=3, loc='lower right')

    def plot_summary(self, input_idx, distribution, taum):

        plt.close('all')
        fig = plt.figure(figsize=[12, 8])
        fig.subplots_adjust(hspace=0.5, wspace=0.7, top=0.9, bottom=0.1,
                            left=0.07, right=0.95)

        self._draw_setup_to_axis(fig, input_idx, distribution)
        self.plot_parameter_distributions(fig, input_idx, distribution, taum)
        self._plot_signals(fig, input_idx, distribution, taum)
        fig.savefig(join(self.figure_folder, 'generic_summary_%d_%s_%1.2f.png'
                                             % (input_idx, distribution, taum)))


    def _draw_setup_to_axis(self, fig, input_idx, distribution):
        example_name = '%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, input_idx, 0,
                                               self.holding_potential, distribution, 1)
        dist_dict = np.load(join(self.sim_folder, 'dist_dict_%s.npy' % example_name)).item()
        ax = fig.add_subplot(152)
        mark_subplots(ax, 'd', xpos=0, ypos=1)
        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name))
        xstart = np.load(join(self.sim_folder, 'xstart_%s.npy' % self.cell_name))
        zstart = np.load(join(self.sim_folder, 'zstart_%s.npy' % self.cell_name))
        xend = np.load(join(self.sim_folder, 'xend_%s.npy' % self.cell_name))
        zend = np.load(join(self.sim_folder, 'zend_%s.npy' % self.cell_name))
        xmid = np.load(join(self.sim_folder, 'xmid_%s.npy' % self.cell_name))
        zmid = np.load(join(self.sim_folder, 'zmid_%s.npy' % self.cell_name))

        ax.plot(xmid[self.cell_plot_idxs], zmid[self.cell_plot_idxs], 'bD', zorder=2, ms=5, mec='none')
        ax.plot(xmid[input_idx], zmid[input_idx], 'y*', zorder=1, ms=15)
        [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
                  color=dist_dict['sec_clrs'][idx], zorder=0) for idx in xrange(len(xmid))]
        ax.plot(xmid[0], zmid[0], 'o', color=dist_dict['sec_clrs'][0], zorder=0, ms=10, mec='none')
        ax.scatter(elec_x, elec_z, c='g', edgecolor='none', s=50)

        ax.axis('off')

    def _make_syaptic_stimuli(self, cell, input_idx):
        import LFPy
        # Define synapse parameters
        synapse_parameters = {
            'idx': input_idx,
            'e': 0.,                   # reversal potential
            'syntype': 'ExpSyn',       # synapse type
            'tau': 10.,                # syn. time constant
            'weight': 0.005,            # syn. weight
            'record_current': True,
        }

        synapse = LFPy.Synapse(cell, **synapse_parameters)
        synapse.set_spike_times(np.array([5.]))
        return cell, synapse, None

    def _run_single_generic_simulation(self, phi, input_idx, distribution, taum):
        electrode_parameters = {
            'sigma': 0.3,
            'x': self.elec_x,
            'y': self.elec_y,
            'z': self.elec_z
        }

        electrode = LFPy.RecExtElectrode(**electrode_parameters)
        neuron.h('forall delete_section()')
        cell = self._return_cell(self.holding_potential, 'generic', phi, distribution, taum)
        self._make_syaptic_stimuli(cell, input_idx)
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)
        self.save_neural_sim_data(cell, electrode, input_idx, phi, distribution, taum)

    def calculate_total_conductance(self, distribution):
        cell = self._return_cell(self.holding_potential, 'generic', 0, distribution, 1)

        total_conductance = 0
        for sec in cell.allseclist:
            for seg in sec:
                # Never mind the units, as long as it is consistent
                total_conductance += nrn.area(seg.x) * seg.gm_QA
        print distribution, total_conductance

    def run_simulations(self):
        distributions = ['linear_decrease']#, 'linear_increase', 'uniform']
        input_idxs = [0, 305, 400, 200, 500]
        taums = [10, 0.1, 1]

        for distribution in distributions:
            for taum in taums:
                for input_idx in input_idxs:
                    for phi in self.phis:
                        self._run_single_generic_simulation(phi, input_idx, distribution, taum)

    def plot_summaries(self):
        distributions = ['linear_decrease', 'linear_increase', 'uniform']
        taums = [10, 0.1, 1]
        input_idxs = [0, 305, 400, 200, 500]
        for distribution in distributions:
            for taum in taums:
                for input_idx in input_idxs:
                    self.plot_summary(input_idx, distribution, taum)

if __name__ == '__main__':
    gs = GenericStudy('hay')
    # gs.calculate_total_conductance('linear_decrease')
    # gs.calculate_total_conductance('linear_increase')
    # gs.calculate_total_conductance('uniform')
    # gs.run_simulations()
    # gs.plot_distributions(-80)
    gs.plot_summaries()