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

    def __init__(self, cell_name, input_type):

        self.cell_name = cell_name
        self.input_type = input_type
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
        self.mus = [2, 0, -0.5]
        self.mu_clr = lambda mu: plt.cm.Dark2(int(256. * (mu - np.min(self.mus))/
                                                  (np.max(self.mus) - np.min(self.mus))))

        self.sec_clr_dict = {'soma': '0.3', 'dend': '0.5', 'apic': '0.7', 'axon': '0.1'}
        self._set_cell_specific_properties()
        self._set_input_specific_properties()

    def _set_input_specific_properties(self):
        if self.input_type == 'wn':
            print "Single white noise input"
            self.plot_psd = True
            self._single_neural_sim_function = self._run_single_wn_simulation
            self.cut_off = 0
            self.end_t = 1000
            self.max_freq = 500
        elif self.input_type == 'synaptic':
            print "Single synaptic input"
            self.plot_psd = False
            self._single_neural_sim_function = self._run_single_synaptic_simulation
            self.cut_off = 0
            self.end_t = 30
        else:
            raise RuntimeError("Unrecognized input type.")

    def _set_cell_specific_properties(self):
        if self.cell_name == 'hay':
            self.elec_x = np.ones(3) * 100
            self.elec_y = np.zeros(3)
            self.elec_z = np.linspace(0, 1000, 3)
            self.electrode_parameters = {
                'sigma': 0.3,
                'x': self.elec_x,
                'y': self.elec_y,
                'z': self.elec_z
                }
            self.elec_markers = ['o', 'D', 's']
            self.cell_plot_idxs = [0, 455, 605]
            self.comp_markers = ['o', 'D', 's']

    def _return_cell(self, holding_potential, conductance_type, mu, distribution, tau_w):
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
                                     'mu_factor': mu,
                                     'g_pas': 0.0002,
                                     'distribution': distribution,
                                     'tau_w': tau_w,
                                     'total_w_conductance': 6.23843378791,
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


    def save_neural_sim_single_input_data(self, cell, electrode, input_idx,
                             mu, distribution, taum):

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
            sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, mu,
                                                        self.holding_potential, distribution, taum)
        elif type(input_idx) in [list, np.ndarray]:
            sim_name = '%s_%s_multiple_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, mu,
                                                              self.holding_potential, distribution, taum)
        else:
            raise RuntimeError("input_idx is not recognized!")

        np.save(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)), cell.tvec)
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

        if self.plot_psd:
            xvec, yvec = aLFP.return_freq_and_psd(tvec, sig)
        else:
            xvec = tvec
            yvec = sig

        for idx, ax in enumerate(ax_list):
            ax.plot(xvec, yvec[idx], color=self.mu_clr(mu), lw=2)

    def _plot_signals(self, fig, input_idx, distribution, tau_w):
        ax_vmem_1 = fig.add_subplot(3, 5, 3)
        ax_vmem_2 = fig.add_subplot(3, 5, 8)
        ax_vmem_3 = fig.add_subplot(3, 5, 13)

        ax_imem_1 = fig.add_subplot(3, 5, 4)
        ax_imem_2 = fig.add_subplot(3, 5, 9)
        ax_imem_3 = fig.add_subplot(3, 5, 14)

        ax_sig_1 = fig.add_subplot(3, 5, 5)
        ax_sig_2 = fig.add_subplot(3, 5, 10)
        ax_sig_3 = fig.add_subplot(3, 5, 15)

        ax_imem_1.set_title('Transmembrane\ncurrents', color='b')
        ax_vmem_1.set_title('Membrane\npotential', color='b')
        ax_sig_1.set_title('Extracellular\npotential', color='g')

        xlabel = '$Hz$' if self.plot_psd else '$ms$'
        [ax.set_ylabel('$nA$', color='b') for ax in [ax_imem_3, ax_imem_2, ax_imem_1]]
        [ax.set_ylabel('$\mu V$', color='g') for ax in [ax_sig_3, ax_sig_2, ax_sig_1]]
        [ax.set_ylabel('$mV$', color='b') for ax in [ax_vmem_3, ax_vmem_2, ax_vmem_1]]
        [ax.set_xlabel(xlabel, color='b') for ax in [ax_vmem_3, ax_vmem_2, ax_vmem_1]]
        [ax.set_xlabel(xlabel, color='b') for ax in [ax_imem_3, ax_imem_2, ax_imem_1]]
        [ax.set_xlabel(xlabel, color='g') for ax in [ax_sig_3, ax_sig_2, ax_sig_1]]

        tvec = np.load(join(self.sim_folder, 'tvec_%s_%s.npy' % (self.cell_name, self.input_type)))

        lines = []
        line_names = []

        for mu in self.mus:
            sim_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, mu,
                                               self.holding_potential, distribution, tau_w)
            LFP = np.load(join(self.sim_folder, 'sig_%s.npy' % sim_name))
            vmem = np.load(join(self.sim_folder, 'vmem_%s.npy' % sim_name))
            imem = np.load(join(self.sim_folder, 'imem_%s.npy' % sim_name))

            lines.append(plt.plot(0, 0, color=self.mu_clr(mu), lw=2)[0])
            line_names.append('$\mu_{factor} = %1.1f$' % mu)
            self._plot_sig_to_axes([ax_sig_3, ax_sig_2, ax_sig_1], LFP, tvec, mu)
            self._plot_sig_to_axes([ax_vmem_3, ax_vmem_2, ax_vmem_1], vmem[self.cell_plot_idxs], tvec, mu)
            self._plot_sig_to_axes([ax_imem_3, ax_imem_2, ax_imem_1], imem[self.cell_plot_idxs], tvec, mu)

        ax_list = [ax_vmem_1, ax_vmem_2, ax_vmem_3, ax_imem_1, ax_imem_2, ax_imem_3,
                   ax_sig_1, ax_sig_2, ax_sig_3]

        if self.plot_psd:
            [ax.set_xticks([1, 10, 100]) for ax in ax_list]
            [ax.set_xscale('log') for ax in ax_list]
            [ax.set_yscale('log') for ax in ax_list]
            [ax.set_xlim([0, self.max_freq]) for ax in ax_list]

            for ax in [ax_vmem_1, ax_vmem_2, ax_vmem_3]:
                max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()[1:]) for l in ax.get_lines()])))
                ax.set_ylim([10**(max_exponent - 4), 10**max_exponent])

            for ax in [ax_imem_1, ax_imem_2, ax_imem_3]:
                max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()) for l in ax.get_lines()])))
                ax.set_ylim([10**(max_exponent - 4), 10**(max_exponent)])

            for ax in [ax_sig_1, ax_sig_2, ax_sig_3]:
                max_exponent = np.ceil(np.log10(np.max([np.max(l.get_ydata()) for l in ax.get_lines()])))
                ax.set_ylim([10**(max_exponent - 4), 10**(max_exponent)])
        else:
            [ax.set_xticks(ax.get_xticks()[::2]) for ax in ax_list]
            [ax.set_yticks(ax.get_yticks()[::2]) for ax in ax_list]
            # [ax.set_ylim([self.holding_potential - 1, self.holding_potential + 10])
            #     for ax in [ax_vmem_1, ax_vmem_2, ax_imem_3]]

        color_axes([ax_sig_3, ax_sig_2, ax_sig_1], 'g')
        color_axes([ax_imem_3, ax_imem_2, ax_imem_1, ax_vmem_3, ax_vmem_2, ax_vmem_1], 'b')
        simplify_axes(ax_list)
        mark_subplots(ax_list, 'efghijklmn')
        fig.legend(lines, line_names, frameon=False, ncol=3, loc='lower right')

    def plot_summary(self, input_idx, distribution, tau_w):

        plt.close('all')
        fig = plt.figure(figsize=[12, 8])
        fig.subplots_adjust(hspace=0.5, wspace=0.7, top=0.9, bottom=0.13,
                            left=0.1, right=0.95)

        self._draw_setup_to_axis(fig, input_idx, distribution)
        self._plot_parameter_distributions(fig, input_idx, distribution, tau_w)
        self._plot_signals(fig, input_idx, distribution, tau_w)
        filename = ('generic_summary_%s_%s_%d_%s_%1.2f'
                                             % (self.cell_name, self.input_type, input_idx,
                                                distribution, tau_w))
        filename = '%s_psd' % filename if self.plot_psd else filename
        fig.savefig(join(self.figure_folder, '%s.png' % filename))

    def _draw_setup_to_axis(self, fig, input_idx, distribution=None, plotpos=152):

        ax = fig.add_subplot(plotpos)

        elec_x = np.load(join(self.sim_folder, 'elec_x_%s.npy' % self.cell_name))
        elec_z = np.load(join(self.sim_folder, 'elec_z_%s.npy' % self.cell_name))
        xstart = np.load(join(self.sim_folder, 'xstart_%s.npy' % self.cell_name))
        zstart = np.load(join(self.sim_folder, 'zstart_%s.npy' % self.cell_name))
        xend = np.load(join(self.sim_folder, 'xend_%s.npy' % self.cell_name))
        zend = np.load(join(self.sim_folder, 'zend_%s.npy' % self.cell_name))
        xmid = np.load(join(self.sim_folder, 'xmid_%s.npy' % self.cell_name))
        zmid = np.load(join(self.sim_folder, 'zmid_%s.npy' % self.cell_name))


        ax.plot(xmid[input_idx], zmid[input_idx], 'y*', zorder=1, ms=15)

        if not distribution is None:
            mark_subplots(ax, 'd', xpos=0, ypos=1)
            ax.plot(xmid[self.cell_plot_idxs], zmid[self.cell_plot_idxs], 'bD', zorder=2, ms=5, mec='none')
            example_name = '%s_%s_%d_%1.1f_%+d_%s_%1.2f' % (self.cell_name, self.input_type, input_idx, 0,
                                               self.holding_potential, distribution, 1)
            dist_dict = np.load(join(self.sim_folder, 'dist_dict_%s.npy' % example_name)).item()
            sec_clrs = dist_dict['sec_clrs']
        else:
            mark_subplots(ax, 'a', xpos=0, ypos=1)
            sec_clrs = ['0.7'] * len(xmid)

        [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], lw=2,
                 color=sec_clrs[idx], zorder=0) for idx in xrange(len(xmid))]
        ax.plot(xmid[0], zmid[0], 'o', color=sec_clrs[0], zorder=0, ms=10, mec='none')

        ax.scatter(elec_x, elec_z, c='g', edgecolor='none', s=50)

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

    def _make_white_noise_stimuli(self, cell, input_idx):
        input_scaling = 0.005
        max_freq = 500
        input_array = input_scaling * self._make_WN_input(cell, max_freq)
        noise_vec = neuron.h.Vector(input_array)
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

    def set_input_spiketrain(self, cell, all_spike_trains, cell_input_idxs, spike_train_idxs, synapse_params):
        """ Makes synapses and feeds them predetermined spiketimes """
        synapse_params = {
            'e': 0,
            'syntype': 'ExpSynI',      #conductance based exponential synapse
            'tau': .1,                #Time constant, rise           #Time constant, decay
            'weight': 0.0001,           #Synaptic weight
            'color': 'r',              #for pl.plot
            'marker': '.',             #for pl.plot
            'record_current': False,    #record synaptic currents
            }

        num_trains = 1000

        plt.seed(1234)
        cell_input_idxs = cell.get_rand_idx_area_norm(section=input_pos, nidx=num_trains,
                                                      z_min=minpos, z_max=maxpos)
        for number, comp_idx in enumerate(cell_input_idxs):
            synapse_params.update({'idx': int(comp_idx)})
            s = LFPy.Synapse(cell, **synapse_params)
            train = LFPy.inputgenerators.stationary_poisson(1, 5, cell.tstartms, cell.tstopms)[0]
            print train
            s.set_spike_times(train)

    def _quickplot_setup(self, cell, electrode):
        plt.plot(cell.xmid[self.cell_plot_idxs], cell.zmid[self.cell_plot_idxs], 'bD', zorder=2, ms=5, mec='none')
        [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], lw=2, zorder=0, color='gray')
         for idx in xrange(len(cell.xmid))]
        plt.plot(cell.xmid[0], cell.zmid[0], 'o', zorder=0, ms=10, mec='none', color='gray')
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
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)
        self.save_neural_sim_single_input_data(cell, electrode, input_idx, mu, distribution, tau_w)

    def calculate_total_conductance(self, distribution):
        cell = self._return_cell(self.holding_potential, 'generic', 0, distribution, 1)
        total_conductance = 0
        for sec in cell.allseclist:
            for seg in sec:
                # Never mind the units, as long as it is consistent
                total_conductance += nrn.area(seg.x) * seg.gm_QA
        print distribution, total_conductance

    def run_all_single_simulations(self):
        distributions = ['uniform', 'linear_decrease', 'linear_increase']
        input_idxs = [0, 605, 455]
        tau_ws = [1, 0.1, 10, 100]
        make_summary_plot = True
        for distribution in distributions:
            for tau_w in tau_ws:
                for input_idx in input_idxs:
                    for mu in self.mus:
                        self._single_neural_sim_function(mu, input_idx, distribution, tau_w)
                    if make_summary_plot:
                        self.plot_summary(input_idx, distribution, tau_w)

    def _run_multiple_wn_simulation(self, mu, input_idxs, distribution, tau_w):
        plt.seed(1234)
        electrode = LFPy.RecExtElectrode(**self.electrode_parameters)
        neuron.h('forall delete_section()')
        cell = self._return_cell(self.holding_potential, 'generic', mu, distribution, tau_w)
        syns = []
        noise_vecs = []
        for input_idx in input_idxs:
            cell, syn, noise_vec = self._make_white_noise_stimuli(cell, input_idx)
            syns.append(syn)
            noise_vecs.append(noise_vec)

        print "Starting simulation ..."
        cell.simulate(rec_imem=True, rec_vmem=True, electrode=electrode)
        self.save_neural_sim_single_input_data(cell, electrode, input_idxs, mu, distribution, tau_w)

    def run_all_multiple_input_simulations(self):
        distributions = ['uniform', 'linear_decrease', 'linear_increase']
        input_idxs = [0, 605, 455]
        tau_ws = [10, 100, 1, 0.1]
        for tau_w in tau_ws:
            for distribution in distributions:
                for mu in self.mus:
                    self._run_multiple_wn_simulation(mu, input_idxs, distribution, tau_w)
            self.plot_multiple_input_EC_signals(tau_w)

    def plot_summaries(self):
        distributions = ['uniform', 'linear_decrease', 'linear_increase']
        taums = [1, 0.1, 10, 100]
        input_idxs = [0, 605, 455]
        for distribution in distributions:
            for taum in taums:
                for input_idx in input_idxs:
                    self.plot_summary(input_idx, distribution, taum)

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
                lines.append(plt.plot(0, 0, color=self.mu_clr(mu), lw=2)[0])
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
                lines.append(plt.plot(0, 0, color=self.mu_clr(mu), lw=2)[0])
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


if __name__ == '__main__':

    gs = GenericStudy('hay', 'wn')
    gs.run_all_multiple_input_simulations()
    # gs.run_all_single_simulations()
    # gs.combine_extracellular_traces(0)
    # gs.combine_extracellular_traces(455)
    # gs.combine_extracellular_traces(605)
    # gs.plot_multiple_input_EC_signals()
    # gs.plot_summary(0, 'uniform', 1)

    # gs.calculate_total_conductance('linear_decrease')
    # gs.calculate_total_conductance('linear_increase')
    # gs.calculate_total_conductance('uniform')
    # gs.plot_distributions(-80)
    # gs.plot_summaries()