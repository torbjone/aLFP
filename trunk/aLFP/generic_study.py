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
        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
        self.timeres = 2**-4
        self.holding_potentials = [-80, -70, -60]
        self.cut_off = 0
        self.end_t = 100
        self.sec_clr_dict = {'soma': 'k', 'dend': 'b', 'apic': 'r', 'axon': 'm'}
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

    def _get_distribution(self, dist_dict, cell, linearized_mod_name):
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
                        try:
                            dist_dict[key][idx] = eval('seg.%s' % key)
                        except NameError:
                            pass
                idx += 1
        return dist_dict

    def plot_distributions(self, holding_potential):
        cell = self._return_cell(holding_potential, 'Ih_linearized')
        linearized_mod_name = "Ih_linearized_v2"

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
        dist_dict = self._get_distribution(dist_dict, cell, linearized_mod_name)

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

if __name__ == '__main__':
    gs = GenericStudy('hay')
    gs.plot_distributions(-80)