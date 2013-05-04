import os
if not os.environ.has_key('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
import pylab as pl
import numpy as np
import sys
import neuron
try:
    from ipdb import set_trace
except:
    pass
from os.path import join

import aLFP

pl.rcParams.update({'font.size' : 8,
    'figure.facecolor' : '1',
    'wspace' : 0.5, 'hspace' : 0.5})
np.random.seed(1234)

def plot_comp_numbers(cell):
    for comp_idx in xrange(len(cell.xmid)):
        pl.plot(cell.zmid[comp_idx], cell.ymid[comp_idx],\
                marker='$%i$'%comp_idx, color='b', markersize=10)
    pl.show()
    pl.close('all')
    sys.exit()

def plot_all_currents(cell, syn, electrode, neural_sim_dict,
                      plotting_params, neur_input_params):
    print "Plotting ..."

    
    pl.close('all')    
    fig = pl.figure(figsize=[14,8])
    fig.suptitle("Model: %s, Input scaling: %s, Input index: %s, Active: %s"
                 %(neural_sim_dict['model'], neur_input_params['input_scaling'],
                   neur_input_params['input_idx'], neural_sim_dict['is_active']))

    
    pl.subplots_adjust(hspace=0.5)
    #ax1a = fig.add_axes([0.05, 0.05, 0.15, 0.22])
    ax1b = fig.add_axes([0.1, 0.1, 0.12, 0.22])
    ax2 = fig.add_axes([0.1, 0.4, 0.12, 0.22])
    ax3 = fig.add_axes([0.1, 0.70, 0.12, 0.22])
    ax4 = fig.add_axes([0.3, 0.1, 0.35, 0.75], frameon=False)
    ax5 = fig.add_axes([0.80, 0.1, 0.14, 0.80], aspect='auto', frameon=False)

    ax3.set_title('Soma membrane potential')
    ax3.set_xlabel('[ms]')
    ax3.set_ylabel('[mV]')
    ax3.plot(cell.tvec, cell.somav)

    ax2.plot(cell.tvec, cell.imem[0,:])
    ax2.set_title('Soma transmembrane current')
    ax2.set_xlabel('[ms]')
    ax2.set_ylabel('[nA]')

    #ax1a.set_title('Input current')
    #ax1a.set_xlabel('[ms]')
    ax1b.set_title('Input PDS')
    ax1b.set_xlabel('[Hz]')
    
    #ax1a.plot(cell.tvec, cell.noiseVec, label='Noise input')
    freqs, power = aLFP.return_psd(cell.noiseVec, neural_sim_dict)
    ax1b.loglog(freqs, power/np.max(power), label='Noise input psd')
    ax1b.grid(True)
    ax1b.set_xlim(1,1000)
    ax1b.set_ylim(1e-1, 1e0)
    

    for sec in neuron.h.allsec():
        idx = cell.get_idx(sec.name())
        a = np.array([x for x in idx if cell.ymid[x] > -300])
        #print idx, a
        #set_trace()
        if len(a) > 0:
            ax4.plot(np.r_[cell.xstart[a], cell.xend[a][-1]],
                     np.r_[cell.ystart[a], cell.yend[a][-1]],
                     color='gray', alpha=0.5, lw=2)

    ax4.plot(cell.xmid[neur_input_params['input_idx']],
             cell.ymid[neur_input_params['input_idx']],
             color='m', marker='D', 
             markersize=10)
    
    ax4.axis(plotting_params['cell_extent'])
    for elec in xrange(len(electrode.x)):
        xpos, ypos = electrode.x[elec], electrode.y[elec]
        ax4.plot(xpos, ypos, 'gD')
        pixel_coor = ax4.transData.transform((xpos, ypos))
        fig_coor = fig.transFigure.inverted().transform((pixel_coor))
        ax_temp = fig.add_axes([fig_coor[0], fig_coor[1], 0.08, 0.12], frameon=False)
        ax_temp.set_xticklabels([])
        ax_temp.set_yticklabels([])
        ax_temp.grid(True)
        ax_temp.loglog(electrode.freqs, electrode.LFP_psd[elec]/np.max(electrode.LFP_psd[elec]))
        ax_temp.set_xlim(1,1000)
        ax_temp.set_ylim(1e-2, 1e0)
    stick = aLFP.return_dipole_stick(cell, ax4)
    sc_stick = ax5.imshow(stick, interpolation='nearest',
                          extent=[cell.tvec[0], cell.tvec[-1], ax4.axis()[2], ax4.axis()[3]],
                          vmax=np.max(np.abs(stick)), vmin=-np.max(np.abs(stick)), cmap='jet_r')
    ax5.axis('auto')
    
    #set_trace()
    ax5.set_title('Summed Imem along y-axis')
    ax5.set_ylabel('y [$\mu m$]')
    ax5.set_xlabel('Time [ms]')
    pl.colorbar(sc_stick, cax = fig.add_axes([0.95, 0.1, 0.01, 0.8]))
    name = "%s_%s_%s_%s" %(neural_sim_dict['model'],
                           neur_input_params['input_scaling'],
                           neur_input_params['input_idx'],
                           neural_sim_dict['is_active'])
    pl.savefig('WN_%s.png' % name)
    #pl.show()

