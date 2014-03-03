
import pylab as plt
from matplotlib.colors import LogNorm
import numpy as np
import sys
import neuron
from neuron import h as nrn

import neuron
try:
    from ipdb import set_trace
except:
    pass
from os.path import join
import LFPy
import aLFP
import matplotlib.mlab as mlab

plt.rcParams.update({'font.size' : 8,
    'figure.facecolor' : '1',
    'wspace' : 0.5, 'hspace' : 0.5})
#np.random.seed(1234)

def plot_comp_numbers(cell, elec_x, elec_y, elec_z):
    plt.axis('equal')
    for comp_idx in xrange(len(cell.xmid)):
        plt.plot(cell.xmid[comp_idx], cell.zmid[comp_idx],\
                marker='$%i$'%comp_idx, color='b', markersize=10)
    plt.scatter(elec_x, elec_z)
    plt.show()
    plt.close('all')
    sys.exit()

def arrow_to_axis(pos, ax_origin, ax_target, clr, x_shift):

    if x_shift > 0:
        upper_pixel_coor = ax_target.transAxes.transform(([0,0.5]))
        lower_pixel_coor = ax_target.transAxes.transform(([0,0]))
        shift = 1000*x_shift
    else:
        upper_pixel_coor = ax_target.transAxes.transform(([1,0.5]))
        lower_pixel_coor = ax_target.transAxes.transform(([1,0]))   
        shift = 2000*x_shift
    
    upper_coor = ax_origin.transData.inverted().transform(upper_pixel_coor)
    lower_coor = ax_origin.transData.inverted().transform(lower_pixel_coor)
    
    upper_line_x = [pos[0], upper_coor[0] + shift]
    upper_line_y = [pos[1], upper_coor[1]]
    
    lower_line_x = [pos[0], lower_coor[0] + shift]
    lower_line_y = [pos[1], lower_coor[1]]    
    #lower_line_x = [origin_coor[0], lower_target_coor[0]]
    #lower_line_y = [origin_coor[1], lower_target_coor[1]]
    
    ax_origin.plot(upper_line_x, upper_line_y, lw=1, 
              color=clr, clip_on=False, alpha=1.)
    #ax_origin.plot(lower_line_x, lower_line_y, lw=1, 
    #          color=clr, clip_on=False, alpha=1.)

def plot_WN_cell_to_ax(ax, xstart, xmid, xend, zstart, zmid, zend, elec_x, elec_z,
                    electrodes, input_idx, elec_clr, soma_idx, apical_idx, tuft_idx, basal_idx):

    [ax.plot([xstart[idx], xend[idx]], [zstart[idx], zend[idx]], color='grey') for idx in xrange(len(xstart))]
    [ax.plot(elec_x[electrodes[idx]], elec_z[electrodes[idx]], 'o', color=elec_clr(idx)) for idx in xrange(len(electrodes))]
    ax.plot(xmid[input_idx], zmid[input_idx], 'g*', ms=20)
    ax.plot(xmid[soma_idx], zmid[soma_idx], 'ms')
    ax.plot(xmid[apical_idx], zmid[apical_idx], 'bD')
    ax.plot(xmid[basal_idx], zmid[basal_idx], 'r*', ms=10)
    ax.plot(xmid[tuft_idx], zmid[tuft_idx], 'k^')

def plot_WN_CA1(folder, elec_x, elec_y, elec_z, input_idx, vrest, syn_strength):

    conductance_list = ['passive_%d' % vrest, 'only_km_%d' % vrest,
                        'only_hd_%d' % vrest, 'active_%d' % vrest]

    tuft_idx = 452
    apical_idx = 433
    basal_idx = 20
    soma_idx = 0

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

    electrodes = np.array([1, 3, 5])
    elec_clr = lambda idx: plt.cm.rainbow(int(256. * idx/(len(electrodes) - 1.)))

    clr = lambda idx: plt.cm.jet(int(256. * idx/(len(conductance_list ) - 1)))

    v_min = -90
    v_max = 0


    plt.close('all')
    fig = plt.figure(figsize=[12,8])
    fig.suptitle('Synaptic input: %s, synaptic strength: %1.2f, V_rest: %1.2f'
                 % (input_idx, syn_strength, vrest))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    ax_morph = fig.add_axes([0.02, 0.1, 0.15, 0.8], frameon=False, xticks=[], yticks=[])
    plot_WN_cell_to_ax(ax_morph, xstart, xmid, xend, zstart, zmid, zend, elec_x,
                    elec_z, electrodes, input_idx, elec_clr, soma_idx, apical_idx, tuft_idx, basal_idx)

    # Somatic Vm
    ax_s1 = fig.add_axes([0.5, 0.05, 0.09, 0.1], ylim=[v_min, v_max], xlabel='ms', ylabel='Somatic Vm [mV]')
    ax_s2 = fig.add_axes([0.63, 0.05, 0.09, 0.1], xlabel='ms', ylabel='Shifted somatic Vm [mV]')
    ax_s3 = fig.add_axes([0.76, 0.05, 0.09, 0.1], xlim=[1, 500], xlabel='Hz', ylim=[1e-6, 1e0],
                         ylabel='Somatic Vm PSD [$mV^2$]')
    ax_s4 = fig.add_axes([0.9, 0.05, 0.09, 0.1], xlim=[1, 500], xlabel='Hz', ylim=[1e-9, 1e1],
                         ylabel='Somatic imem PSD')
    # Apic Vm
    ax_a1 = fig.add_axes([0.5, 0.45, 0.09, 0.1], ncols, 10, ylim=[v_min, v_max],
                         xlabel='ms', ylabel='Apical Vm [mV]')
    ax_a2 = fig.add_axes([0.63, 0.45, 0.09, 0.1], 11, xlabel='ms', ylabel='Shifted apical Vm [mV]')
    ax_a3 = fig.add_axes([0.76, 0.45, 0.09, 0.1], xlim=[1, 500], xlabel='Hz', ylabel='Apical Vm PSD [$mV^2$]',
                         ylim=[1e-6,1e0])
    ax_a4 = fig.add_axes([0.9, 0.45, 0.09, 0.1], xlim=[1, 500], xlabel='Hz', ylim=[1e-9,1e-3],
                         ylabel='Apic imem PSD')
    # Basal Vm
    ax_b1 = fig.add_axes([0.5, 0.25, 0.09, 0.1], ncols, 10, ylim=[v_min, v_max],
                         xlabel='ms', ylabel='Basal Vm [mV]')
    ax_b2 = fig.add_axes([0.63, 0.25, 0.09, 0.1], 11, xlabel='ms', ylabel='Shifted basal Vm [mV]')
    ax_b3 = fig.add_axes([0.76, 0.25, 0.09, 0.1], xlim=[1, 500], xlabel='Hz', ylabel='Basal Vm PSD [$mV^2$]',
                         ylim=[1e-6,1e0])
    ax_b4 = fig.add_axes([0.9, 0.25, 0.09, 0.1], xlim=[1, 500], xlabel='Hz',
                         ylim=[1e-9,1e-3], ylabel='Basal imem PSD')
    # Tuft Vm
    ax_t1 = fig.add_axes([0.5, 0.65, 0.09, 0.1], ncols, 10, ylim=[v_min, v_max],
                         xlabel='ms', ylabel='Apical tuft Vm [mV]')
    ax_t2 = fig.add_axes([0.63, 0.65, 0.09, 0.1], 11, xlabel='ms', ylabel='Shifted apical tuft Vm [mV]')
    ax_t3 = fig.add_axes([0.76, 0.65, 0.09, 0.1], xlim=[1, 500], xlabel='Hz', ylabel='Apical tuft Vm PSD [$mV^2$]',
                         ylim=[1e-6,1e0])
    ax_t4 = fig.add_axes([0.9, 0.65, 0.09, 0.1], xlim=[1, 500], xlabel='Hz',
                         ylim=[1e-9,1e-3], ylabel='Apical tuft imem PSD')


    # Electrodes
    ax_e1 = fig.add_axes([0.2, 0.1, 0.1, 0.2], xlabel='ms', ylabel='$\mu V$')
    ax_e2 = fig.add_axes([0.2, 0.4, 0.1, 0.2], xlabel='ms', ylabel='$\mu V$')
    ax_e3 = fig.add_axes([0.2, 0.7, 0.1, 0.2], xlabel='ms', ylabel='$\mu V$', title='Extracellular')

    ax_e1_psd = fig.add_axes([0.35, 0.1, 0.1, 0.2], xlim=[1, 500], xlabel='Hz', ylabel='$\mu V^2$', ylim=[1e-8,1e-3])
    ax_e2_psd = fig.add_axes([0.35, 0.4, 0.1, 0.2], xlim=[1, 500], xlabel='Hz', ylabel='$\mu V^2$', ylim=[1e-8,1e-3])
    ax_e3_psd = fig.add_axes([0.35, 0.7, 0.1, 0.2], xlim=[1, 500], xlabel='Hz', ylabel='$\mu V^2$', ylim=[1e-8,1e-3],
                             title='Extracellular PSD')

    e_ax = [ax_e1, ax_e2, ax_e3]
    e_ax_psd = [ax_e1_psd, ax_e2_psd, ax_e3_psd]

    axes = [ax_s1, ax_s2, ax_s3, ax_a1, ax_a2, ax_a3, ax_e1, ax_t1, ax_t2, ax_t3,ax_b1, ax_b2, ax_b3,
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
    ax_t3.grid(True)
    ax_b3.grid(True)

    ax_s4.grid(True)
    ax_a4.grid(True)
    ax_t4.grid(True)
    ax_b4.grid(True)

    ax_e1_psd.grid(True)
    ax_e2_psd.grid(True)
    ax_e3_psd.grid(True)

    #ax_s4.grid(True)
    lines = []
    line_names = []
    freqs = np.load(join(folder, 'freqs.npy'))
    for idx, conductance_type in enumerate(conductance_list):

        identifier = '%d_%1.3f_%+1.3f_%s.npy' %(input_idx, syn_strength, 0, conductance_type)
        #somav = np.load(join(folder, 'somav_%s' %identifier))
        vmem = np.load(join(folder, 'vmem_%s' %identifier))
        vmem_psd = np.load(join(folder, 'vmem_psd_%s' %identifier))

        #imem = np.load(join(folder, 'imem_%s' %identifier))
        imem_psd = np.load(join(folder, 'imem_psd_%s' %identifier))

        sig = np.load(join(folder, 'signal_%s' %(identifier)))[:,:]
        sig_psd = np.load(join(folder, 'psd_%s' %(identifier)))[:,:]


        for eidx, elec in enumerate(electrodes):
            e_ax[eidx].plot(sig[eidx] - sig[eidx, 0], color=clr(idx))
            e_ax_psd[eidx].loglog(freqs, sig_psd[eidx], color=clr(idx))

        ax_s1.plot(vmem[soma_idx], color=clr(idx))
        ax_s2.plot(vmem[soma_idx] - vmem[soma_idx, 0], color=clr(idx))
        ax_s3.loglog(freqs, vmem_psd[soma_idx], color=clr(idx))
        ax_s4.loglog(freqs, imem_psd[soma_idx], color=clr(idx))

        ax_t1.plot(vmem[tuft_idx], color=clr(idx))
        ax_t2.plot(vmem[tuft_idx] - vmem[tuft_idx, 0], color=clr(idx))
        ax_t3.loglog(freqs, vmem_psd[tuft_idx], color=clr(idx))
        ax_t4.loglog(freqs, imem_psd[tuft_idx], color=clr(idx))

        ax_b1.plot(vmem[basal_idx], color=clr(idx))
        ax_b2.plot(vmem[basal_idx] - vmem[basal_idx, 0], color=clr(idx))
        ax_b3.loglog(freqs, vmem_psd[basal_idx], color=clr(idx))
        ax_b4.loglog(freqs, imem_psd[basal_idx], color=clr(idx))

        ax_a1.plot(vmem[apical_idx], color=clr(idx))
        ax_a2.plot(vmem[apical_idx] - vmem[apical_idx, 0], color=clr(idx))
        ax_a3.loglog(freqs, vmem_psd[apical_idx], color=clr(idx))
        l, = ax_a4.loglog(freqs, imem_psd[apical_idx], color=clr(idx))

        line_names.append(conductance_type)
        lines.append(l)

    ax_s1.plot(ax_s1.get_xlim()[0], ax_s1.get_ylim()[1], 'ms', clip_on=False)
    ax_s2.plot(ax_s2.get_xlim()[0], ax_s2.get_ylim()[1], 'ms', clip_on=False)
    ax_s3.plot(ax_s3.get_xlim()[0], ax_s3.get_ylim()[1], 'ms', clip_on=False)
    ax_s4.plot(ax_s4.get_xlim()[0], ax_s4.get_ylim()[1], 'ms', clip_on=False)

    ax_a1.plot(ax_a1.get_xlim()[0], ax_a1.get_ylim()[1], 'bD', clip_on=False)
    ax_a2.plot(ax_a2.get_xlim()[0], ax_a2.get_ylim()[1], 'bD', clip_on=False)
    ax_a3.plot(ax_a3.get_xlim()[0], ax_a3.get_ylim()[1], 'bD', clip_on=False)
    ax_a4.plot(ax_a4.get_xlim()[0], ax_a4.get_ylim()[1], 'bD', clip_on=False)

    ax_b1.plot(ax_b1.get_xlim()[0], ax_b1.get_ylim()[1], 'r*', clip_on=False, ms=10)
    ax_b2.plot(ax_b2.get_xlim()[0], ax_b2.get_ylim()[1], 'r*', clip_on=False, ms=10)
    ax_b3.plot(ax_b3.get_xlim()[0], ax_b3.get_ylim()[1], 'r*', clip_on=False, ms=10)
    ax_b4.plot(ax_b4.get_xlim()[0], ax_b4.get_ylim()[1], 'r*', clip_on=False, ms=10)

    ax_t1.plot(ax_t1.get_xlim()[0], ax_t1.get_ylim()[1], 'k^', clip_on=False)
    ax_t2.plot(ax_t2.get_xlim()[0], ax_t2.get_ylim()[1], 'k^', clip_on=False)
    ax_t3.plot(ax_t3.get_xlim()[0], ax_t3.get_ylim()[1], 'k^', clip_on=False)
    ax_t4.plot(ax_t4.get_xlim()[0], ax_t4.get_ylim()[1], 'k^', clip_on=False)


    ax_sparse = [ax_a1, ax_a2, ax_s1, ax_s2, ax_e1, ax_e2, ax_e3]
    for ax in ax_sparse:
        ax.set_xticks(ax.get_xticks()[::2])

    fig.legend(lines, line_names, frameon=False)

    fig.savefig(join('WN_figs', 'WN_%d_%1.2f_%+1.2f.png' %
                (input_idx, syn_strength, vrest)), dpi=150)



def plot_active_currents(ifolder, input_scaling, input_idx, plot_params, simulation_params,
                         plot_compartments, conductance_type, epas=None):

    if epas == None:
        cur_name = '%d_%1.3f_%s' %(input_idx, input_scaling, conductance_type)
    else:
        cur_name = '%d_%1.3f_%s_%g' %(input_idx, input_scaling, conductance_type, epas)

    # Loading all needed data
    imem = np.load(join(ifolder, 'imem_%s.npy' %(cur_name)))
    imem_psd = np.load(join(ifolder, 'imem_psd_%s.npy' %(cur_name)))
    somav_psd = np.load(join(ifolder, 'somav_psd_%s.npy' %(cur_name)))
    somav = np.load(join(ifolder, 'somav_%s.npy' %(cur_name)))
    icap = np.load(join(ifolder, 'icap_psd_%s.npy' %(cur_name)))
    ipas = np.load(join(ifolder, 'ipas_psd_%s.npy' %(cur_name)))

    active_dict = {}
    clr_list = ['r', 'b', 'g', 'm']
    
    for cur in simulation_params['rec_variables']:
        try:
            active_dict[cur] = np.load(join(ifolder, '%s_psd_%s.npy'%(cur, cur_name)))
        except:
            print "Failed to load ", cur
            pass
    
    freqs = np.load(join(ifolder, 'freqs.npy'))    
    tvec = np.load(join(ifolder, 'tvec.npy'))
    xmid = np.load(join(ifolder, 'xmid.npy' ))
    ymid = np.load(join(ifolder, 'ymid.npy' ))
    zmid = np.load(join(ifolder, 'zmid.npy' ))
    xstart = np.load(join(ifolder, 'xstart.npy' ))
    ystart = np.load(join(ifolder, 'ystart.npy' ))
    zstart = np.load(join(ifolder, 'zstart.npy' ))
    xend = np.load(join(ifolder, 'xend.npy' ))
    yend = np.load(join(ifolder, 'yend.npy' ))
    zend = np.load(join(ifolder, 'zend.npy' ))    
    diam = np.load(join(ifolder, 'diam.npy'))
    
    # Initializing figure
    plt.close('all')    
    fig = plt.figure(figsize=[14,8])
    fig.suptitle("Model: %s, input_idx: %d, input_scaling: %g "%(ifolder, input_idx, input_scaling))
    plt.subplots_adjust(hspace=0.5)
    ax_vm = fig.add_axes([0.05, 0.60, 0.10, 0.20], title='Soma $V_m$')    
    ax_vm_psd = fig.add_axes([0.05, 0.30, 0.10, 0.20], title='Soma $V_m$ PSD')    

    ax_neur = fig.add_axes([0.45, 0.1, 0.35, 0.75], frameon=False, aspect='equal', xticks=[], yticks=[])
    ax_vm_psd.grid(True)
    ax_vm.set_xlabel('ms')
    ax_vm.set_ylabel('mV')
    ax_vm_psd.set_xlabel('Hz')
    ax_vm_psd.set_ylabel('mV')
    ax_vm.set_ylim(-110,-20)
    ax_vm_psd.set_ylim(1e-2,1e1)
    ax_vm_psd.set_xlim(1e0,1e3)

    #Sorting compartemts after y-height
    argsort = np.argsort([ymid[comp] for comp in plot_compartments])
    plot_compartments = np.array(plot_compartments)[argsort[:]]
    comp_clr_list = []
    for idx in xrange(len(plot_compartments)):
        i = 256. * idx
        if len(plot_compartments) > 1:
            i /= len(plot_compartments) - 1.
        else:
            i /= len(plot_compartments)
        comp_clr_list.append(plt.cm.rainbow(int(i)))
    ymin = plot_params['ymin']
    ymax = plot_params['ymax']
    yticks = np.arange(ymin, ymax + 1 , 250)
    yticks = yticks[np.where((np.min(ymid) <= yticks) * (yticks <= np.max(ymid)))]
    ext_ax_width = 0.2
    ext_ax_height = 1.0/len(plot_compartments)
    clr = 'k'
    
    # Starting plotting
    ax_vm_psd.loglog(freqs, somav_psd, color=clr, lw=2)
    ax_vm.plot(tvec, somav, color=clr, lw=2)
    
    for comp in xrange(len(xmid)):
        ax_neur.plot([xstart[comp], xend[comp]], [ystart[comp], yend[comp]], lw=diam[comp], color='gray')
    ax_neur.axis([-200, 200, ymin -50, ymax + 50])
    ax_neur.plot(xmid[input_idx], ymid[input_idx], '*', color='y', label='Input', markersize=15)
    ax_neur.legend(bbox_to_anchor=[.6, 1.05], numpoints=1)
    
    for numb, comp in enumerate(plot_compartments):
        ax_neur.plot(xmid[comp], ymid[comp], 'o', color=comp_clr_list[numb])
        if numb % 2:
            x_shift = -ext_ax_width 
        else:
            x_shift = 0.2
        ax_temp = fig.add_axes([0.5 + x_shift, 0.1 + numb*(ext_ax_height+0.05)/2, 
                                ext_ax_width, ext_ax_height])
        ax_temp.tick_params(color=comp_clr_list[numb])
        for spine in ax_temp.spines.values():
            spine.set_edgecolor(comp_clr_list[numb])
        #ax_temp.set_yticklabels([])
        ax_temp.grid(True)        
        for cur_numb, cur in enumerate(active_dict):

            ax_temp.plot(freqs[:], active_dict[cur][comp,:], label=cur, color=clr_list[cur_numb], lw=2)
            ax_temp.plot(1.5, active_dict[cur][comp,0], '+', color=clr_list[cur_numb])
            #except ValueError:
            #    print "Skipping ", cur
            #     pass
        ax_temp.plot(freqs, ipas[comp,:], label='Ipas', color='y', lw=2)
        ax_temp.plot(1.5, ipas[comp,0], '+', color='y')
        ax_temp.plot(freqs, icap[comp,:], label='Icap', color='grey', lw=2)
        ax_temp.plot(1.5, icap[comp,0], '+', color='grey')
        
        ax_temp.plot(freqs, imem_psd[comp], color=clr, lw=1, label='Imem')      
        ax_temp.plot(1.5, imem_psd[comp,0], '+', color=clr)
        pos = [xmid[comp], ymid[comp]]
        if numb == 0:
            #ax_temp.legend(bbox_to_anchor=[1.4, 1.22])
            ax_temp.set_xlabel('Hz')
            ax_temp.set_ylabel('Norm. amp')
            ax_temp.legend(bbox_to_anchor=[1.4,1.0])
            #ax_temp.axis(ax_temp.axis('tight'))
        ax_temp.set_xlim(1,1000)
        ax_temp.set_ylim(1e-10, 1e0)
        ax_temp.set_xscale('log')
        ax_temp.set_yscale('log')
        ax_temp.set_yticks(ax_temp.get_yticks()[::2])
        #ax_temp.set_yticklabels(ax_temp.get_yticklabels()[::2])
        #ax_temp.set_yticks([10**(4*n) for n in np.arange(-5,1)])
        #ax_temp.set_yticklabels(['10$^{%d}$' %(4*n) for n in np.arange(-3,1)])
        arrow_to_axis(pos, ax_neur, ax_temp, comp_clr_list[numb], x_shift)

    #xmin, xmax = ax_neur.get_xaxis().get_view_interval()
    #ax_neur.add_artist(plt.Line2D((xmin, xmin), (np.min(ymid), np.max(ymid)), color='b', linewidth=3))
    plt.savefig('%s_%s.png' % (ifolder, cur_name), dpi=150)


def plot_cell_probe(folder, simulation_params, syn_strength, shift):


    name = '%s_probe_%1.2f_%d.npy' %('%s', syn_strength, shift)

    # Loading all needed data
    xmid = np.load(join(folder, 'xmid.npy' ))
    ymid = np.load(join(folder, 'ymid.npy' ))
    zmid = np.load(join(folder, 'zmid.npy' ))
    xstart = np.load(join(folder, 'xstart.npy' ))
    ystart = np.load(join(folder, 'ystart.npy' ))
    zstart = np.load(join(folder, 'zstart.npy' ))
    xend = np.load(join(folder, 'xend.npy' ))
    yend = np.load(join(folder, 'yend.npy' ))
    zend = np.load(join(folder, 'zend.npy' ))    
    diam = np.load(join(folder, 'diam.npy'))
    
    imem = np.load(join(folder, name % 'imem'))
    vmem = np.load(join(folder, name % 'vmem'))
    icap = np.load(join(folder, name % 'icap'))
    ipas = np.load(join(folder, name % 'ipas'))

    ionic_currents = {}
    clr_list = ['r', 'b', 'g', 'm']

    clrs = {'imem': 'grey',
            'ipas': 'y',
            'icap': 'k',
            'ik':  'r',
            'ina': 'm',
            'i_hd': 'g',
            'ica': 'b'
            }
    names = ['ipas', 'icap', 'ina', 'ik', 'ica', 'i_hd']


    ion_names = ['ina', 'ik', 'ica', 'i_hd']
    
    ik_names = ['ik_km', 'ik_KahpM95', 'ik_kad', 'ik_kap', 'ik_kdr']
    
    colors = ['y','k','m','r', 'b', 'g']

    
    norm = lambda f: np.abs(f - np.mean(f))
    rms = lambda f: np.sqrt(np.average((f- np.mean(f))**2)) 
    
    for ion in simulation_params['rec_variables']:
        ionic_currents[ion] = np.load(join(folder, name % ion))

    comp_list = [0, 20, 100, 275, 282, 433, 452, 470, 475]
        
    for comp in comp_list:

        plt.close('all')
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        ax1 = plt.subplot(121, frameon=False, xticks=[], yticks=[], aspect='equal')
        [ax1.plot([xstart[i], xend[i]], [zstart[i], zend[i]], 'k') for i in xrange(len(diam))]
        ax1.plot([xstart[comp], xend[comp]], [zstart[comp], zend[comp]], 'r', lw=2)

        
        ax2 = plt.subplot(422, title='vmem', ylim=[np.min(vmem), np.max(vmem)])
        ax2.plot(np.arange(len(vmem[comp])), vmem[comp])

        ax2 = plt.subplot(424, ylim=[np.min(imem), np.max(imem)])
        ax2.plot(np.arange(len(imem[comp])), imem[comp], label='imem', color=clrs['imem'])
        ax2.plot(np.arange(len(ipas[comp])), ipas[comp], label='ipas', color=clrs['ipas'])
        ax2.plot(np.arange(len(icap[comp])), icap[comp], label='icap', color=clrs['icap'])
        
        for ion in ion_names:
            ax2.plot(np.arange(len(imem[comp])), ionic_currents[ion][comp], label=ion, color=clrs[ion])
        plt.legend(frameon=False, bbox_to_anchor=[1.25,1])

        ax3 = plt.subplot(426)
        ax3.plot(np.arange(len(ipas[comp])), ipas[comp] - ipas[comp, 0], label='ipas', color=clrs['ipas'])
        ax3.plot(np.arange(len(icap[comp])), icap[comp] - icap[comp, 0], label='icap', color=clrs['icap'])
        
        for ion in ion_names:
            ax3.plot(np.arange(len(imem[comp])), ionic_currents[ion][comp] - ionic_currents[ion][comp, 0],
                     label=ion, color=clrs[ion])
        
        ax4 = plt.subplot(4, 4, 15, title='Imem Fractions', aspect='equal')
        
        stack = [rms(ipas[comp]), rms(icap[comp]), rms(ionic_currents['ina'][comp]),
                 rms(ionic_currents['ik'][comp]), rms(ionic_currents['ica'][comp]),
                 rms(ionic_currents['i_hd'][comp])]
        stack /= np.sum(stack)
        ax4.pie(stack, labels=names, colors=colors)
        

        ax5 = plt.subplot(4, 4, 16, title='Ik Fractions', aspect='equal')
        ik_stack = []
        for ik in ik_names:
            ik_stack.append(rms(ionic_currents[ik][comp]))
        ik_stack /= np.sum(ik_stack)
        ax5.pie(ik_stack, labels=ik_names, colors=colors)
        plt.savefig(join('comps', '%04d_%1.2f_%d.png' % (comp, syn_strength, shift)))


def compare_cell_currents(folder, syn_strength, shift, conductance_list, input_pos):

    # Loading all needed data
    xmid = np.load(join(folder, 'xmid.npy' ))
    ymid = np.load(join(folder, 'ymid.npy' ))
    zmid = np.load(join(folder, 'zmid.npy' ))
    xstart = np.load(join(folder, 'xstart.npy' ))
    ystart = np.load(join(folder, 'ystart.npy' ))
    zstart = np.load(join(folder, 'zstart.npy' ))
    xend = np.load(join(folder, 'xend.npy' ))
    yend = np.load(join(folder, 'yend.npy' ))
    zend = np.load(join(folder, 'zend.npy' ))    
    diam = np.load(join(folder, 'diam.npy'))

    comp_markers = '*osD^<>+x'
    cond_colors = 'kgbmr'
    
    imem_dict = {}
    vmem_dict = {}

    for conductance_type in conductance_list:
        stem = '%s_%s_1.00_%1.3f_%d_sim_0.npy' %(conductance_type, input_pos, syn_strength, shift)
        imem_dict[conductance_type] = np.load(join(folder, 'imem_%s' % stem))
        vmem_dict[conductance_type] = np.load(join(folder, 'vmem_%s' % stem))

    divide_into_welch = 4.
    comp_list = np.array([0, 20, 282, 433, 452, 475])
    argsort = np.argsort(-ymid[comp_list])
    nrows = len(comp_list)

    plt.close('all')
    fig = plt.figure(figsize=[9,10])
    fig.subplots_adjust(hspace=0.5, wspace=0.6, bottom=0.05, top=0.90, right=0.98)
    
    ax0 = fig.add_axes([0, 0, 0.25, 0.9], frameon=False, xticks=[], yticks=[], aspect='equal')
    [ax0.plot([xstart[i], xend[i]], [ystart[i], yend[i]], 'grey') for i in xrange(len(diam))]

    fig.suptitle('Synaptic strength: %1.2f\nPassive reversal potential shift: %+d' %(syn_strength, shift))
    
    lines = []
    line_names = []
    for numb, comp in enumerate(comp_list[argsort]):
        ax0.plot(xmid[comp], ymid[comp], comp_markers[numb], color='k')
        ax1 = fig.add_subplot(nrows, 5, 5 * numb + 2)
        ax2 = fig.add_subplot(nrows, 5, 5 * numb + 3)
        ax3 = fig.add_subplot(nrows, 5, 5 * numb + 4)
        ax4 = fig.add_subplot(nrows, 5, 5 * numb + 5, ylim=[1e-16, 1e-8])

        ax2.grid(True)
        ax4.grid(True)
        if numb == 0:
            ax1.set_title('Vmem')
            ax2.set_title('Vmem PSD')
            ax3.set_title('Imem')
            ax4.set_title('Imem PSD')
            ax1.set_xlabel('ms')
            ax1.set_ylabel('mV')
            ax2.set_xlabel('Hz')
            ax2.set_ylabel('mV$^2$/Hz')
            ax3.set_xlabel('ms')
            ax3.set_ylabel('nA')
            ax4.set_xlabel('Hz')
            ax4.set_ylabel('mV$^2$/Hz')
        for cond_number, conductance_type in enumerate(conductance_list):
            vmem = vmem_dict[conductance_type][comp, :]
            imem = imem_dict[conductance_type][comp, :]
            
            vmem_psd, freqs = mlab.psd(vmem, Fs=1000., NFFT=int(len(vmem)/divide_into_welch),
                                                        noverlap=int(len(vmem)/divide_into_welch/2),
                                                        window=plt.window_hanning, detrend=plt.detrend_mean)

            imem_psd, freqs = mlab.psd(imem, Fs=1000., NFFT=int(len(imem)/divide_into_welch),
                                                        noverlap=int(len(imem)/divide_into_welch/2),
                                                        window=plt.window_hanning, detrend=plt.detrend_mean)
            l, = ax1.plot(np.arange(len(imem)), vmem, color=cond_colors[cond_number])
            ax2.loglog(freqs, vmem_psd, color=cond_colors[cond_number], lw=0.5)
            ax3.plot(np.arange(len(imem)), imem, color=cond_colors[cond_number], lw=0.5)
            ax4.loglog(freqs, imem_psd, color=cond_colors[cond_number], lw=0.5)
            if numb == 0:
                lines.append(l)
                line_names.append(conductance_type)
            
        for ax in [ax1, ax3]:
            ax.set_xticks(ax.get_xticks()[::2])
            #ax.set_yticks(ax.get_yticks()[::2])
        for ax in [ax1, ax2, ax3, ax4]:
            ax.plot(ax.get_xticks()[0], ax.get_yticks()[-1], comp_markers[numb], color='k', clip_on=False)
    fig.legend(lines, line_names, frameon=False, loc='upper left')

    fig.savefig('vmem_imem_%1.2f_%+d_-65.png' %(syn_strength, shift), dpi=150)


def plot_transfer_functions(ifolder, input_scaling, input_idx, plot_params, simulation_params,
                            plot_compartments):

    cur_name = '%d_%1.3f' %(input_idx, input_scaling)

    # Loading all needed data
    #imem = np.load(join(ifolder, 'imem_%s.npy' %(cur_name)))
    imem_active = np.load(join(ifolder, 'imem_psd_%d_%1.3f_%s.npy' %(input_idx, input_scaling, 'active')))
    imem_reduced = np.load(join(ifolder, 'imem_psd_%d_%1.3f_%s.npy' %(input_idx, input_scaling, 'reduced_Ih')))
    imem_passive = np.load(join(ifolder, 'imem_psd_%d_%1.3f_%s.npy' %(input_idx, input_scaling, 'passive')))
    #icap = np.load(join(ifolder, 'icap_psd_%s.npy' %(cur_name)))
    #ipas = np.load(join(ifolder, 'ipas_psd_%s.npy' %(cur_name)))
    active_dict = {}
    clr_list = ['r', 'b', 'g', 'm']
    
    #for cur in simulation_params['rec_variables']:
    #    try:
    #        active_dict[cur] = np.load(join(ifolder, '%s_psd_%s.npy'%(cur, cur_name)))
    #        print cur, active_dict[cur]
    #    except:
    #        print "Failed to load ", cur
    #        pass
    freqs = np.load(join(ifolder, 'freqs.npy'))    
    tvec = np.load(join(ifolder, 'tvec.npy'))
    xmid = np.load(join(ifolder, 'xmid.npy' ))
    ymid = np.load(join(ifolder, 'ymid.npy' ))
    zmid = np.load(join(ifolder, 'zmid.npy' ))
    xstart = np.load(join(ifolder, 'xstart.npy' ))
    ystart = np.load(join(ifolder, 'ystart.npy' ))
    zstart = np.load(join(ifolder, 'zstart.npy' ))
    xend = np.load(join(ifolder, 'xend.npy' ))
    yend = np.load(join(ifolder, 'yend.npy' ))
    zend = np.load(join(ifolder, 'zend.npy' ))    
    diam = np.load(join(ifolder, 'diam.npy'))
    
    # Initializing figure
    plt.close('all')    
    
    fig = plt.figure(figsize=[8,8])
    fig.suptitle("Model: %s, input_idx: %d, input_scaling: %g "%(ifolder, input_idx, input_scaling))

    plt.subplots_adjust(hspace=0.5)
    ax_neur = fig.add_axes([0.3, 0.1, 0.3, 0.8], frameon=False, aspect='equal', xticks=[], yticks=[])

    #Sorting compartemts after y-height
    argsort = np.argsort([ymid[comp] for comp in plot_compartments])
    plot_compartments = np.array(plot_compartments)[argsort[:]]
    comp_clr_list = []
    for idx in xrange(len(plot_compartments)):
        i = 256. * idx
        if len(plot_compartments) > 1:
            i /= len(plot_compartments) - 1.
        else:
            i /= len(plot_compartments)
        comp_clr_list.append(plt.cm.rainbow(int(i)))
    ymin = plot_params['ymin']
    ymax = plot_params['ymax']
    yticks = np.arange(ymin, ymax + 1 , 250)
    yticks = yticks[np.where((np.min(ymid) <= yticks) * (yticks <= np.max(ymid)))]
    ext_ax_width = 0.2
    ext_ax_height = 0.8/len(plot_compartments)

    reduced_clr = 'r' 
    act_clr = 'k'
    pas_clr = 'gray'
       
    for comp in xrange(len(xmid)):
        ax_neur.plot([xstart[comp], xend[comp]], [ystart[comp], yend[comp]], lw=diam[comp], color='gray')
    ax_neur.axis([-200, 200, ymin -50, ymax + 50])
    ax_neur.plot(xmid[input_idx], ymid[input_idx], '*', color='y', label='Input', markersize=15)
    ax_neur.legend(bbox_to_anchor=[1.1, 1.05], numpoints=1)
    for numb, comp in enumerate(plot_compartments):
        ax_neur.plot(xmid[comp], ymid[comp], 'o', color=comp_clr_list[numb])
        if numb % 2:
            x_pos = 0.2
        else:
            x_pos = 0.6
        ax_temp = fig.add_axes([x_pos, 0.1 + numb*(ext_ax_height+0.05)/2, 
                                ext_ax_width, ext_ax_height])
        ax_temp.tick_params(color=comp_clr_list[numb])
        for spine in ax_temp.spines.values():
            spine.set_edgecolor(comp_clr_list[numb])
        #ax_temp.set_xticklabels([])
        #ax_temp.set_yticklabels([])
        ax_temp.grid(True)        
        ## for cur_numb, cur in enumerate(active_dict):
        ##     #print active_dict[cur][idx]
        ##     #set_trace()
        ##     #if any(active_dict[cur][idx,:] >= 0):
        ##          #if active_dict[cur][idx,:]
        ##          #set_trace()
        ##     print cur, active_dict[cur][comp,:]
        ##     if 1:
        ##         ax_temp.plot(freqs, active_dict[cur][comp,:], label=cur, color=clr_list[cur_numb], lw=2)
        ##         ax_temp.plot(1.5, active_dict[cur][comp,0], '+', color=clr_list[cur_numb])
        ##     else:
        ##         ax_temp.plot(freqs, active_dict[cur][comp,:]/active_dict[cur][input_idx,:], label=cur, color=clr_list[cur_numb], lw=2)
        ##         ax_temp.plot(1.5, active_dict[cur][comp,0]/active_dict[cur][input_idx,0], '+', color=clr_list[cur_numb])
        ##     #except ValueError:
        ##     #    print "Skipping ", cur
        ##     #     pass
        if 1:
            sim_type = 'current'
            #ax_temp.plot(freqs, ipas[comp,:], label='Ipas', color='y', lw=2)
            #ax_temp.plot(1.5, ipas[comp,0], '+', color='y')
            #ax_temp.plot(freqs, icap[comp,:], '--', label='Icap', color='grey', lw=2)
            #ax_temp.plot(1.5, icap[comp,0], '+', color='grey')
            ax_temp.plot(freqs, imem_passive[comp], color=pas_clr, lw=1, label='Passive')      
            ax_temp.plot(1.5, imem_passive[comp,0], '+', color=pas_clr)
            ax_temp.plot(freqs, imem_reduced[comp], color=reduced_clr, lw=1, label='Reduced')      
            ax_temp.plot(1.5, imem_reduced[comp,0], '+', color=reduced_clr)
            
            ax_temp.plot(freqs, imem_active[comp], color=act_clr, lw=1, label='active')      
            ax_temp.plot(1.5, imem_active[comp,0], '+', color=act_clr)
            ax_temp.set_ylim(1e-7, 1e1)
        else:
            sim_type = 'transfer'
            #ax_temp.plot(freqs, ipas[comp,:] / ipas[input_idx,:], label='Ipas', color='y', lw=2)
            #ax_temp.plot(1.5, ipas[comp,0]/ ipas[input_idx,0], '+', color='y')
            #ax_temp.plot(freqs, icap[comp,:]/ icap[input_idx,:], '--', label='Icap', color='grey', lw=2)
            #ax_temp.plot(1.5, icap[comp,0]/ icap[input_idx,0], '+', color='grey')
            ax_temp.plot(freqs, imem_passive[comp]/ imem_passive[input_idx,:], color=pas_clr, lw=1, label='Passive')      
            ax_temp.plot(1.5, imem_passive[comp,0]/ imem_passive[input_idx,0], '+', color=pas_clr)
            ax_temp.plot(freqs, imem_reduced[comp]/ imem_reduced[input_idx,:], color=reduced_clr, lw=1, label='Reduced')      
            ax_temp.plot(1.5, imem_reduced[comp,0]/ imem_reduced[input_idx,0], '+', color=reduced_clr)            
            ax_temp.plot(freqs, imem_active[comp]/ imem_active[input_idx,:], color=act_clr, lw=1, label='Active')      
            ax_temp.plot(1.5, imem_active[comp,0]/ imem_active[input_idx,0], '+', color=act_clr)
            ax_temp.set_ylim(1e-5, 1e1)
            
        pos = [xmid[comp], ymid[comp]]
        if numb == 0:
            ax_temp.legend(bbox_to_anchor=[1.8, 1.22])
            ax_temp.set_xlabel('Hz')
            ax_temp.set_ylabel('Transfer factor')
            #ax_temp.axis(ax_temp.axis('tight'))
        ax_temp.set_xlim(1,1000)
        
        ax_temp.set_xscale('log')
        ax_temp.set_yscale('log')
        arrow_to_axis(pos, ax_neur, ax_temp, comp_clr_list[numb], x_pos)
    plt.savefig('imem_%s_%s_%s.png' % (ifolder, cur_name, sim_type), dpi=300)

def plot_synaptic_currents(ifolder, input_scaling, input_idx, plot_params, simulation_params,
                            plot_compartments, epas=None):


    conductance_type_list = ['active', 'passive', 'reduced_Ih', 'Ih_linearized']

    line_style = ['r', 'k', 'b', 'g', 'y', 'm', 'grey']
    if epas == None:
        sim_name = '%d_%1.3f' %(input_idx, input_scaling)
    else:
        sim_name = '%d_%1.3f_%d' %(input_idx, input_scaling, epas)
    
    imem_dict = {}
    for conductance_type in conductance_type_list:
        if epas == None:
            imem_name = '%d_%1.3f_%s' %(input_idx, input_scaling, conductance_type)
        else:
            imem_name = '%d_%1.3f_%s_%d' %(input_idx, input_scaling, conductance_type, epas)
        imem_dict[conductance_type] = np.load(join(ifolder, 'imem_%s.npy' % imem_name))
    
    freqs = np.load(join(ifolder, 'freqs.npy'))    
    tvec = np.load(join(ifolder, 'tvec.npy'))
    xmid = np.load(join(ifolder, 'xmid.npy' ))
    ymid = np.load(join(ifolder, 'ymid.npy' ))
    zmid = np.load(join(ifolder, 'zmid.npy' ))
    xstart = np.load(join(ifolder, 'xstart.npy' ))
    ystart = np.load(join(ifolder, 'ystart.npy' ))
    zstart = np.load(join(ifolder, 'zstart.npy' ))
    xend = np.load(join(ifolder, 'xend.npy' ))
    yend = np.load(join(ifolder, 'yend.npy' ))
    zend = np.load(join(ifolder, 'zend.npy' ))    
    diam = np.load(join(ifolder, 'diam.npy'))

    #Sorting compartemts after y-height
    argsort = np.argsort([ymid[comp] for comp in plot_compartments])
    plot_compartments = np.array(plot_compartments)[argsort[:]]
    comp_clr_list = []
    for idx in xrange(len(plot_compartments)):
        i = 256. * idx
        if len(plot_compartments) > 1:
            i /= len(plot_compartments) - 1.
        else:
            i /= len(plot_compartments)
        comp_clr_list.append(plt.cm.rainbow(int(i)))
    ymin = plot_params['ymin']
    ymax = plot_params['ymax']
    yticks = np.arange(ymin, ymax + 1 , 250)
    yticks = yticks[np.where((np.min(ymid) <= yticks) * (yticks <= np.max(ymid)))]
    ext_ax_width = 0.2
    ext_ax_height = 0.8/len(plot_compartments)

    reduced_clr = 'r' 
    act_clr = 'k'
    pas_clr = 'gray'

    # Initializing time-axis figure
    plt.close('all')    
    fig = plt.figure(figsize=[8,8])
    fig.suptitle("Model: %s, input_idx: %d, input_scaling: %g, epas: %d"
                 %(ifolder, input_idx, input_scaling, epas))

    plt.subplots_adjust(hspace=0.5)
    ax_neur = fig.add_axes([0.3, 0.1, 0.3, 0.8], frameon=False, aspect='equal', xticks=[], yticks=[])    
    for comp in xrange(len(xmid)):
        ax_neur.plot([xstart[comp], xend[comp]], [ystart[comp], yend[comp]], lw=diam[comp], color='gray')
    ax_neur.axis([-200, 200, ymin -50, ymax + 50])
    ax_neur.plot(xmid[input_idx], ymid[input_idx], '*', color='y', label='Input', markersize=15)
    ax_neur.legend(bbox_to_anchor=[1.1, 1.05], numpoints=1)
    for numb, comp in enumerate(plot_compartments):
        ax_neur.plot(xmid[comp], ymid[comp], 'o', color=comp_clr_list[numb])
        if numb % 2:
            x_pos = 0.1
        else:
            x_pos = 0.6
        ax_temp = fig.add_axes([x_pos, 0.1 + numb*(ext_ax_height+0.05)/2, 
                                ext_ax_width, ext_ax_height])
        ax_temp.tick_params(color=comp_clr_list[numb])
        for spine in ax_temp.spines.values():
            spine.set_edgecolor(comp_clr_list[numb])
        ax_temp.grid(True)        

        for line_numb, conductance_type in enumerate(conductance_type_list):
            lw = 0.5 + 0.5 / ((line_numb + 1.) / len(conductance_type_list))
            ax_temp.plot(tvec, imem_dict[conductance_type][comp], 
                     line_style[line_numb], lw=lw, label=conductance_type)      

        pos = [xmid[comp], ymid[comp]]
        if numb == 0:
            ax_temp.legend(bbox_to_anchor=[1.8, 1.22])
            ax_temp.set_xlabel('[ms]')
            ax_temp.set_ylabel('Transmemb. current')
            #ax_temp.axis(ax_temp.axis('tight'))
            #ax_temp.set_xlim(1,1000)
        
        #ax_temp.set_xscale('log')
        #ax_temp.set_yscale('log')
        arrow_to_axis(pos, ax_neur, ax_temp, comp_clr_list[numb], x_pos)
    plt.savefig('K_check_%s_%s.png' % (ifolder, sim_name), dpi=300)
    #plt.show()

    
    ## # Initializing frequency-axis figure
    ## plt.close('all')    
    ## fig = plt.figure(figsize=[8,8])
    ## fig.suptitle("Model: %s, input_idx: %d, input_scaling: %g, epas: %d "
    ##              % (ifolder, input_idx, input_scaling, epas))

    ## plt.subplots_adjust(hspace=0.5)
    ## ax_neur = fig.add_axes([0.3, 0.1, 0.3, 0.8], frameon=False, aspect='equal', xticks=[], yticks=[])    
    ## for comp in xrange(len(xmid)):
    ##     ax_neur.plot([xstart[comp], xend[comp]], [ystart[comp], yend[comp]], lw=diam[comp], color='gray')
    ## ax_neur.axis([-200, 200, ymin -50, ymax + 50])
    ## ax_neur.plot(xmid[input_idx], ymid[input_idx], '*', color='y', label='Input', markersize=15)
    ## ax_neur.legend(bbox_to_anchor=[1.1, 1.05], numpoints=1)
    ## for numb, comp in enumerate(plot_compartments):
    ##     ax_neur.plot(xmid[comp], ymid[comp], 'o', color=comp_clr_list[numb])
    ##     if numb % 2:
    ##         x_pos = 0.1
    ##     else:
    ##         x_pos = 0.6
    ##     ax_temp = fig.add_axes([x_pos, 0.1 + numb*(ext_ax_height+0.05)/2, 
    ##                             ext_ax_width, ext_ax_height])
    ##     ax_temp.tick_params(color=comp_clr_list[numb])
    ##     for spine in ax_temp.spines.values():
    ##         spine.set_edgecolor(comp_clr_list[numb])
    ##     ax_temp.grid(True)        

    ##     #ax_temp.plot(freqs, ipas[comp,:], label='Ipas', color='y', lw=2)
    ##     #ax_temp.plot(1.5, ipas[comp,0], '+', color='y')
    ##     #ax_temp.plot(freqs, icap[comp,:], '--', label='Icap', color='grey', lw=2)
    ##     #ax_temp.plot(1.5, icap[comp,0], '+', color='grey')
    ##     #set_trace()
    ##     ax_temp.plot(freqs, imem_psd_passive[comp], color=pas_clr, lw=2, label='Passive')      
    ##     #ax_temp.plot(1.5, imem_passive[comp,0], '+', color=pas_clr)
    ##     ax_temp.plot(freqs, imem_psd_reduced[comp], color=reduced_clr, lw=1, label='Reduced')      
    ##     #ax_temp.plot(1.5, imem_reduced[comp,0], '+', color=reduced_clr)
    ##     ax_temp.plot(freqs, imem_psd_active[comp], '--', color=act_clr, lw=1, label='active')      
    ##     #ax_temp.plot(1.5, imem_active[comp,0], '+', color=act_clr)
    ##     ax_temp.set_ylim(1e-9, 1e-1)
            
    ##     pos = [xmid[comp], ymid[comp]]
    ##     if numb == 0:
    ##         ax_temp.legend(bbox_to_anchor=[1.8, 1.22])
    ##         ax_temp.set_xlabel('Hz')
    ##         ax_temp.set_ylabel('Transmemb. current')
    ##         #ax_temp.axis(ax_temp.axis('tight'))
    ##         #ax_temp.set_xlim(1,1000)
        
    ##     ax_temp.set_xscale('log')
    ##     ax_temp.set_yscale('log')
    ##     arrow_to_axis(pos, ax_neur, ax_temp, comp_clr_list[numb], x_pos)
    ## plt.savefig('synaptic_%s_%s_psd.png' % (ifolder, sim_name), dpi=300)



    
def plot_active_currents_deprecated(ifolder, input_scaling, input_idx, simulation_params, conductance_type):

    #name = "%d_%1.3f" %(input_idx, input_scaling)
    name = "psd_%d_%1.3f_%s" %(input_idx, input_scaling, conductance_type)
    freqs = np.load(join(ifolder, 'freqs.npy'))
    
    tvec = np.load(join(ifolder, 'tvec.npy'))
    imem = 1000 * np.load(join(ifolder, 'imem_%s.npy' %(name)))
    icap = 1000 * np.load(join(ifolder, 'icap_%s.npy' %(name)))
    ipas = 1000 * np.load(join(ifolder, 'ipas_%s.npy' %(name)))
    active_dict = {}

    clr_list = ['r', 'b', 'g', 'm']
    
    for cur in simulation_params['rec_variables']:
        try:
            active_dict[cur] = 1000 * np.load(join(ifolder, '%s_%s.npy'%(cur, name)))
        except:
            pass
    sim_name = '%d_%1.3f' %(input_idx, input_scaling)
    plt.close('all')
    n_cols = 6
    n_rows = 6
    fig = plt.figure(figsize=[15,10])
    if n_cols * n_rows < imem.shape[0]:
        n_plots = n_cols * n_rows
        plots = np.array(np.linspace(0, imem.shape[0] - 1, n_plots), dtype=int)
    else:
        n_plots = imem.shape[0]
        plots = xrange(n_plots)
    
    for plot_number, idx in enumerate(plots):
        ax = fig.add_subplot(n_rows, n_cols, plot_number + 1)
        ax.grid(True)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(1e0,1e3)
        ax.set_ylim(1e-7, 1e3)
        for cur_numb, cur in enumerate(active_dict):
            ax.plot(freqs[1:], active_dict[cur][idx,1:], label=cur, color=clr_list[cur_numb], lw=2)
            ax.plot(1.5, active_dict[cur][idx,0], '+', color=clr_list[cur_numb])
            
        ax.plot(freqs, ipas[idx,:], label='Ipas', color='y', lw=2)
        ax.plot(1.5, ipas[idx,0], '+', color='y')
        ax.plot(freqs, icap[idx,:], label='Icap', color='grey', lw=2)
        ax.plot(1.5, icap[idx,0], '+', color='grey')
        ax.plot(freqs, imem[idx,:], label='Imem', color='k', lw=1)
        ax.plot(1.5, imem[idx,0], '+', color='k')        
        
        if plot_number == n_rows - 1:
            ax.legend(bbox_to_anchor=[1.7,1.0])
    plt.savefig('active_currents_%s_%s_%s_psd.png' % (ifolder, sim_name, conductance_type), dpi=150)
    #plt.savefig('active_currents_%s_%s_True_psd.pdf' % (ifolder, sim_name))
    
    
def compare_active_passive(ifolder, input_scaling, input_idx, elec_x, elec_y, elec_z, plot_params):

    name = "%d_%1.3f" %(input_idx, input_scaling)
    name_active = "%d_%1.3f_%s" %(input_idx, input_scaling, 'active')
    name_passive = "%d_%1.3f_%s" %(input_idx, input_scaling, 'passive')
    name_reduced = "%d_%1.3f_%s" %(input_idx, input_scaling, 'reduced')

    # Loading all needed data
    imem_active = np.load(join(ifolder, 'imem_%s.npy' %(name_active)))
    imem_psd_active = np.load(join(ifolder, 'imem_psd_%s.npy' %(name_active)))
    somav_psd_active = np.load(join(ifolder, 'somav_psd_%s.npy' %(name_active)))
    somav_active = np.load(join(ifolder, 'somav_%s.npy' %(name_active)))
    sig_active = np.load(join(ifolder, 'signal_%s.npy' %(name_active)))
    psd_active = np.load(join(ifolder, 'psd_%s.npy' %(name_active)))
    stick_active = np.load(join(ifolder, 'stick_%s.npy' %(name_active)))
    stick_psd_active = np.load(join(ifolder, 'stick_psd_%s.npy' %(name_active)))

    imem_reduced = np.load(join(ifolder, 'imem_%s.npy' %(name_reduced)))
    imem_psd_reduced = np.load(join(ifolder, 'imem_psd_%s.npy' %(name_reduced)))
    somav_psd_reduced = np.load(join(ifolder, 'somav_psd_%s.npy' %(name_reduced)))
    somav_reduced = np.load(join(ifolder, 'somav_%s.npy' %(name_reduced)))
    sig_reduced = np.load(join(ifolder, 'signal_%s.npy' %(name_reduced)))
    psd_reduced = np.load(join(ifolder, 'psd_%s.npy' %(name_reduced)))
    stick_reduced = np.load(join(ifolder, 'stick_%s.npy' %(name_reduced)))
    stick_psd_reduced = np.load(join(ifolder, 'stick_psd_%s.npy' %(name_reduced)))
    
    imem_passive = np.load(join(ifolder, 'imem_%s.npy' %(name_passive)))
    imem_psd_passive = np.load(join(ifolder, 'imem_psd_%s.npy' %(name_passive)))
    somav_psd_passive = np.load(join(ifolder, 'somav_psd_%s.npy' %(name_passive)))    
    somav_passive = np.load(join(ifolder, 'somav_%s.npy' %(name_passive)))
    sig_passive = np.load(join(ifolder, 'signal_%s.npy' %(name_passive)))
    psd_passive = np.load(join(ifolder, 'psd_%s.npy' %(name_passive)))
    stick_passive = np.load(join(ifolder, 'stick_%s.npy' %(name_passive)))
    stick_psd_passive = np.load(join(ifolder, 'stick_psd_%s.npy' %(name_passive)))
    
    freqs = np.load(join(ifolder, 'freqs.npy'))    
    tvec = np.load(join(ifolder, 'tvec.npy'))
    input_array = input_scaling * np.load(join(ifolder, 'input_array.npy'))[-len(tvec):]
    input_array_psd = input_scaling * np.load(join(ifolder, 'input_array_psd.npy'))
    xmid = np.load(join(ifolder, 'xmid.npy' ))
    ymid = np.load(join(ifolder, 'ymid.npy' ))
    zmid = np.load(join(ifolder, 'zmid.npy' ))
    xstart = np.load(join(ifolder, 'xstart.npy' ))
    ystart = np.load(join(ifolder, 'ystart.npy' ))
    zstart = np.load(join(ifolder, 'zstart.npy' ))
    xend = np.load(join(ifolder, 'xend.npy' ))
    yend = np.load(join(ifolder, 'yend.npy' ))
    zend = np.load(join(ifolder, 'zend.npy' ))    
    diam = np.load(join(ifolder, 'diam.npy'))

    # Initializing figure
    plt.close('all')    
    fig = plt.figure(figsize=[14,8])
    fig.suptitle("Model: %s, Input scaling: %s, Input index: %s"
                 %(ifolder, input_scaling, input_idx))

    plt.subplots_adjust(hspace=0.5)
    ax_in = fig.add_axes([0.05, 0.1, 0.10, 0.20], title='Input')
    ax_im = fig.add_axes([0.05, 0.4, 0.1, 0.2], title='Soma $I_m$')
    ax_vm = fig.add_axes([0.05, 0.70, 0.10, 0.20], title='Soma $V_m$')    
    ax_act_imshow = fig.add_axes([0.73, 0.72, 0.25, 0.13], title='Active')
    ax_pas_imshow = fig.add_axes([0.73, 0.52, 0.25, 0.13], title='Passive')
    ax_im_psd = fig.add_axes([0.18, 0.4, 0.1, 0.2], title='Soma $I_m$ PSD')
    ax_in_psd = fig.add_axes([0.18, 0.1, 0.10, 0.20], title='Input PSD')    
    ax_vm_psd = fig.add_axes([0.18, 0.70, 0.10, 0.20], title='Soma $V_m$ PSD')    
    ax_act_psd_imshow = fig.add_axes([0.73, 0.32, 0.25, 0.13], title='Active PSD', xscale='log')
    ax_pas_psd_imshow = fig.add_axes([0.73, 0.1, 0.25, 0.13], title='Passive PSD', xscale='log')    
    ax_neur = fig.add_axes([0.25, 0.1, 0.35, 0.75], frameon=False, aspect='equal', xticks=[])

    plt.figtext(0.74, 0.9, 'Sum of transmembrane \n currents along y-axis', size=15)
    ax_vm.set_xlabel('ms')
    ax_vm.set_ylabel('mV')
    ax_vm_psd.set_xlabel('Hz')
    ax_im.set_xlabel('ms')
    ax_im.set_ylabel('pA')
    ax_im_psd.set_xlabel('Hz')
    ax_in.set_xlabel('ms')
    ax_in.set_ylabel('pA')
    ax_in.set_xlabel('ms')    
    ax_in_psd.set_xlabel('Hz')
    ax_pas_psd_imshow.set_xlabel('Hz')
    ax_act_psd_imshow.set_xlabel('Hz')
    ax_pas_imshow.set_xlabel('ms')
    ax_act_imshow.set_xlabel('ms')
    ax_neur.set_ylabel('y [$\mu m$]', color='b')
    ax_act_imshow.set_ylabel('y [$\mu m$]', color='b')
    ax_pas_imshow.set_ylabel('y [$\mu m$]', color='b')
    ax_act_psd_imshow.set_ylabel('y [$\mu m$]', color='b')
    ax_pas_psd_imshow.set_ylabel('y [$\mu m$]', color='b')

    ax_im_psd.grid(True)
    ax_vm_psd.grid(True)
    ax_in_psd.grid(True)

    ax_vm_psd.set_xlim(1,1000)
    ax_vm_psd.set_ylim(1e-5,1e2)
    ax_im_psd.set_xlim(1,1000)
    ax_im_psd.set_ylim(1e-5,1e2)
    ax_in_psd.set_xlim(1,1000)
    ax_in_psd.set_ylim(1e-5, 1e2)
    ax_pas_psd_imshow.set_xlim(1e0, 1e3)
    ax_act_psd_imshow.set_xlim(1e0, 1e3)

    # Setting up helpers and colors
    act_clr = 'k'
    pas_clr = 'grey'
    reduced_clr = 'g'
    elec_clr_list = []
    for idx in xrange(len(elec_x)):
        i = 256. * idx
        if len(elec_x) > 1:
            i /= len(elec_x) - 1.
        else:
            i /= len(elec_x)
        elec_clr_list.append(plt.cm.rainbow(int(i)))
    ymin = plot_params['ymin']
    ymax = plot_params['ymax']
    yticks = np.arange(ymin, ymax + 1 , 250)
    yticks = yticks[np.where((np.min(ymid) <= yticks) * (yticks <= np.max(ymid)))]
    for ax in [ax_neur, ax_act_psd_imshow, ax_pas_psd_imshow, ax_pas_imshow, ax_act_imshow]:
        ax.get_yaxis().tick_left()
        ax.set_yticks(yticks)
        for tl in ax.get_yticklabels():
            tl.set_color('b')
    
    # Starting plotting
    ax_vm.plot(tvec, somav_active, color=act_clr)
    ax_vm.plot(tvec, somav_passive, color=pas_clr)
    ax_vm.plot(tvec, somav_reduced, color=reduced_clr)
    ax_vm_psd.loglog(freqs, somav_psd_active, act_clr, lw=2, label='Active')
    ax_vm_psd.loglog(freqs, somav_psd_reduced, reduced_clr, lw=2, label='Reduced')
    ax_vm_psd.loglog(freqs, somav_psd_passive, pas_clr, lw=2, label='Passive')
    ax_vm_psd.legend(bbox_to_anchor=[1.4,1.0])

    ax_im.plot(tvec, 1000*imem_active[0,:], act_clr)
    ax_im.plot(tvec, 1000*imem_passive[0,:], pas_clr)
    ax_im.plot(tvec, 1000*imem_reduced[0,:], reduced_clr)
    
    ax_im_psd.loglog(freqs, 1000*imem_psd_active[0,:], act_clr, lw=2)
    ax_im_psd.loglog(freqs, 1000*imem_psd_passive[0,:], pas_clr, lw=2)
    ax_im_psd.loglog(freqs, 1000*imem_psd_reduced[0,:], reduced_clr, lw=2)

    ax_in.plot(tvec, 1000*input_array, 'k')
    ax_in_psd.loglog(freqs, 1000*input_array_psd, 'k', lw=2)

    for comp in xrange(len(xmid)):
        ax_neur.plot([xstart[comp], xend[comp]], [ystart[comp], yend[comp]], lw=diam[comp], color='gray')
    ax_neur.plot(xmid[input_idx], ymid[input_idx], '*', color='y', label='Input', markersize=15)
    ax_neur.legend(bbox_to_anchor=[.6, 1.05], numpoints=1)
    ax_neur.axis([-200, 200, np.min([np.min(elec_y) - 50, ymin]) , np.max([np.max(elec_y) + 50, ymax])])
    
    ext_ax_width=0.1
    ext_ax_height = 0.6/len(elec_x)
    for elec in xrange(len(elec_x)):
        ax_neur.plot(elec_x[elec], elec_y[elec], 'o', color=elec_clr_list[elec])
        ax_temp = fig.add_axes([0.55, 0.1 + elec*(ext_ax_height+0.05), 
                                ext_ax_width, ext_ax_height])
        ax_temp.tick_params(color=elec_clr_list[elec])
        for spine in ax_temp.spines.values():
            spine.set_edgecolor(elec_clr_list[elec])

        ax_temp.set_xticklabels([])
        ax_temp.set_yticklabels([])
        ax_temp.grid(True)
        ax_temp.loglog(freqs, psd_active[elec]/np.max(psd_active[elec]), 
                       color=act_clr, lw=2)
        ax_temp.loglog(freqs, psd_passive[elec]/np.max(psd_passive[elec]), 
                       color=pas_clr, lw=2)
        
        ax_temp.loglog(freqs, psd_reduced[elec]/np.max(psd_reduced[elec]), 
                       color=reduced_clr, lw=2)        
        pos = [elec_x[elec], elec_y[elec]]
        
        if elec == 0:
            #ax_temp.legend(bbox_to_anchor=[1.4, 1.22])
            ax_temp.set_xlabel('Hz')
            ax_temp.set_ylabel('Norm. amp')
        ax_temp.set_xlim(1,1000)
        ax_temp.set_ylim(1e-2, 2e0)
        arrow_to_axis(pos, ax_neur, ax_temp, elec_clr_list[elec])

    ## # COLOR PLOT OF y-AXIS SUMMED IMEM CONTRIBUTION
    ## stick_pos = np.linspace(np.max(ymid), np.min(ymid), stick_passive.shape[0])
    ## X, Y = np.meshgrid(freqs, stick_pos)
    
    ## sc_stick_pas = ax_pas_imshow.pcolormesh(X, Y, 1000*stick_passive, cmap='jet_r',
    ##                                         vmax=np.max(np.abs(1000*stick_passive)), 
    ##                                         vmin=-np.max(np.abs(1000*stick_passive)))

    ## sc_stick_act = ax_act_imshow.pcolormesh(X, Y, 1000*stick_active, cmap='jet_r', 
    ##                                         vmax=np.max(np.abs(1000*stick_active)), 
    ##                                         vmin=-np.max(np.abs(1000*stick_active)))
    
    ## sc_stick_psd_pas = ax_pas_psd_imshow.pcolormesh(X, Y, 1000*stick_psd_passive, cmap='jet',
    ##         norm=LogNorm(vmax=np.max(1000*stick_psd_passive), vmin=1e-4))
    ## sc_stick_psd_act = ax_act_psd_imshow.pcolormesh(X, Y, 1000*stick_psd_active, cmap='jet',
    ##                                  norm=LogNorm(vmax=np.max(1000*stick_psd_active), vmin=1e-4))

    ## ax_pas_imshow.axis('auto')
    ## ax_act_imshow.axis('auto')
    ## plt.colorbar(sc_stick_act, ax=ax_act_imshow)
    ## plt.colorbar(sc_stick_pas, ax=ax_pas_imshow)
    ## plt.colorbar(sc_stick_psd_pas, ax=ax_pas_psd_imshow)
    ## plt.colorbar(sc_stick_psd_act, ax=ax_act_psd_imshow)

    ## xmin, xmax = ax_neur.get_xaxis().get_view_interval()
    ## ax_neur.add_artist(plt.Line2D((xmin, xmin), (np.min(ymid), np.max(ymid)), color='b', linewidth=3))
    
    plt.savefig('WN_%s_%s_reduced.png' % (ifolder, name))





    
def stationary_currents(ifolder, plot_params, plot_compartments, name):


    # Loading all needed data
    cur_name = '0_0.000_' + name
    imem = np.load(join(ifolder, 'imem_%s.npy' %(cur_name)))
    imem_psd = np.load(join(ifolder, 'imem_psd_%s.npy' %(cur_name)))
    somav_psd = np.load(join(ifolder, 'somav_psd_%s.npy' %(cur_name)))
    somav = np.load(join(ifolder, 'somav_%s.npy' %(cur_name)))
    stick = np.load(join(ifolder, 'stick_%s.npy' %(cur_name)))
    stick_psd = np.load(join(ifolder, 'stick_psd_%s.npy' %(cur_name)))

    
    freqs = np.load(join(ifolder, 'freqs.npy'))    
    tvec = np.load(join(ifolder, 'tvec.npy'))
    xmid = np.load(join(ifolder, 'xmid.npy' ))
    ymid = np.load(join(ifolder, 'ymid.npy' ))
    zmid = np.load(join(ifolder, 'zmid.npy' ))
    xstart = np.load(join(ifolder, 'xstart.npy' ))
    ystart = np.load(join(ifolder, 'ystart.npy' ))
    zstart = np.load(join(ifolder, 'zstart.npy' ))
    xend = np.load(join(ifolder, 'xend.npy' ))
    yend = np.load(join(ifolder, 'yend.npy' ))
    zend = np.load(join(ifolder, 'zend.npy' ))    
    diam = np.load(join(ifolder, 'diam.npy'))
    
    # Initializing figure
    plt.close('all')    
    fig = plt.figure(figsize=[14,8])
    fig.suptitle("Model: %s"%(ifolder))

    plt.subplots_adjust(hspace=0.5)

    ax_im = fig.add_axes([0.05, 0.4, 0.1, 0.2], title='Soma $I_m$')
    ax_vm = fig.add_axes([0.05, 0.70, 0.10, 0.20], title='Soma $V_m$')    
    ax_imshow = fig.add_axes([0.73, 0.72, 0.25, 0.13])
    ax_im_psd = fig.add_axes([0.18, 0.4, 0.1, 0.2], title='Soma $I_m$ PSD')

    ax_vm_psd = fig.add_axes([0.18, 0.70, 0.10, 0.20], title='Soma $V_m$ PSD')    
    ax_psd_imshow = fig.add_axes([0.73, 0.32, 0.25, 0.13], title='PSD', xscale='log')
    ax_neur = fig.add_axes([0.25, 0.1, 0.35, 0.75], frameon=False, aspect='equal', xticks=[])

    plt.figtext(0.74, 0.9, 'Sum of transmembrane \n currents along y-axis', size=15)
    ax_vm.set_xlabel('ms')
    ax_vm.set_ylabel('mV')
    ax_vm_psd.set_xlabel('Hz')
    ax_im.set_xlabel('ms')
    ax_im.set_ylabel('pA')
    ax_im_psd.set_xlabel('Hz')

    ax_psd_imshow.set_xlabel('Hz')
    ax_imshow.set_xlabel('ms')
    ax_neur.set_ylabel('y [$\mu m$]', color='b')
    ax_imshow.set_ylabel('y [$\mu m$]', color='b')
    ax_psd_imshow.set_ylabel('y [$\mu m$]', color='b')

    ax_im_psd.grid(True)
    ax_vm_psd.grid(True)

    ax_vm_psd.set_xlim(1,1000)
    ax_vm_psd.set_ylim(1e-5,1e2)
    ax_im_psd.set_xlim(1,1000)
    ax_im_psd.set_ylim(1e-5,1e2)
    ax_psd_imshow.set_xlim(1e0, 1e3)

    # Setting up helpers and colors
    act_clr = 'k'

    #Sorting compartemts after y-height
    argsort = np.argsort([ymid[comp] for comp in plot_compartments])
    plot_compartments = np.array(plot_compartments)[argsort[:]]
    comp_clr_list = []
    for idx in xrange(len(plot_compartments)):
        i = 256. * idx
        if len(plot_compartments) > 1:
            i /= len(plot_compartments) - 1.
        else:
            i /= len(plot_compartments)
        comp_clr_list.append(plt.cm.rainbow(int(i)))
    ymin = plot_params['ymin']
    ymax = plot_params['ymax']
    yticks = np.arange(ymin, ymax + 1 , 250)
    yticks = yticks[np.where((np.min(ymid) <= yticks) * (yticks <= np.max(ymid)))]
    for ax in [ax_neur, ax_psd_imshow, ax_imshow]:
        ax.get_yaxis().tick_left()
        ax.set_yticks(yticks)
        for tl in ax.get_yticklabels():
            tl.set_color('b')
    
    # Starting plotting
    ax_vm.plot(tvec, somav, color=act_clr)
    ax_vm_psd.loglog(freqs, somav_psd, act_clr, lw=2)


    ax_im.plot(tvec, 1000*imem[0,:], act_clr)
    
    ax_im_psd.loglog(freqs, 1000*imem_psd[0,:], act_clr, lw=2)
    ax_im.plot(1.5, imem_psd[0,0], '+', color=act_clr)
    
    for comp in xrange(len(xmid)):
        ax_neur.plot([xstart[comp], xend[comp]], [ystart[comp], yend[comp]], lw=diam[comp], color='gray')
    ax_neur.axis([-200, 200, ymin -50, ymax + 50])
    
    ext_ax_width=0.1
    ext_ax_height = 0.6/len(plot_compartments)
    for numb, comp in enumerate(plot_compartments):
        ax_neur.plot(xmid[comp], ymid[comp], 'o', color=comp_clr_list[numb])
        ax_temp = fig.add_axes([0.55, 0.1 + numb*(ext_ax_height+0.05), 
                                ext_ax_width, ext_ax_height])
        ax_temp.tick_params(color=comp_clr_list[numb])
        for spine in ax_temp.spines.values():
            spine.set_edgecolor(comp_clr_list[numb])

        ax_temp.set_xticklabels([])
        ax_temp.set_yticklabels([])
        ax_temp.grid(True)
        ax_temp.loglog(freqs, imem_psd[comp], 
                       color=act_clr, lw=2)      
        ax_temp.plot(1.5, imem_psd[comp,0], '+', color=act_clr)
        pos = [xmid[comp], ymid[comp]]
        if numb == 0:
            #ax_temp.legend(bbox_to_anchor=[1.4, 1.22])
            ax_temp.set_xlabel('Hz')
            ax_temp.set_ylabel('Norm. amp')
            ax_temp.axis(ax_temp.axis('tight'))
            #ax_temp.set_xlim(1,1000)
        #ax_temp.set_ylim(1e-5, 1e0)
        arrow_to_axis(pos, ax_neur, ax_temp, comp_clr_list[numb])

    # COLOR PLOT OF y-AXIS SUMMED IMEM CONTRIBUTION
    stick_pos = np.linspace(np.max(ymid), np.min(ymid), stick.shape[0])
    X, Y = np.meshgrid(freqs, stick_pos)
    X_t, Y_t = np.meshgrid(tvec, stick_pos)


    sc_stick = ax_imshow.pcolormesh(X_t, Y_t, 1000*stick, cmap='jet_r', 
                                            vmax=np.max(np.abs(1000*stick)), 
                                            vmin=-np.max(np.abs(1000*stick)))
    
    sc_stick_psd = ax_psd_imshow.pcolormesh(X, Y, 1000*stick_psd, cmap='jet',
                                     norm=LogNorm(vmax=np.max(1000*stick_psd), vmin=1e-4))

    ax_imshow.axis('auto')
    plt.colorbar(sc_stick, ax=ax_imshow)

    plt.colorbar(sc_stick_psd, ax=ax_psd_imshow)

    xmin, xmax = ax_neur.get_xaxis().get_view_interval()
    ax_neur.add_artist(plt.Line2D((xmin, xmin), (np.min(ymid), np.max(ymid)), color='b', linewidth=3))
    
    plt.savefig('%s_%s.png' % (ifolder, name))

def compare_LFPs(ifolder, input_scaling, input_idx, elec_x, elec_y, elec_z, 
                 plot_params, conductance_list, input_type=''):

    if input_type == 'WN':
        freqs = np.load(join(ifolder, 'freqs.npy'))    
    tvec = np.load(join(ifolder, 'tvec_%s.npy' % input_type))
    xmid = np.load(join(ifolder, 'xmid.npy' ))
    ymid = np.load(join(ifolder, 'ymid.npy' ))
    zmid = np.load(join(ifolder, 'zmid.npy' ))
    xstart = np.load(join(ifolder, 'xstart.npy' ))
    ystart = np.load(join(ifolder, 'ystart.npy' ))
    zstart = np.load(join(ifolder, 'zstart.npy' ))
    xend = np.load(join(ifolder, 'xend.npy' ))
    yend = np.load(join(ifolder, 'yend.npy' ))
    zend = np.load(join(ifolder, 'zend.npy' ))    
    diam = np.load(join(ifolder, 'diam.npy'))
    sim_name = '%d_%1.3f_%s' %(input_idx, input_scaling, input_type)
    
    sig_dict = {}
    sig_psd_dict = {}
    #imem_dict = {}
    vmem_dict = {}
    vmem_psd_dict = {}
    conductance_color_dict = {} 
    print sim_name   
    for cond_number, conductance_type in enumerate(conductance_list):
        conductance_name = "%d_%1.3f_%s_%s" %(input_idx, input_scaling, input_type, conductance_type)
        
        #imem_dict[conductance_type] = np.load(join(ifolder, 'imem_%s.npy' %(conductance_name)))
        vmem_dict[conductance_type] = np.load(join(ifolder, 'vmem_%s.npy' %(conductance_name)))
        if input_type == 'WN':
            vmem_psd_dict[conductance_type] = np.load(join(ifolder, 'vmem_psd_%s.npy' %(conductance_name)))
            sig_psd_dict[conductance_type] = np.load(join(ifolder, 'sig_psd_%s.npy' %(conductance_name)))
        sig_dict[conductance_type] = np.load(join(ifolder, 'sig_%s.npy' %(conductance_name)))
        
        if len(conductance_list) > 1:
            clr_number = 256. * cond_number/(len(conductance_list) - 1.)
        else:
            clr_number = 256. * cond_number/(len(conductance_list))
        conductance_color_dict[conductance_type] = plt.cm.jet(int(clr_number))

    # Initializing figure
    plt.close('all')    
    fig = plt.figure(figsize=[14,8])
    fig.suptitle("Model: %s, Input scaling: %s, Input index: %s"
                 %(ifolder, input_scaling, input_idx))

    plt.subplots_adjust(hspace=0.5)

    if input_type == 'ZAP':
        ax_in = fig.add_axes([0.04, 0.3, 0.10, 0.15], title='Input compartment $V_m$')   
        ax_in_shifted = fig.add_axes([0.04, 0.05, 0.10, 0.15], title='Shifted input compartment $V_m$')   

        ax_vm = fig.add_axes([0.04, 0.80, 0.10, 0.15], title='Soma $V_m$')
        ax_vm_shifted = fig.add_axes([0.04, 0.55, 0.10, 0.15], title='Shifted soma $V_m$')    
    
    else:
        ax_vm = fig.add_axes([0.04, 0.80, 0.10, 0.15], title='Soma $V_m$')
        ax_vm_shifted = fig.add_axes([0.04, 0.55, 0.10, 0.15], title='Shifted soma $V_m$')    

        ax_in = fig.add_axes([0.04, 0.3, 0.10, 0.15], title='Input compartment $V_m$')   
        ax_in_shifted = fig.add_axes([0.04, 0.05, 0.10, 0.15], title='Shifted input compartment $V_m$')   

    
    if input_type == 'WN':
        ax_vm_psd = fig.add_axes([0.2, 0.55, 0.10, 0.40], title='Soma $V_m$ PSD')
        ax_in_psd = fig.add_axes([0.2, 0.05, 0.10, 0.40], title='Input compartment $V_m$ PSD')
        ax_vm_psd.set_xlabel('Hz')
        ax_vm_psd.set_ylabel('mV')
        ax_in_psd.set_xlabel('Hz')
        ax_in_psd.set_ylabel('mV')    

        ax_in_psd.set_xlim(1e0, 1e3)
        ax_vm_psd.set_xlim(1e0, 1e3)
        ax_in_psd.set_ylim(1e-6, 1e1)
        ax_vm_psd.set_ylim(1e-6, 1e1)

        
    
    ax_neur = fig.add_axes([0.5, 0.1, 0.1, 0.75], frameon=False, 
                            xticks=[], yticks=[])
    #plt.figtext(0.74, 0.9, 'Sum of transmembrane \n currents along y-axis', size=15)
    ax_vm.set_xlabel('ms')
    ax_vm.set_ylabel('mV')
    ax_in.set_xlabel('ms')
    ax_in.set_ylabel('mV')

    #ax_neur.set_ylabel('y [$\mu m$]', color='b')

    if input_type == 'WN':
        ax_list = [ax_vm, ax_in, ax_in_shifted, ax_vm_shifted, ax_vm_psd, ax_in_psd]
    else:
        ax_list = [ax_vm, ax_in, ax_in_shifted, ax_vm_shifted]
    for ax in ax_list:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    # Setting up helpers and colors
    elec_clr_list = []
    for idx in xrange(len(elec_x)):
        i = 256. * idx
        if len(elec_x) > 1:
            i /= len(elec_x) - 1.
        else:
            i /= len(elec_x)
        elec_clr_list.append(plt.cm.rainbow(int(i)))
    
    ymin = plot_params['ymin']
    ymax = plot_params['ymax']
    #yticks = np.arange(ymin, ymax + 1 , 250)
    #yticks = yticks[np.where((np.min(ymid) <= yticks) * (yticks <= np.max(ymid)))]
    #for ax in [ax_neur]:
    #    ax.get_yaxis().tick_left()
    #    ax.set_yticks(yticks)
    #    for tl in ax.get_yticklabels():
    #        tl.set_color('b')
    
    # Starting plotting
    for conductance_type in conductance_list:
        print conductance_type
        ax_vm.plot(tvec, vmem_dict[conductance_type][0,:], lw=1,
                   color=conductance_color_dict[conductance_type], label=conductance_type)
        ax_vm_shifted.plot(tvec, vmem_dict[conductance_type][0,:] - vmem_dict[conductance_type][0,0], 
                         lw=1, color=conductance_color_dict[conductance_type])
        ax_in.plot(tvec, vmem_dict[conductance_type][input_idx,:], lw=1,
                   color=conductance_color_dict[conductance_type])
        ax_in_shifted.plot(tvec, vmem_dict[conductance_type][input_idx,:] - \
                         vmem_dict[conductance_type][input_idx,0], lw=1,
                         color=conductance_color_dict[conductance_type])
        if input_type == 'WN':
            ax_vm_psd.loglog(freqs, vmem_psd_dict[conductance_type][0,:], lw=1,
                             color=conductance_color_dict[conductance_type], label=conductance_type)
            ax_in_psd.loglog(freqs, vmem_psd_dict[conductance_type][input_idx,:], lw=1,
                         color=conductance_color_dict[conductance_type])
    
    if input_type == 'WN':
        ax_vm_psd.legend(bbox_to_anchor=[2.1,1])
    else:
        ax_vm.legend(bbox_to_anchor=[2.1,1])
    for comp in xrange(len(xmid)):
        if comp == 0:
            ax_neur.scatter(xmid[comp], ymid[comp], s=diam[comp]*10, c='gray', edgecolor='none')
        else:
            ax_neur.plot([xstart[comp], xend[comp]], [ystart[comp], yend[comp]], 
                         lw=diam[comp], color='gray')

    ax_neur.plot(xmid[input_idx], ymid[input_idx], '*', color='y', label='Input', markersize=15)
    ax_neur.legend(bbox_to_anchor=[.6, 1.05], numpoints=1)
    ax_neur.axis([-200, 200, np.min([np.min(elec_y) - 50, ymin]) , np.max([np.max(elec_y) + 50, ymax])])
    
    ext_ax_width=0.1
    ext_ax_height = 0.6/len(elec_x)
    neur_ax = ax_neur.axis()
    
    for elec in xrange(len(elec_x)):
        if elec_x[elec] > 0:
            ax_xpos = 0.65
        else:
            ax_xpos = 0.35
        ax_neur.plot(elec_x[elec], elec_y[elec], 'o', color=elec_clr_list[elec])
        ax_temp = fig.add_axes([ax_xpos, 0.15 + elec*(ext_ax_height+0.05)/2., 
                                ext_ax_width, ext_ax_height])
        ax_neur.axis(neur_ax)
        ax_temp.spines['top'].set_visible(False)
        ax_temp.spines['right'].set_visible(False)
        ax_temp.get_xaxis().tick_bottom()
        ax_temp.get_yaxis().tick_left()
        ax_temp.tick_params(color=elec_clr_list[elec])
        for spine in ax_temp.spines.values():
            spine.set_edgecolor(elec_clr_list[elec])

        #ax_temp.set_xticklabels([])
        #ax_temp.set_yticklabels([])
        #ax_temp.grid(True)
        for idx, conductance_type in enumerate(conductance_list):
            if input_type == 'WN':
                ax_temp.loglog(freqs, sig_psd_dict[conductance_type][elec], 
                               color=conductance_color_dict[conductance_type], lw=0.7)
            else:
                ax_temp.plot(tvec, sig_dict[conductance_type][elec] - sig_dict[conductance_type][elec][0], 
                               color=conductance_color_dict[conductance_type], lw=0.7)
                
        pos = [elec_x[elec], elec_y[elec]]
        if elec == 0:
            #ax_temp.legend(bbox_to_anchor=[1.4, 1.22])
            if input_type == 'WN':
                ax_temp.set_xlabel('[Hz]')
            else:
                ax_temp.set_xlabel('[ms]')
            ax_temp.set_ylabel('Norm. amp')

        if input_type == 'WN':
            ax_temp.set_xlim(1e0, 1e3)
            ax_temp.set_ylim(1e-6, 1e0)
        elif input_type == 'synaptic':
            ax_temp.set_xlim(0, 200)
        elif input_type == 'ZAP':
            ax_temp.set_xlim(0, 20000)

        LFP_arrow_to_axis(pos, ax_neur, ax_temp, elec_clr_list[elec], ax_xpos)
    ax_neur.axis(neur_ax)
    plt.savefig('LFP_%s_%s.png' % (ifolder, sim_name), dpi=150)
    #plt.show()

def LFP_arrow_to_axis(pos, ax_origin, ax_target, clr, ax_xpos):
    if ax_xpos < 0.5:
        upper_pixel_coor = ax_target.transAxes.transform(([1,0.5]))
    else:
        upper_pixel_coor = ax_target.transAxes.transform(([0,0.5]))
    upper_coor = ax_origin.transData.inverted().transform(upper_pixel_coor)

    upper_line_x = [pos[0], upper_coor[0]]
    upper_line_y = [pos[1], upper_coor[1]]
    
    ax_origin.plot(upper_line_x, upper_line_y, lw=1, 
              color=clr, clip_on=False, alpha=1.)


def explore_morphology(morph_path):

    cell_params = {
        'morphology': morph_path,
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
    }

    cell = LFPy.Cell(**cell_params)
    # for sec in nrn.allsec():
    #     n3d = int(nrn.n3d())
    #     for i in xrange(n3d):
    #         plt.plot(nrn.x3d(i), nrn.y3d(i), '.')
    # plt.axis('equal')
    # plt.show()

    comp = 0
    for sec in cell.allseclist:
        name = sec.name()
        clr = 'r' if name == 'apic[92]' else 'k'
        for segnum, seg in enumerate(sec):
            plt.plot([cell.xstart[comp], cell.xend[comp]], [cell.ystart[comp], cell.yend[comp]],
                     lw=cell.diam[comp], color=clr)
            #if segnum == sec.nseg - 1:
            #    plt.text(cell.xend[comp], cell.yend[comp], name)
            comp += 1
    plt.axis('equal')
    plt.show()