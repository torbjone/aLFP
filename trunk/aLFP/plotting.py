import pylab as pl
from matplotlib.colors import LogNorm
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

def arrow_to_axis(pos, ax_origin, ax_target, clr):
    
    upper_pixel_coor = ax_target.transAxes.transform(([0,1]))
    lower_pixel_coor = ax_target.transAxes.transform(([0,0]))
    
    upper_coor = ax_origin.transData.inverted().transform(upper_pixel_coor)
    lower_coor = ax_origin.transData.inverted().transform(lower_pixel_coor)
    shift = 300
    upper_line_x = [pos[0], upper_coor[0] + shift]
    upper_line_y = [pos[1], upper_coor[1]]
    
    lower_line_x = [pos[0], lower_coor[0] + shift]
    lower_line_y = [pos[1], lower_coor[1]]    
    #lower_line_x = [origin_coor[0], lower_target_coor[0]]
    #lower_line_y = [origin_coor[1], lower_target_coor[1]]
    
    ax_origin.plot(upper_line_x, upper_line_y, lw=1, 
              color=clr, clip_on=False, alpha=1.)
    ax_origin.plot(lower_line_x, lower_line_y, lw=1, 
              color=clr, clip_on=False, alpha=1.)

    
def plot_active_currents(ifolder, input_scaling, input_idx, simulation_params):


    name = "%d_%1.3f" %(input_idx, input_scaling)
    name_active = "psd_%d_%1.3f_%s" %(input_idx, input_scaling, 'True')
    freqs = np.load(join(ifolder, 'freqs.npy'))
    freqs[0] += 1e-4
    
    tvec = np.load(join(ifolder, 'tvec.npy'))
    imem = 1000 * np.load(join(ifolder, 'imem_%s.npy' %(name_active)))
    icap = 1000 * np.load(join(ifolder, 'icap_%s.npy' %(name_active)))
    ipas = 1000 * np.load(join(ifolder, 'ipas_%s.npy' %(name_active)))
    active_dict = {}


    clr_list = ['r', 'b', 'g', 'm']
    
    for cur in simulation_params['rec_variables']:
        active_dict[cur] = 1000 * np.load(join(ifolder, '%s_%s.npy'%(cur, name_active)))
    sim_name = '%d_%1.3f' %(input_idx, input_scaling)
    pl.close('all')
    n_cols = 6
    n_rows = 6
    fig = pl.figure(figsize=[15,10])
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
    pl.savefig('active_currents_%s_%s_True_psd.png' % (ifolder, sim_name), dpi=300)
    #pl.savefig('active_currents_%s_%s_True_psd.pdf' % (ifolder, sim_name))
    
    
def compare_active_passive(ifolder, input_scaling, input_idx, elec_x, elec_y, elec_z, plot_params):

    name = "%d_%1.3f" %(input_idx, input_scaling)
    name_active = "%d_%1.3f_%s" %(input_idx, input_scaling, 'True')
    name_passive= "%d_%1.3f_%s" %(input_idx, input_scaling, 'False')

    # Loading all needed data
    imem_active = np.load(join(ifolder, 'imem_%s.npy' %(name_active)))
    imem_psd_active = np.load(join(ifolder, 'imem_psd_%s.npy' %(name_active)))
    somav_psd_active = np.load(join(ifolder, 'somav_psd_%s.npy' %(name_active)))
    somav_active = np.load(join(ifolder, 'somav_%s.npy' %(name_active)))
    sig_active = np.load(join(ifolder, 'signal_%s.npy' %(name_active)))
    psd_active = np.load(join(ifolder, 'psd_%s.npy' %(name_active)))
    stick_active = np.load(join(ifolder, 'stick_%s.npy' %(name_active)))
    stick_psd_active = np.load(join(ifolder, 'stick_psd_%s.npy' %(name_active)))
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
    pl.close('all')    
    fig = pl.figure(figsize=[14,8])
    fig.suptitle("Model: %s, Input scaling: %s, Input index: %s"
                 %(ifolder, input_scaling, input_idx))

    pl.subplots_adjust(hspace=0.5)
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

    pl.figtext(0.74, 0.9, 'Sum of transmembrane \n  currents along y-axis', size=15)
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
    elec_clr_list = []
    for idx in xrange(len(elec_x)):
        i = 256. * idx
        if len(elec_x) > 1:
            i /= len(elec_x) - 1.
        else:
            i /= len(elec_x)
        elec_clr_list.append(pl.cm.rainbow(int(i)))
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
    ax_vm_psd.loglog(freqs, somav_psd_active, act_clr, lw=2, label='Active')
    ax_vm_psd.loglog(freqs, somav_psd_passive, pas_clr, lw=2, label='Passive')
    ax_vm_psd.legend(bbox_to_anchor=[1.4,1.0])

    ax_im.plot(tvec, 1000*imem_active[0,:], act_clr)
    ax_im.plot(tvec, 1000*imem_passive[0,:], pas_clr)
    ax_im_psd.loglog(freqs, 1000*imem_psd_active[0,:], act_clr, lw=2)
    ax_im_psd.loglog(freqs, 1000*imem_psd_passive[0,:], pas_clr, lw=2)

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
        pos = [elec_x[elec], elec_y[elec]]
        
        if elec == 0:
            #ax_temp.legend(bbox_to_anchor=[1.4, 1.22])
            ax_temp.set_xlabel('Hz')
            ax_temp.set_ylabel('Norm. amp')
        ax_temp.set_xlim(1,1000)
        ax_temp.set_ylim(1e-2, 2e0)
        arrow_to_axis(pos, ax_neur, ax_temp, elec_clr_list[elec])

    # COLOR PLOT OF y-AXIS SUMMED IMEM CONTRIBUTION
    stick_pos = np.linspace(np.max(ymid), np.min(ymid), stick_passive.shape[0])
    X, Y = np.meshgrid(freqs, stick_pos)
    
    sc_stick_pas = ax_pas_imshow.pcolormesh(X, Y, 1000*stick_passive, cmap='jet_r',
                                            vmax=np.max(np.abs(1000*stick_passive)), 
                                            vmin=-np.max(np.abs(1000*stick_passive)))

    sc_stick_act = ax_act_imshow.pcolormesh(X, Y, 1000*stick_active, cmap='jet_r', 
                                            vmax=np.max(np.abs(1000*stick_active)), 
                                            vmin=-np.max(np.abs(1000*stick_active)))
    
    sc_stick_psd_pas = ax_pas_psd_imshow.pcolormesh(X, Y, 1000*stick_psd_passive, cmap='jet',
            norm=LogNorm(vmax=np.max(1000*stick_psd_passive), vmin=1e-4))
    sc_stick_psd_act = ax_act_psd_imshow.pcolormesh(X, Y, 1000*stick_psd_active, cmap='jet',
                                     norm=LogNorm(vmax=np.max(1000*stick_psd_active), vmin=1e-4))

    ax_pas_imshow.axis('auto')
    ax_act_imshow.axis('auto')
    pl.colorbar(sc_stick_act, ax=ax_act_imshow)
    pl.colorbar(sc_stick_pas, ax=ax_pas_imshow)
    pl.colorbar(sc_stick_psd_pas, ax=ax_pas_psd_imshow)
    pl.colorbar(sc_stick_psd_act, ax=ax_act_psd_imshow)

    xmin, xmax = ax_neur.get_xaxis().get_view_interval()
    ax_neur.add_artist(pl.Line2D((xmin, xmin), (np.min(ymid), np.max(ymid)), color='b', linewidth=3))
    
    pl.savefig('WN_%s_%s.png' % (ifolder, name))

