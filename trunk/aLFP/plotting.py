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
    pl.close('all')    
    fig = pl.figure(figsize=[14,8])
    fig.suptitle("Model: %s, input_idx: %d, input_scaling: %g "%(ifolder, input_idx, input_scaling))
    pl.subplots_adjust(hspace=0.5)
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
        comp_clr_list.append(pl.cm.rainbow(int(i)))
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
    #ax_neur.add_artist(pl.Line2D((xmin, xmin), (np.min(ymid), np.max(ymid)), color='b', linewidth=3))
    pl.savefig('%s_%s.png' % (ifolder, cur_name), dpi=150)

    
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
    pl.close('all')    
    
    fig = pl.figure(figsize=[8,8])
    fig.suptitle("Model: %s, input_idx: %d, input_scaling: %g "%(ifolder, input_idx, input_scaling))

    pl.subplots_adjust(hspace=0.5)
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
        comp_clr_list.append(pl.cm.rainbow(int(i)))
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
    pl.savefig('imem_%s_%s_%s.png' % (ifolder, cur_name, sim_type), dpi=300)

def plot_synaptic_currents(ifolder, input_scaling, input_idx, plot_params, simulation_params,
                            plot_compartments, epas=None):


    if epas == None:
        cur_name = '%d_%1.3f' %(input_idx, input_scaling)
    else:
        cur_name = '%d_%1.3f_%g' %(input_idx, input_scaling, epas)

    # Loading all needed data
    imem_psd_active = np.load(join(ifolder, 'imem_psd_%d_%1.3f_%s.npy' 
                                   %(input_idx, input_scaling, 'active')))
    imem_active = np.load(join(ifolder, 'imem_%d_%1.3f_%s.npy' 
                               %(input_idx, input_scaling, 'active')))
    imem_psd_reduced = np.load(join(ifolder, 'imem_psd_%d_%1.3f_%s.npy' 
                                    %(input_idx, input_scaling, 'reduced_Ih')))
    imem_reduced = np.load(join(ifolder, 'imem_%d_%1.3f_%s.npy' 
                                %(input_idx, input_scaling, 'reduced_Ih')))
    imem_psd_passive = np.load(join(ifolder, 'imem_psd_%d_%1.3f_%s.npy' 
                                    %(input_idx, input_scaling, 'passive')))
    imem_passive = np.load(join(ifolder, 'imem_%d_%1.3f_%s.npy' 
                                %(input_idx, input_scaling, 'passive')))
    
    #icap = np.load(join(ifolder, 'icap_psd_%s.npy' %(cur_name)))
    #ipas = np.load(join(ifolder, 'ipas_psd_%s.npy' %(cur_name)))
    active_dict = {}
    clr_list = ['r', 'b', 'g', 'm']

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
        comp_clr_list.append(pl.cm.rainbow(int(i)))
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
    pl.close('all')    
    fig = pl.figure(figsize=[8,8])
    fig.suptitle("Model: %s, input_idx: %d, input_scaling: %g "%(ifolder, input_idx, input_scaling))

    pl.subplots_adjust(hspace=0.5)
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

        #ax_temp.plot(freqs, ipas[comp,:], label='Ipas', color='y', lw=2)
        #ax_temp.plot(1.5, ipas[comp,0], '+', color='y')
        #ax_temp.plot(freqs, icap[comp,:], '--', label='Icap', color='grey', lw=2)
        #ax_temp.plot(1.5, icap[comp,0], '+', color='grey')
        #set_trace()
        ax_temp.plot(tvec, imem_passive[comp] - imem_passive[comp, 0], color=pas_clr, lw=2, label='Passive')      
        #ax_temp.plot(1.5, imem_passive[comp,0], '+', color=pas_clr)
        ax_temp.plot(tvec, imem_reduced[comp] - imem_reduced[comp, 0], color=reduced_clr, lw=1, label='Reduced')      
        #ax_temp.plot(1.5, imem_reduced[comp,0], '+', color=reduced_clr)
        ax_temp.plot(tvec, imem_active[comp] - imem_active[comp, 0], '--', color=act_clr, lw=1, label='active')      
        #ax_temp.plot(1.5, imem_active[comp,0], '+', color=act_clr)'
        #ax_temp.set_ylim(1e-7, 1e1)
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
    pl.savefig('synaptic_%s_%s.png' % (ifolder, cur_name), dpi=300)


    
    # Initializing frequency-axis figure
    pl.close('all')    
    fig = pl.figure(figsize=[8,8])
    fig.suptitle("Model: %s, input_idx: %d, input_scaling: %g "%(ifolder, input_idx, input_scaling))

    pl.subplots_adjust(hspace=0.5)
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

        #ax_temp.plot(freqs, ipas[comp,:], label='Ipas', color='y', lw=2)
        #ax_temp.plot(1.5, ipas[comp,0], '+', color='y')
        #ax_temp.plot(freqs, icap[comp,:], '--', label='Icap', color='grey', lw=2)
        #ax_temp.plot(1.5, icap[comp,0], '+', color='grey')
        #set_trace()
        ax_temp.plot(freqs, imem_psd_passive[comp], color=pas_clr, lw=2, label='Passive')      
        #ax_temp.plot(1.5, imem_passive[comp,0], '+', color=pas_clr)
        ax_temp.plot(freqs, imem_psd_reduced[comp], color=reduced_clr, lw=1, label='Reduced')      
        #ax_temp.plot(1.5, imem_reduced[comp,0], '+', color=reduced_clr)
        ax_temp.plot(freqs, imem_psd_active[comp], '--', color=act_clr, lw=1, label='active')      
        #ax_temp.plot(1.5, imem_active[comp,0], '+', color=act_clr)
        ax_temp.set_ylim(1e-9, 1e-1)
            
        pos = [xmid[comp], ymid[comp]]
        if numb == 0:
            ax_temp.legend(bbox_to_anchor=[1.8, 1.22])
            ax_temp.set_xlabel('Hz')
            ax_temp.set_ylabel('Transmemb. current')
            #ax_temp.axis(ax_temp.axis('tight'))
            #ax_temp.set_xlim(1,1000)
        
        ax_temp.set_xscale('log')
        ax_temp.set_yscale('log')
        arrow_to_axis(pos, ax_neur, ax_temp, comp_clr_list[numb], x_pos)
    pl.savefig('synaptic_%s_%s_psd.png' % (ifolder, cur_name), dpi=300)



    
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
    pl.savefig('active_currents_%s_%s_%s_psd.png' % (ifolder, sim_name, conductance_type), dpi=150)
    #pl.savefig('active_currents_%s_%s_True_psd.pdf' % (ifolder, sim_name))
    
    
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

    pl.figtext(0.74, 0.9, 'Sum of transmembrane \n currents along y-axis', size=15)
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
    ## pl.colorbar(sc_stick_act, ax=ax_act_imshow)
    ## pl.colorbar(sc_stick_pas, ax=ax_pas_imshow)
    ## pl.colorbar(sc_stick_psd_pas, ax=ax_pas_psd_imshow)
    ## pl.colorbar(sc_stick_psd_act, ax=ax_act_psd_imshow)

    ## xmin, xmax = ax_neur.get_xaxis().get_view_interval()
    ## ax_neur.add_artist(pl.Line2D((xmin, xmin), (np.min(ymid), np.max(ymid)), color='b', linewidth=3))
    
    pl.savefig('WN_%s_%s_reduced.png' % (ifolder, name))

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
    pl.close('all')    
    fig = pl.figure(figsize=[14,8])
    fig.suptitle("Model: %s"%(ifolder))

    pl.subplots_adjust(hspace=0.5)

    ax_im = fig.add_axes([0.05, 0.4, 0.1, 0.2], title='Soma $I_m$')
    ax_vm = fig.add_axes([0.05, 0.70, 0.10, 0.20], title='Soma $V_m$')    
    ax_imshow = fig.add_axes([0.73, 0.72, 0.25, 0.13])
    ax_im_psd = fig.add_axes([0.18, 0.4, 0.1, 0.2], title='Soma $I_m$ PSD')

    ax_vm_psd = fig.add_axes([0.18, 0.70, 0.10, 0.20], title='Soma $V_m$ PSD')    
    ax_psd_imshow = fig.add_axes([0.73, 0.32, 0.25, 0.13], title='PSD', xscale='log')
    ax_neur = fig.add_axes([0.25, 0.1, 0.35, 0.75], frameon=False, aspect='equal', xticks=[])

    pl.figtext(0.74, 0.9, 'Sum of transmembrane \n currents along y-axis', size=15)
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
        comp_clr_list.append(pl.cm.rainbow(int(i)))
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
    pl.colorbar(sc_stick, ax=ax_imshow)

    pl.colorbar(sc_stick_psd, ax=ax_psd_imshow)

    xmin, xmax = ax_neur.get_xaxis().get_view_interval()
    ax_neur.add_artist(pl.Line2D((xmin, xmin), (np.min(ymid), np.max(ymid)), color='b', linewidth=3))
    
    pl.savefig('%s_%s.png' % (ifolder, name))

