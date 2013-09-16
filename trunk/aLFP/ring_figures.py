import pylab as plt
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
import matplotlib.patches as mpatches

plt.rcParams.update({'font.size' : 8,
                     'figure.facecolor' : '1',
                     'wspace' : 0.5, 
                     'hspace' : 0.5,
                     'legend.fontsize' : 7,
                     })
#np.random.seed(1234)



def return_circle_idxs(ring_dict, radius_idx, height_idx, elec_x, elec_y, elec_z):
    circle_idxs = []
    circle_R = ring_dict['radiuses'][radius_idx]
    circle_height = ring_dict['heights'][height_idx]
    
    for elec_idx in xrange(len(elec_x)):
        R = np.sqrt(elec_x[elec_idx]**2 + elec_z[elec_idx]**2)
        if np.abs(elec_y[elec_idx] - circle_height) < 1e-8 and \
           np.abs(R - circle_R) < 1e-8:
            circle_idxs.append(elec_idx)
    if not len(circle_idxs) == ring_dict['numpoints_on_ring']:
        raise RuntimeError
    return circle_idxs


def plot_all_circle_signals(xmid, ymid, zmid, xstart, ystart, zstart, xend, yend, zend, diam, 
                            circle_idxs, elec_x,elec_y, elec_z, tvec, freqs, 
                            sig_psd, sig_dict, circle_avrg_sig, circle_avrg_sig_psd, 
                            ring_dict, sim_name, height_idx, radius_idx, conductance_type):
    plt.close('all')
    fig = plt.figure()                 
    ax1 = fig.add_subplot(222)
    ax2 = fig.add_subplot(224)
    ax_top = fig.add_subplot(221, aspect='equal', frameon=False, xticks=[])     
    ax_side = fig.add_subplot(223, aspect='equal', frameon=False, xticks=[])     
    for comp in xrange(len(xmid)):
        if comp == 0:
            ax_top.scatter(xmid[comp], zmid[comp], s=diam[comp]*10, c='gray', 
                                  edgecolor='none')
            ax_side.scatter(xmid[comp], ymid[comp], s=diam[comp]*10, c='gray', 
                                  edgecolor='none')
        else:
            ax_top.plot([xstart[comp], xend[comp]], [zstart[comp], zend[comp]], 
                               lw=diam[comp], color='gray')
            ax_side.plot([xstart[comp], xend[comp]], [ystart[comp], yend[comp]], 
                               lw=diam[comp], color='gray')

    for sig_idx in circle_idxs:

        angle = np.arctan2(elec_z[sig_idx], elec_x[sig_idx])
        if angle < 0:
            angle += 2*np.pi
        ax_top.plot(elec_x[sig_idx], elec_z[sig_idx], 'o', 
                 color=plt.cm.hsv(int(256*angle/(2*np.pi))))
        ax_side.plot(elec_x[sig_idx], elec_y[sig_idx], 'o', 
                 color=plt.cm.hsv(int(256*angle/(2*np.pi))))
        ax1.plot(tvec, sig_dict[conductance_type][sig_idx], 
                 color=plt.cm.hsv(int(256*angle/(2*np.pi))))

        ax2.loglog(freqs, sig_psd[sig_idx],
                 color=plt.cm.hsv(int(256*angle/(2*np.pi))))

    ax1.plot(tvec, circle_avrg_sig, 'k', lw=2)
    ax2.loglog(freqs, circle_avrg_sig_psd[0,:], 'k', lw=2)
    ax2.set_xlim(1e0,1.1e3)
    ax2.set_ylim(1e-8,1e-4)
    fig.suptitle("Height: %g, Radius: %g, Name: %s" %(ring_dict['heights'][height_idx], 
                                                   ring_dict['radiuses'][radius_idx],
                                                   sim_name)) 
    fig.savefig('circle_averaged_signal_%d_%d_%s_%s.png' 
                %(height_idx, radius_idx, sim_name, conductance_type))


def make_freq_dist_fig(freqs, psd_vs_dist_dict, conductance_list, num_heights, ring_dict, multiple_input, input_idx_scale, input_idx):

    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.normalize(vmin=1, vmax=1000))
    sm._A = []
    
    plt.close('all')
    freq_clr = lambda freq_idx: plt.cm.jet(int(256. * freq_idx/(len(np.where(freqs <= 1000)[0]) - 1.)))

    freq_distfig = plt.figure(figsize=[12,6])
    freq_distfig.subplots_adjust(wspace=0.5, hspace=0.5)
    for height_idx in xrange(num_heights):
        for cond_number, conductance_type in enumerate(conductance_list): 
            plotnumber = (num_heights - height_idx - 1) * len(conductance_list) + cond_number + 1
            ax = freq_distfig.add_subplot(num_heights, len(conductance_list), 
                                          plotnumber)
            ax.grid(True)
            # Plotting decay shower
            x = np.array([1000, 1500])
            ax.plot(x, 1e0 * 1000**2/x**2, color='k', lw=2, label='$R^2$')
            ax.set_xlim(200, 1600)
            ax.set_ylim(1e-3,1.2)
            ax.set_yscale('log')
            if height_idx == 0 and cond_number == 0:
                ax.legend(bbox_to_anchor=[1.1,1.2])
                cbar_ax = freq_distfig.add_axes([0.93, 0.25, 0.01, 0.5])
                cbar = plt.colorbar(sm, cax=cbar_ax)
                cbar.ax.set_ylabel('Frequency (Hz)')
                ax.set_xlabel('Distance [$\mu m$]')
                ax.set_ylabel('Normalized amplitude')
            ax.set_title('%s, H: %g' %(conductance_type, ring_dict['heights'][height_idx]))
            for freq_idx in xrange(1,len(freqs)):
                if freqs[freq_idx] > 1000:
                    continue
                ax.plot(ring_dict['radiuses'], psd_vs_dist_dict[conductance_type][height_idx,:,freq_idx] / np.max(psd_vs_dist_dict[conductance_type][height_idx,:,freq_idx]), 
                             color=freq_clr(freq_idx), rasterized=True, zorder=1, lw=0.5)
                
                #plt.colorbar()
    if multiple_input:
        freq_distfig.savefig('freq_vs_dist_PSD_multiple_input_%d.png' % len(input_idx_scale), dpi=150)
    else:
        freq_distfig.savefig('freq_vs_dist_PSD_%d.png' %(input_idx), dpi=150)

        
def average_circle(ifolder, conductance_list, input_idx_scale, 
                   ring_dict, elec_x, elec_y, elec_z):

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
    freqs = np.load(join(ifolder, 'freqs.npy'))    
    num_heights = len(ring_dict['heights'])
    num_radii = len(ring_dict['radiuses'])
    
    try:
        input_idx, input_scaling = input_idx_scale
        multiple_input = False
    except ValueError:
        multiple_input = True
    input_type = 'WN'
    sig_dict = {}
    vmem_dict = {}
    vmem_psd_dict = {}
    conductance_color_dict = {} 
    psd_vs_dist_dict = {}
    
    if multiple_input:
        sim_name = 'multiple_input_%d_%s' %(len(input_idx_scale), input_type)
    else:
        sim_name = '%d_%1.3f_%s' %(input_idx, input_scaling, input_type)
    tvec = np.load(join(ifolder, 'tvec_%s.npy' % input_type))
    timestep = (tvec[1] - tvec[0])/1000.
    print sim_name   
    for cond_number, conductance_type in enumerate(conductance_list):
        conductance_name = "%s_%s" %(sim_name, conductance_type)
        #vmem_dict[conductance_type] = np.load(join(ifolder, 'vmem_%s.npy' %(conductance_name)))
        #if input_type == 'WN':
        #    vmem_psd_dict[conductance_type] = np.load(join(ifolder, 'vmem_psd_%s.npy' %(conductance_name)))
        sig_dict[conductance_type] = np.load(join(ifolder, 'sig_%s.npy' %(conductance_name)))
        psd_vs_dist_dict[conductance_type] = np.zeros((num_heights, num_radii, len(freqs)))
        if len(conductance_list) > 1:
            clr_number = 256. * cond_number/(len(conductance_list) - 1.)
        else:
            clr_number = 256. * cond_number/(len(conductance_list))
        conductance_color_dict[conductance_type] = plt.cm.jet(int(clr_number))

    ## plt.close('all')
    ## bigfig = plt.figure(figsize=[10,6])
    ## bigfig.suptitle(("PSD of potential averaged over %d points on circle around cell.\n"+
    ##                 "Columns correspond to different radii and rows to different heights "+
    ##                 "as indicated by colored circles. White noise input marked by star""")
    ##                 % ring_dict['numpoints_on_ring'])
    ## bigfig.subplots_adjust(wspace=0.5)

    ## ax_side = bigfig.add_axes([0.01, 0.25, 0.2, 0.3], frameon=False, aspect='equal', xticks=[], yticks=[])
    ## ax_side.axis([-1600, 1600, -1000, 2000])

    ## ax_top = bigfig.add_axes([0.01, 0.55, 0.2, 0.3], frameon=False, aspect='equal', xticks=[], yticks=[])
    ## ax_top.axis([-1600, 1600, -1600, 1600])
 
    ## for comp in xrange(len(xmid)):
    ##     if comp == 0:
    ##         ax_side.scatter(xmid[comp], ymid[comp], s=diam[comp], c='gray', 
    ##                               edgecolor='none')
    ##         ax_top.scatter(xmid[comp], zmid[comp], s=diam[comp], c='gray', 
    ##                               edgecolor='none')
    ##     else:
    ##         ax_side.plot([xstart[comp], xend[comp]], [ystart[comp], yend[comp]], 
    ##                            lw=diam[comp]/2, color='gray')
    ##         ax_top.plot([xstart[comp], xend[comp]], [zstart[comp], zend[comp]], 
    ##                            lw=diam[comp]/2, color='gray')
    ## if multiple_input:
    ##     for input_idx, input_scaling in input_idx_scale:
    ##         ax_side.scatter(xmid[input_idx], ymid[input_idx], s=100, c='y', marker='*',
    ##                         edgecolor='none', zorder=1000)
    ##         ax_top.scatter(xmid[input_idx], zmid[input_idx], s=100, c='y', marker='*',
    ##                        edgecolor='none', zorder=1000)
    ## else:
    ##     ax_side.scatter(xmid[input_idx], ymid[input_idx], s=100, c='y', marker='*',
    ##                     edgecolor='none', zorder=1000)
    ##     ax_top.scatter(xmid[input_idx], zmid[input_idx], s=100, c='y', marker='*',
    ##                    edgecolor='none', zorder=1000)
    ## ax_top.text(0, np.max(elec_z) + 50, 'R=%g $\mu m$' %np.max(ring_dict['radiuses']))
    ## ax_side.text(0, np.max(ring_dict['heights']) + 250, 'H=%g $\mu m$' %np.max(ring_dict['heights']), angle=90)
    
    numrings = num_heights * num_radii

    ring_clr = lambda ring_idx: plt.cm.rainbow(int(256. * ring_idx/(numrings - 1.)))
    ring_idx = 0

    for height_idx in xrange(num_heights):
        for radius_idx in xrange(num_radii):

            ## side_c = mpatches.Ellipse((0, ring_dict['heights'][height_idx]), 
            ##                      2*ring_dict['radiuses'][radius_idx], ring_dict['radiuses'][radius_idx]/10, 
            ##                      fc="none", ec=ring_clr(ring_idx), lw=1, zorder=100)
            ## ax_side.add_patch(side_c)
            ## if height_idx == num_heights -1:
            ##     top_c = mpatches.Ellipse((0, 0), 
            ##                              2*ring_dict['radiuses'][radius_idx], 
            ##                              2*ring_dict['radiuses'][radius_idx], 
            ##                              fc="none", ec=ring_clr(ring_idx), lw=1, zorder=100)
            ##     ax_top.add_patch(top_c)
        
            ## plotnumber = (num_radii + 1) * (num_heights - height_idx - 1) + radius_idx + 2
            for cond_number, conductance_type in enumerate(conductance_list):    
                
                circle_idxs = return_circle_idxs(ring_dict, radius_idx, height_idx, elec_x, elec_y, elec_z)

                circle_avrg_sig = np.average(sig_dict[conductance_type][circle_idxs], axis=0)

                sig_psd, freqs = aLFP.find_LFP_PSD(sig_dict[conductance_type], timestep)
                circle_avrg_sig_psd, freqs = aLFP.find_LFP_PSD(np.array([circle_avrg_sig]), timestep)
                psd_vs_dist_dict[conductance_type][height_idx, radius_idx, :] = circle_avrg_sig_psd
    ##             ax = bigfig.add_subplot(num_heights, num_radii + 1, plotnumber)
    ##             ax.grid(True)
    ##             ax.tick_params(color=ring_clr(ring_idx))
    ##             for spine in ax.spines.values():
    ##                 spine.set_edgecolor(ring_clr(ring_idx))
    ##             ax.loglog(freqs, circle_avrg_sig_psd[0,:], lw=1, 
    ##                       color=conductance_color_dict[conductance_type], label=conductance_type)
                
    ##             if height_idx == num_heights -1:
    ##                 ax.set_title('R=%g $\mu m$' % (ring_dict['radiuses'][radius_idx]))
    ##             if radius_idx == num_radii - 1:
    ##                 ax.text(1200,1e-6, "H=%g $\mu m$" %ring_dict['heights'][height_idx])
                    
    ##             ax.set_xlim(1e0,1.1e3)
    ##             ax.set_ylim(1e-8,1e-4)       
    ##             if height_idx == 0 and radius_idx == 0:
    ##                 ax.set_ylabel('Amplitude')
    ##                 ax.set_xlabel('Hz')
    ##             if radius_idx == num_radii - 1 and height_idx == num_heights -1:
    ##                 ax.legend(bbox_to_anchor=[1.5, 1.]) 
    ##             #plot_all_circle_signals(xmid, ymid, zmid, xstart, ystart, zstart, xend, yend, zend, 
    ##             #                        diam, circle_idxs, elec_x,elec_y, elec_z, tvec, freqs, sig_psd, 
    ##             #                        sig_dict, circle_avrg_sig, circle_avrg_sig_psd, 
    ##             #                        ring_dict, sim_name, height_idx, radius_idx, conductance_type)
    ##         ring_idx += 1
    
    ## if multiple_input:
    ##     bigfig.savefig('distance_n_height_PSD_multiple_input_%d.png' % len(input_idx_scale), dpi=150)
    ## else:
    ##     bigfig.savefig('distance_n_height_PSD_%d.png' %(input_idx), dpi=150)
    if multiple_input:
        make_freq_dist_fig(freqs, psd_vs_dist_dict, conductance_list, num_heights, ring_dict, multiple_input, input_idx_scale, input_idx=None)
    else:
        make_freq_dist_fig(freqs, psd_vs_dist_dict, conductance_list, num_heights, ring_dict, multiple_input, input_idx_scale, input_idx)
                
def plot_ring(ifolder,elec_x, elec_y, elec_z):

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
    fig.suptitle("Model: %s" %(ifolder))

    ax_neur_above = fig.add_axes([0.05, 0.1, 0.45, 0.75], frameon=False, 
                                 xticks=[], yticks=[])
    ax_neur_side = fig.add_axes([0.55, 0.1, 0.45, 0.75], frameon=False, 
                            xticks=[], yticks=[], sharex=ax_neur_above)

    ax_neur_above.plot(elec_x, elec_z, 'ko')
    ax_neur_side.plot(elec_x, elec_y, 'ko')

    for comp in xrange(len(xmid)):
        if comp == 0:
            ax_neur_side.scatter(xmid[comp], ymid[comp], s=diam[comp]*10, c='gray', edgecolor='none')
            ax_neur_above.scatter(xmid[comp], zmid[comp], s=diam[comp]*10, c='gray', edgecolor='none')
        else:
            ax_neur_side.plot([xstart[comp], xend[comp]], [ystart[comp], yend[comp]], 
                         lw=diam[comp], color='gray')
            ax_neur_above.plot([xstart[comp], xend[comp]], [zstart[comp], zend[comp]], 
                         lw=diam[comp], color='gray')

    ax_neur_side.legend(bbox_to_anchor=[.6, 1.05], numpoints=1)
    ax_neur_side.axis([np.min(elec_x) -100, np.max(elec_x) + 100, 
                       np.min([np.min(elec_y) - 50, np.min(yend)]) , 
                       np.max([np.max(elec_y) + 50, np.max(yend)])])
    fig.savefig('ring_setup.png')

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

def _LFP_arrow_to_axis(pos, ax_origin, ax_target, clr, ax_xpos):
    if ax_xpos < 0.5:
        upper_pixel_coor = ax_target.transAxes.transform(([1,0.5]))
    else:
        upper_pixel_coor = ax_target.transAxes.transform(([0,0.5]))
    upper_coor = ax_origin.transData.inverted().transform(upper_pixel_coor)

    upper_line_x = [pos[0], upper_coor[0]]
    upper_line_y = [pos[1], upper_coor[1]]
    
    ax_origin.plot(upper_line_x, upper_line_y, lw=1, 
              color=clr, clip_on=False, alpha=1.)
