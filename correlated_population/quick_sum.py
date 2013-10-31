import numpy as np
import sys, os
from os.path import join
import aLFP
import pylab as plt
import matplotlib.mlab as mlab

n_elecs = 8
elec_x = np.zeros(n_elecs)
elec_y = np.zeros(n_elecs)
elec_z = np.linspace(-200, 1200, n_elecs)

def population_size():

    folder = 'stallo'
    xmid = np.load(join(folder, 'xmid.npy' ))
    ymid = np.load(join(folder, 'ymid.npy' ))
    xstart = np.load(join(folder, 'xstart.npy' ))
    ystart = np.load(join(folder, 'ystart.npy' ))
    zstart = np.load(join(folder, 'zstart.npy' ))
    xend = np.load(join(folder, 'xend.npy' ))
    yend = np.load(join(folder, 'yend.npy' ))
    zend = np.load(join(folder, 'zend.npy' ))    
    diam = np.load(join(folder, 'diam.npy'))
    n_elecs = len(elec_z)
    elec_clr = lambda elec_idx: plt.cm.rainbow(int(256. * elec_idx/(n_elecs - 1.)))
    population_radius = 1000
    population_radii = np.linspace(50, population_radius, 51)
    divide_into_welch = 8
    
    for input_pos in ['dend']:
        for correlation in [0, 1.0]:
            for radius in population_radii:
                signal_dict = {}
                signal_psd_dict = {}
                signal_psd_welch_dict = {}
                break_loop = False
                for conductance_type in ['active', 'passive_vss', 'Ih_linearized']:
                    stem='signal_%s_%s_%1.2f_total_pop_size_%04d' %(conductance_type, input_pos, correlation,
                                                                    int(radius))
                    signal_dict[conductance_type] = np.load(join(folder, '%s.npy' % stem))
                    signal_psd_dict[conductance_type], freqs = aLFP.find_LFP_PSD(signal_dict[conductance_type][:,:], (1.0)/1000.)
                    
                    foo, freqs_welch = mlab.psd(signal_dict[conductance_type][0,:], Fs=1000., NFFT=int(1001./divide_into_welch),
                                    noverlap=int(1001./divide_into_welch/2),
                                                      window=plt.window_hanning)
                    signal_psd_welch_dict[conductance_type] = np.zeros((n_elecs, len(foo)))
                    for elec in xrange(n_elecs):
                        signal_psd_welch_dict[conductance_type][elec, :], freqs_welch = mlab.psd(signal_dict[conductance_type][elec,:], Fs=1000.,
                                                                                                 NFFT=int(1001./divide_into_welch),
                                                                                                 noverlap=int(1001./divide_into_welch/2),
                                                                                                 window=plt.window_hanning, detrend=plt.detrend_mean)
                plt.close('all')
                fig = plt.figure(figsize=[7,10])
                fig.suptitle('Stimulation: %s, Correlation: %1.2f, Population size: %d $\mu m$'
                             % (input_pos, correlation, int(radius)))
                fig.subplots_adjust(hspace=0.5, wspace=0.5)
                ax_ = fig.add_subplot(1, 3, 1, frameon=False, xticks=[], yticks=[], ylim=[-300, 1300])
                for comp in xrange(len(xstart)):
                    if comp == 0:
                        ax_.scatter(xmid[comp], ymid[comp], s=diam[comp], color='k')
                    else:
                        ax_.plot([xstart[comp], xend[comp]], [ystart[comp], yend[comp]], lw=diam[comp], color='k')

                for elec in xrange(n_elecs):

                    ax_.plot(elec_x[elec], elec_z[elec], 'o', color=elec_clr(elec))
                    ax_psd_a = fig.add_subplot(n_elecs, 4, 4*(n_elecs - elec - 1) + 2, ylim=[1e-5, 1e1], xlim=[1e0, 1e3])
                    ax_psd_h = fig.add_subplot(n_elecs, 4, 4*(n_elecs - elec - 1) + 3, ylim=[1e-5, 1e1], xlim=[1e0, 1e3])
                    ax_psd_p = fig.add_subplot(n_elecs, 4, 4*(n_elecs - elec - 1) + 4, ylim=[1e-5, 1e1], xlim=[1e0, 1e3])

                    for ax in [ax_psd_a, ax_psd_h, ax_psd_p]:

                        ax.tick_params(color=elec_clr(elec))
                        for spine in ax.spines.values():
                            spine.set_edgecolor(elec_clr(elec))
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.get_xaxis().tick_bottom()
                        ax.get_yaxis().tick_left()
                        ax.grid(True)

                    if elec == n_elecs - 1:
                        ax_psd_a.set_title('Active')
                        ax_psd_h.set_title('Ih_linearized')
                        ax_psd_p.set_title('Passive')

                    ax_psd_a.loglog(freqs, signal_psd_dict['active'][elec], color='k')
                    ax_psd_h.loglog(freqs, signal_psd_dict['Ih_linearized'][elec], color='k')
                    ax_psd_p.loglog(freqs, signal_psd_dict['passive_vss'][elec], color='k')
                    
                    ax_psd_a.loglog(freqs_welch, np.sqrt(signal_psd_welch_dict['active'][elec]), lw=2, color='r')
                    ax_psd_h.loglog(freqs_welch, np.sqrt(signal_psd_welch_dict['Ih_linearized'][elec]), lw=2, color='r')
                    ax_psd_p.loglog(freqs_welch, np.sqrt(signal_psd_welch_dict['passive_vss'][elec]), lw=2, color='r')

                    ax_psd_p.set_xlim([1e0, 1e3])
                    if elec == 0:
                        ax_psd_a.set_ylabel('$\mu V$')
                        ax_psd_a.set_xlabel('Hz')

                fig.savefig('population_size_%s_%1.2f_%04d.png' %(input_pos, correlation, int(radius)))

def testplot():

    folder = 'stallo'

    xmid = np.load(join(folder, 'xmid.npy' ))
    ymid = np.load(join(folder, 'ymid.npy' ))
    xstart = np.load(join(folder, 'xstart.npy' ))
    ystart = np.load(join(folder, 'ystart.npy' ))
    zstart = np.load(join(folder, 'zstart.npy' ))
    xend = np.load(join(folder, 'xend.npy' ))
    yend = np.load(join(folder, 'yend.npy' ))
    zend = np.load(join(folder, 'zend.npy' ))    
    diam = np.load(join(folder, 'diam.npy'))
    n_elecs = len(elec_z)
    elec_clr = lambda elec_idx: plt.cm.rainbow(int(256. * elec_idx/(n_elecs - 1.)))
    
    for input_pos in ['dend']:
        for correlation in [0, 1.0]:
            signal_dict = {}
            signal_psd_dict = {}
            break_loop = False
            for conductance_type in ['active', 'passive_vss', 'Ih_linearized']:
                stem='signal_%s_%s_%1.2f_total' %(conductance_type, input_pos, correlation)
                try: 
                    signal_dict[conductance_type] = np.load(join(folder, '%s.npy' % stem))
                except:
                    break_loop = True
                    break
                signal_psd_dict[conductance_type], freqs = aLFP.find_LFP_PSD(signal_dict[conductance_type][:,:1000], (1.0)/1000.)
            if break_loop:
                continue

            plt.close('all')
            fig = plt.figure(figsize=[7,10])
            fig.suptitle('Stimulation: %s, Correlation: %1.2f'
                         % (input_pos, correlation))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)

            ax_ = fig.add_subplot(1, 3, 1, frameon=False, xticks=[], yticks=[], ylim=[-300, 1300])
            for comp in xrange(len(xstart)):
                if comp == 0:
                    ax_.scatter(xmid[comp], ymid[comp], s=diam[comp], color='k')
                else:
                    ax_.plot([xstart[comp], xend[comp]], [ystart[comp], yend[comp]], lw=diam[comp], color='k')
            
            for elec in xrange(n_elecs):

                ax_.plot(elec_x[elec], elec_z[elec], 'o', color=elec_clr(elec))
                ax_psd_a = fig.add_subplot(n_elecs, 4, 4*(n_elecs - elec - 1) + 2, ylim=[1e-4, 1e0])
                ax_psd_h = fig.add_subplot(n_elecs, 4, 4*(n_elecs - elec - 1) + 3, ylim=[1e-4, 1e0])
                ax_psd_p = fig.add_subplot(n_elecs, 4, 4*(n_elecs - elec - 1) + 4, ylim=[1e-4, 1e0])
                
                for ax in [ax_psd_a, ax_psd_h, ax_psd_p]:
                
                    ax.tick_params(color=elec_clr(elec))
                    for spine in ax.spines.values():
                        spine.set_edgecolor(elec_clr(elec))
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.get_xaxis().tick_bottom()
                    ax.get_yaxis().tick_left()
                    ax.grid(True)

                if elec == n_elecs - 1:
                    ax_psd_a.set_title('Active')
                    ax_psd_h.set_title('Ih_linearized')
                    ax_psd_p.set_title('Passive')

                ax_psd_a.loglog(freqs, signal_psd_dict['active'][elec], color='k')
                ax_psd_h.loglog(freqs, signal_psd_dict['Ih_linearized'][elec], color='k')
                ax_psd_p.loglog(freqs, signal_psd_dict['passive_vss'][elec], color='k')
                ax_psd_p.set_xlim([1e0, 1e3])
                if elec == 0:
                    ax_psd_a.set_ylabel('$\mu V$')
                    ax_psd_a.set_xlabel('Hz')

            fig.savefig('full_population_%s_%1.2f.png' %(input_pos, correlation))


def quick_sum():
    
    folder = 'hay'
    full_list = os.listdir(folder)    

    stems = [## 'signal_Ih_linearized_apic_1.00_sim_',
             ## 'signal_Ih_linearized_apic_0.10_sim_',
             ## 'signal_Ih_linearized_apic_0.01_sim_',
             ## 'signal_Ih_linearized_apic_0.00_sim_',
             ## 'signal_Ih_linearized_dend_1.00_sim_',
             ## 'signal_Ih_linearized_dend_0.10_sim_',
             ## 'signal_Ih_linearized_dend_0.01_sim_',
             ## 'signal_Ih_linearized_dend_0.00_sim_',
             'signal_Ih_linearized_homogeneous_1.00_sim_',
             'signal_Ih_linearized_homogeneous_0.10_sim_',
             'signal_Ih_linearized_homogeneous_0.01_sim_',
             'signal_Ih_linearized_homogeneous_0.00_sim_',
             ## 'signal_active_apic_1.00_sim_',
             ## 'signal_active_apic_0.10_sim_',
             ## 'signal_active_apic_0.01_sim_',
             ## 'signal_active_apic_0.00_sim_',
             ## 'signal_active_dend_1.00_sim_',
             ## 'signal_active_dend_0.10_sim_',
             ## 'signal_active_dend_0.01_sim_',
             ## 'signal_active_dend_0.00_sim_',
             'signal_active_homogeneous_1.00_sim_',
             'signal_active_homogeneous_0.10_sim_',
             'signal_active_homogeneous_0.01_sim_',
             'signal_active_homogeneous_0.00_sim_',
             ## 'signal_passive_vss_apic_1.00_sim_',
             ## 'signal_passive_vss_apic_0.10_sim_',
             ## 'signal_passive_vss_apic_0.01_sim_',
             ## 'signal_passive_vss_apic_0.00_sim_',
             ## 'signal_passive_vss_dend_1.00_sim_',
             ## 'signal_passive_vss_dend_0.10_sim_',
             ## 'signal_passive_vss_dend_0.01_sim_',
             ## 'signal_passive_vss_dend_0.00_sim_',
             'signal_passive_vss_homogeneous_1.00_sim_',
             'signal_passive_vss_homogeneous_0.10_sim_',
             'signal_passive_vss_homogeneous_0.01_sim_',
             'signal_passive_vss_homogeneous_0.00_sim_',
             ]
             
    for stem in stems:
        f_list = [f for f in full_list if stem in f]
        f_list.sort()
        if len(f_list) < 500:
            print "Skipping %s ..." %stem
            continue
        summed_sig = np.zeros((8,1001))

        numbs = []
        for f in f_list:
            numb = int(f.split('_')[-1][:-4])
            numbs.append(numb)
            if numb > 1463:
                continue
            summed_sig += np.load(os.path.join(folder, f))
        print stem, len(f_list)
        np.save('%stotal.npy' %(stem), summed_sig)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python %s <function-name> <additional arguments>\n" % sys.argv[0])
        raise SystemExit(1)
    func = eval('%s' % sys.argv[1])
    func()
