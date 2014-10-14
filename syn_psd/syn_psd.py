__author__ = 'torbjone'

import numpy as np
import pylab as plt
import matplotlib.mlab as mlab
import aLFP


end_t = 2000
timeres_python = 2**-4
num_tsteps = round(end_t/timeres_python + 1)
tvec = np.arange(num_tsteps) * timeres_python
divide_into_welch = 8.
welch_dict = {'Fs': 1000 / 2**-4,
                           'NFFT': int(num_tsteps/divide_into_welch),
                           'noverlap': int(num_tsteps/divide_into_welch/2.),
                           'window': plt.window_hanning,
                           'detrend': plt.detrend_mean,
                           'scale_by_freq': True,
                           }

idx = 0
syni = np.load('syn.npy')
average_psd = np.zeros(10001)

def plot_one_idx(idx):
    freqs, sig_psd = aLFP.return_freq_and_psd(timeres_python/1000., syni[idx, :])
    sig_psd_welch, freqs_welch = mlab.psd(syni[idx, :], **welch_dict)
    plt.close('all')
    plt.subplot(211)
    plt.plot(tvec, syni[idx, :])
    plt.subplot(212)
    plt.loglog(freqs, sig_psd[0])
    plt.loglog(freqs_welch, np.sqrt(sig_psd_welch))
    plt.savefig('syn_psd_%d.png' % idx)

def plot_psd_sig_combined(sig, idx, average_psd):
    num_tsteps = len(sig)
    welch_dict = {'Fs': 1000 / 2**-4,
                  'NFFT': int(num_tsteps/divide_into_welch),
                  'noverlap': int(num_tsteps/divide_into_welch/2.),
                  'window': plt.window_hanning,
                  'detrend': plt.detrend_mean,
                  'scale_by_freq': True,
                  }

    freqs, sig_psd = aLFP.return_freq_and_psd(timeres_python/1000., sig)
    sig_psd_welch, freqs_welch = mlab.psd(sig, **welch_dict)
    sig_psd_welch = np.sqrt(sig_psd_welch)
    average_psd += sig_psd_welch

    plt.close('all')
    plt.figure(figsize=[10, 5])
    plt.subplot(121, xlabel='ms', ylabel='Synaptic current', ylim=[-0.15, 0.02], xlim=[0, 10000])
    plt.plot(np.arange(len(sig)) * timeres_python, sig)
    plt.xticks([0, 2500, 5000, 7500, 10000])
    plt.subplot(122, ylim=[1e-7, 1e-3], xlim=[1e-1, 1e4], xlabel='Hz', ylabel='PSD')
    plt.grid(True)
    plt.loglog(freqs, sig_psd[0], c='gray', lw=0.5)
    plt.loglog(freqs_welch, sig_psd_welch, 'r')

    plt.savefig('syn2_%d.png' % idx)
    return average_psd, freqs_welch

# [plot_one_idx(idx) for idx in [1,2,3,4,5,6,7,10]]

combine_number = 5
for idx in xrange(20):
    combine_idx = combine_number * idx
    sig = np.r_[syni[combine_idx, :], syni[combine_idx + 1, :], syni[combine_idx + 2, :],
                syni[combine_idx + 3, :], syni[combine_idx + 4, :]]
    average_psd, freqs_average = plot_psd_sig_combined(sig, idx, average_psd)

# plt.close('all')
# plt.loglog(freqs_average, average_psd / 20)
# plt.show()