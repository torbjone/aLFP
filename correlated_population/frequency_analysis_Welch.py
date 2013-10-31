import matplotlib.mlab as mlab
import os
from os.path import join
import numpy as np
import pylab as plt
import scipy.fftpack as ff
from ipdb import set_trace

def find_LFP_PSD(sig, timestep):
    """ Returns the power and freqency of the input signal"""
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig, axis=1)[:,pidxs[0]]
    power = np.abs(Y)/Y.shape[1]
    return power, freqs


def find_LFP_PSD_simple(sig, timestep):
    """ Returns the power and freqency of the input signal"""
    sample_freq = ff.fftfreq(len(sig), d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig)[:,pidxs[0]]
    power = np.abs(Y)/len(Y)
    return power, freqs


def simplest_test():

    npts = 1001
    t = np.linspace(0, 1, npts)
    timestep = t[1] - t[0]

    
    sig = np.sum([np.sin(2*np.pi*f*t + 2*np.pi*np.random.random()) for f in np.arange(100, 200)], axis=0) + np.random.normal(0,2, size=npts)
    sig /= np.std(sig)
    divide_into = 8
    sig_psd, freqs = find_LFP_PSD_simple(sig, timestep)
    welch_psd, freqs_welch = mlab.psd(sig, Fs=(1./timestep), NFFT=int(npts/divide_into), noverlap=int(npts/divide_into/2),
                                      window=plt.window_hanning, detrend=plt.detrend_mean)
    
    plt.close('all')
    fig = plt.figure(figsize=[7,5])
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, xlim=[0,300])

    ax1.set_title('Time domain')
    ax2.set_title('Frequency domain')

    ax1.plot(t, sig)
    ax2.plot(freqs, sig_psd, 'g', label='FFT')
    ax2.plot(freqs_welch, np.sqrt(welch_psd), 'rx-', label='Welch')
    plt.legend(bbox_to_anchor=[1.1, 1.2])
    plt.savefig('simplest_test.png', dpi=150)

def main():
    folder = 'stallo'

    n_elecs = 8
    timestep = 1./1000
    npts = 1001
    tvec = np.linspace(0,1., npts)

    conductance_type = 'Ih_linearized'
    sig = np.load(join(folder, 'signal_%s_dend_1.00_total_pop_size_1000.npy' % conductance_type))
    sig_psd, freqs = find_LFP_PSD(sig[:,:], timestep)
    
    n_plot_cols = 2
    plt.close('all')
    fig = plt.figure(figsize=[5,10])
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    divide_into = 8
    for elec in xrange(n_elecs):
        plot_number = n_elecs - elec - 1
        ax1 = fig.add_subplot(n_elecs, n_plot_cols, n_plot_cols*(n_elecs - elec - 1) + 1)
        ax2 = fig.add_subplot(n_elecs, n_plot_cols, n_plot_cols*(n_elecs - elec - 1) + 2, xlim=[1e0, 1e3], ylim=[1e-4, 1e2])

        ax2.grid(True)
        if elec == n_elecs - 1:
            ax1.set_title('Time domain')
            ax2.set_title('FFT analysis')

        ax1.plot(tvec, sig[elec,:])
        ax2.loglog(freqs, sig_psd[elec,:])
        welch_psd, freqs_welch = mlab.psd(sig[elec,:], Fs=1./timestep, NFFT=int(npts/divide_into),
                                          noverlap=int(npts/divide_into/2), detrend=plt.detrend_mean, window=plt.window_hanning)
        ax2.loglog(freqs_welch, np.sqrt(welch_psd), 'r')

    plt.savefig('frequency_analysis_test_%s.png' % conductance_type, dpi=150)


    
main()
#simplest_test()
