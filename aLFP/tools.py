import numpy as np
import scipy.fftpack as ff
import pylab as pl
from matplotlib import mlab


def make_WN_input(cell_params):
    """ White Noise input ala Linden 2010 is made """
    tot_ntsteps = round((cell_params['tstopms'])/\
                  cell_params['timeres_NEURON'] + 1)
    I = np.zeros(tot_ntsteps)
    tvec = np.arange(tot_ntsteps) * cell_params['timeres_NEURON']
    #I = np.random.random(tot_ntsteps) - 0.5
    for freq in xrange(1,1001):
        I += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())
    I /= np.std(I)
    return I

def return_dipole_stick(imem, ymid):
    """ Sums imem contriburions along y-axis for imshow plotting"""
    n_points = 20
    stick = np.linspace(np.min(ymid), np.max(ymid), n_points)
    stick_values = np.zeros((n_points, imem.shape[1]))
    for comp in xrange(imem.shape[0]):
        idx = np.argmin(np.abs(stick - ymid[comp]))
        stick_values[idx] += imem[comp]
    return stick_values[::-1] # Reverse array


def return_freq_and_psd(tvec, sig):
    """ Returns the frequency and power of the input signal"""
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    timestep = (tvec[1] - tvec[0])/1000. if type(tvec) in [list, np.ndarray] else tvec
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig, axis=1)[:, pidxs[0]]
    power = np.abs(Y)**2/Y.shape[1]
    return freqs, power

def return_freq_and_fft(tvec, sig):
    """ Returns the frequency and amplitude of the input signal"""
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    timestep = (tvec[1] - tvec[0])/1000. if type(tvec) in [list, np.ndarray] else tvec
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig, axis=1)[:, pidxs[0]]
    amplitude = np.abs(Y)/Y.shape[1]
    return freqs, amplitude


def return_freq_and_angle(tvec, sig):
    """ Returns the frequency and angle (phase) of the input signal"""
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    timestep = (tvec[1] - tvec[0])/1000. if type(tvec) in [list, np.ndarray] else tvec
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig, axis=1)[:, pidxs[0]]
    #power = np.abs(Y)**2/Y.shape[1]
    phase = np.angle(Y, deg=1)
    return freqs, phase

def return_freq_and_psd_welch(sig, welch_dict):
    """ Returns the frequency and power of the input signal using Welch's average method"""
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    psd = []
    freqs = None
    for idx in xrange(sig.shape[0]):
        yvec_w, freqs = mlab.psd(sig[idx, :], **welch_dict)
        psd.append(yvec_w)
    return freqs, np.array(psd)

