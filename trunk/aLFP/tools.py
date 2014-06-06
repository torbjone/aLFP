import numpy as np
import scipy.fftpack as ff
import pylab as pl
from matplotlib import mlab

def return_r_m_tilde(cell):
    """ Returns renormalized membrane potential """
    r_tilde = (cell.somav[-1] - cell.somav[0])/\
              (cell.imem[0,-1] - cell.imem[0, 0]) * cell.area[0] * 10**-2
    return r_tilde


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
#
# def return_time_const(cell):
#     start_t_idx = np.argmin(np.abs(cell.tvec - input_delay))
#     v = cell.somav[:] - cell.somav[0]
#     idx = np.argmin(np.abs(v - 0.63*v[-1]))
#     print cell.tvec[idx] - cell.tvec[start_t_idx]
#     return cell.tvec[idx] - cell.tvec[start_t_idx]

def norm_it(sig):
    return (sig - sig[0])

def return_dipole_stick(imem, ymid):
    """ Sums imem contriburions along y-axis for imshow plotting"""
    n_points = 20
    stick = np.linspace(np.min(ymid), np.max(ymid), n_points)
    stick_values = np.zeros((n_points, imem.shape[1]))
    for comp in xrange(imem.shape[0]):
        idx = np.argmin(np.abs(stick - ymid[comp]))
        stick_values[idx] += imem[comp]
    return stick_values[::-1] # Reverse array


def make_white_noise(N):
    return np.random.random(N)

def return_freq_and_psd(tvec, sig):
    """ Returns the power and freqency of the input signal"""
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
    power = np.abs(Y)/Y.shape[1]
    return freqs, power

def return_freq_and_psd_welch(sig, welch_dict):
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
    return freqs, np.sqrt(np.array(psd))


def return_psd(sig, neural_sim_params):

    sample_freq = ff.fftfreq(len(sig), d=neural_sim_params['timeres']/1000)
    pidxs = np.where(sample_freq > 0)
    freqs = sample_freq[pidxs]

    Y = ff.fft(sig)[pidxs]
    power = np.abs(Y)/len(Y)
    #offset = (np.abs(ff.fft(sig))/len(ff.fft(sig)))[0]
    freqIndex = power[:].argmax()
    freq = freqs[freqIndex]
    #amp = power[freqIndex]
    #amps[numb] = amp
    #phase = np.angle(Y[freqIndex], deg=1)
    #print freq, amp, phase, offset
    return freqs, power
