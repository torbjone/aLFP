import numpy as np
import scipy.fftpack as ff
import pylab as pl


def return_r_m_tilde(cell):
    """ Returns renormalized membrane potential """
    r_tilde = (cell.somav[-1] - cell.somav[0])/\
              (cell.imem[0,-1] - cell.imem[0, 0]) * cell.area[0] * 10**-2
    return r_tilde


def make_WN_input(neural_sim_dict):
    """ White Noise input ala Linden 2010 is made """
    tot_ntsteps = round((neural_sim_dict['tstopms'] + neural_sim_dict['cut_off'])/\
                  neural_sim_dict['timeres'] + 1)
    I = np.zeros(tot_ntsteps)
    tvec = np.arange(tot_ntsteps) * neural_sim_dict['timeres']
    for freq in xrange(1,1001):
        
        I += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())
    I /= np.std(I)
    
    if 0:
        pl.subplot(211)
        pl.plot(tvec, I)
        pl.subplot(212)
        freqs, power = return_psd(I, neural_sim_dict)
        pl.loglog(freqs, power)
        pl.xlim(1,1000)
        pl.show()
    return I

def return_time_const(cell):
    start_t_idx = np.argmin(np.abs(cell.tvec - input_delay))
    v = cell.somav[:] - cell.somav[0]
    idx = np.argmin(np.abs(v - 0.63*v[-1]))
    print cell.tvec[idx] - cell.tvec[start_t_idx]
    return cell.tvec[idx] - cell.tvec[start_t_idx]

def norm_it(sig):
    return (sig - sig[0])

def return_dipole_stick(cell, ax):
    n_points = 20
    stick = np.linspace(ax.axis()[2], ax.axis()[3], n_points)
    stick_values = np.zeros((n_points, len(cell.tvec)))
    #print np.min(cell.ymid), np.max(cell.ymid)
    #print stick
    for comp in xrange(cell.totnsegs):
        idx = np.argmin(np.abs(stick - cell.ymid[comp]))
        #print stick[idx], cell.ymid[comp]
        stick_values[idx] += cell.imem[comp]
    return stick_values[::-1] # Reverse array


def make_white_noise(N):
    return np.random.random(N)

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
