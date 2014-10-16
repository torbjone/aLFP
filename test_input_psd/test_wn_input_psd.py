__author__ = 'torbjone'

import numpy as np
import pylab as plt
import aLFP

plt.seed(1234)

repeats = 5

tstopms = 1000 * repeats
tstartms = 0
timeres_NEURON = 2**-4
max_freq = 500
tot_ntsteps = round((tstopms - tstartms)/timeres_NEURON)
tvec = np.arange(tot_ntsteps) * timeres_NEURON + tstartms

input_freqs = range(1, max_freq + 1)

wn = np.zeros(tot_ntsteps)
for freq in input_freqs:
    wn += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())

plt.subplot(121, xlim=[-500, 5500])
plt.plot(tvec, wn)

wn = wn[np.where(tvec >= 0)]
freqs, sig_psd = aLFP.return_freq_and_psd(timeres_NEURON/1000., wn)

plt.subplot(122, ylim=[1e-1, 1e1])
plt.grid(True)
plt.loglog(freqs, sig_psd[0])
plt.loglog(freqs[:len(input_freqs)*repeats + 1:repeats],
           sig_psd[0, :len(input_freqs)*repeats + 1:repeats], 'r')

plt.savefig('wn_test.png')