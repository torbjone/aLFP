import numpy as np
import pylab as plt 

import scipy.fftpack as ff

npts = 10000

x = np.linspace(0,1,npts)
y = np.zeros(npts)
y[len(y)/2] = 1
plt.close('all')

sample_freq = ff.fftfreq(x.shape[0], d=(x[1] - x[0]))
pidxs = np.where(sample_freq >= 0)
freqs = sample_freq[pidxs]
Y = ff.fft(y)[pidxs]
amp = np.abs(Y)/len(Y)

plt.plot(freqs, amp)
plt.show()

angle = np.angle(Y)


fig = plt.figure()

ax1 = fig.add_subplot(121)
ax1.plot(x,y, lw=1, color='k')

ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
ax2.set_rasterization_zorder(1)

fy = np.zeros(len(x))

for idx, freq in enumerate(freqs):
    foo = amp[idx] * np.cos(2*np.pi*freq * x + angle[idx])
    fy += foo
    #ax2.plot(x, fy, lw=0.1, color='grey')

ax2.plot(x, fy, lw=1, color='k')




fig.show()
