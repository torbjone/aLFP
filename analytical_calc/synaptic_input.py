import numpy as np
import pylab as plt
from scipy.integrate import quad
from ipdb import set_trace

class SynapticInput:
    """ Analytical solving of linearized cable equation for synaptic input"""
    def __init__(self, tau_W, mu, gamma_R, lambd, w_syn, R_inf):

        self.tau_W = tau_W
        self.tau = tau_W * 2
        self.tau_syn = 0.2 * self.tau
        self.mu = mu
        self.gamma_R = gamma_R
        self.lambd = lambd
        self.w_syn = w_syn
        self.R_inf = R_inf
        
    def V(self, x, t):

        return self.R_inf * self.w_syn * self.tau_syn * np.exp(1) / (2*np.pi) * self.integrate_input(x, t)[0]
        
    def b(self, omega):
        re = self.gamma_R + self.mu / (1 + (self.tau_W * omega)**2)
        im = omega * ( self.tau - self.mu * self.tau_W / (1 + (self.tau_W * omega)**2))
        return np.sqrt(complex(re, im))

    def integrand(self, omega, x, t):
        return np.exp(-self.b(omega) * x / self.lambd) / self.b(omega) * \
          (complex(1, self.tau_syn * omega))**(-2) * np.exp(complex(0,omega*t))

    def integrate_input(self, x, t):
        return quad(self.integrand, -np.Inf, np.Inf, args=(x, t))

diam = 2 # mu m
R_a = 1 # Ohm cm
lambd = 1 # mu m
R_m = 1
R_inf = 1#np.sqrt(R_m * R_a / (np.pi**2 * diam**3))

conductance_pas = {'mu': 0.,
                   'tau_W': 1.,
                   'gamma_R': 2.,
                   'w_syn': 1.,
                   'lambd': lambd,
                   'R_inf': R_inf,
                   }

conductance_NaP = {'mu': -1.,
                   'tau_W': 1.,
                   'gamma_R': 2.,
                   'w_syn': 1.,
                   'lambd': lambd,   
                   'R_inf': R_inf,                   
                   }

conductance_KLT = {'mu': 4.,
                   'tau_W': 1.0,
                   'gamma_R': 2.,
                   'w_syn': 1.,
                   'lambd': lambd,         
                   'R_inf': R_inf,                   
                   }

pas = SynapticInput(**conductance_pas)
NaP = SynapticInput(**conductance_NaP)
KLT = SynapticInput(**conductance_KLT)


tvec = np.linspace(0,10,100)
values_pas = np.zeros(len(tvec))
values_NaP = np.zeros(len(tvec))
values_KLT = np.zeros(len(tvec))


for pos in [0, 1, 2]:
    for tstep in xrange(len(tvec)): 
        values_pas[tstep] = pas.V(pos, tvec[tstep])
        values_NaP[tstep] = NaP.V(pos, tvec[tstep])
        values_KLT[tstep] = KLT.V(pos, tvec[tstep])
    plt.subplot(231)
    plt.plot(tvec, values_pas, 'g')
    plt.subplot(232)
    plt.plot(tvec, values_KLT, 'b')
    plt.subplot(233)
    plt.plot(tvec, values_NaP, 'r')

    plt.subplot(234)
    plt.plot(tvec, values_pas/np.max(values_pas), 'g', label='Passive')
    plt.legend()
    plt.subplot(235)
    plt.plot(tvec, values_KLT/np.max(values_KLT), 'b', label='4')
    plt.legend()
    plt.subplot(236)
    plt.plot(tvec, values_NaP/np.max(values_NaP), 'r', label='-1')
    
    plt.legend()

 
plt.show()
