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
        self.synaptic_input = lambda t: self.w_syn / self.tau_syn * t * np.exp(1 - t / self.tau_syn)

    def Vm(self, x, t):
        return self.R_inf * self.w_syn * self.tau_syn * np.exp(1) / (2*np.pi) * self.integrate_input(x, t, self.integrand_Vm)[0]

    def Im(self, x, t):
        return self.R_inf * self.w_syn * self.tau_syn * np.exp(1) / (2*np.pi) * self.integrate_input(x, t, self.integrand_Im)[0]

    def b(self, omega):
        re = self.gamma_R + self.mu / (1 + (self.tau_W * omega)**2)
        im = omega * ( self.tau - self.mu * self.tau_W / (1 + (self.tau_W * omega)**2))
        return np.sqrt(complex(re, im))

    def integrand_Vm(self, omega, x, t):
        return np.exp(-self.b(omega) * x / self.lambd) / self.b(omega) * \
          (complex(1, self.tau_syn * omega))**(-2) * np.exp(complex(0,omega*t))

    def integrand_Im(self, omega, x, t):
        return np.exp(-self.b(omega) * x / self.lambd) * self.b(omega) * \
          (complex(1, self.tau_syn * omega))**(-2) * np.exp(complex(0,omega*t))
          
    def integrate_input(self, x, t, integrand_func):
        return quad(integrand_func, -np.Inf, np.Inf, args=(x, t))

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
Vm_pas = np.zeros(len(tvec))
Vm_NaP = np.zeros(len(tvec))
Vm_KLT = np.zeros(len(tvec))

Im_pas = np.zeros(len(tvec))
Im_NaP = np.zeros(len(tvec))
Im_KLT = np.zeros(len(tvec))

plt.figure(figsize=[10,10])
for pos in [0, 1, 2]:
    for tstep in xrange(len(tvec)): 
        Vm_pas[tstep] = pas.Vm(pos, tvec[tstep])
        Vm_NaP[tstep] = NaP.Vm(pos, tvec[tstep])
        Vm_KLT[tstep] = KLT.Vm(pos, tvec[tstep])
        if pos == 0:
            Im_pas[tstep] = pas.Im(pos, tvec[tstep]) - pas.synaptic_input(tvec[tstep])
            Im_NaP[tstep] = NaP.Im(pos, tvec[tstep]) - NaP.synaptic_input(tvec[tstep])
            Im_KLT[tstep] = KLT.Im(pos, tvec[tstep]) - KLT.synaptic_input(tvec[tstep])

        else:
            Im_pas[tstep] = pas.Im(pos, tvec[tstep])
            Im_NaP[tstep] = NaP.Im(pos, tvec[tstep])
            Im_KLT[tstep] = KLT.Im(pos, tvec[tstep])

            
    plt.subplot(331)
    plt.plot(tvec, Vm_pas, 'g')
    plt.subplot(332)
    plt.plot(tvec, Vm_KLT, 'b')
    plt.subplot(333)
    plt.plot(tvec, Vm_NaP, 'r')

    plt.subplot(334)
    plt.plot(tvec, Vm_pas/np.max(Vm_pas), 'g')
    plt.legend()
    plt.subplot(335)
    plt.plot(tvec, Vm_KLT/np.max(Vm_KLT), 'b')
    plt.legend()
    plt.subplot(336)
    plt.plot(tvec, Vm_NaP/np.max(Vm_NaP), 'r')

    plt.subplot(337)
    plt.plot(tvec, Im_pas, 'g', label='0')
    plt.legend()
    plt.subplot(338)
    plt.plot(tvec, Im_KLT, 'b', label='4')
    plt.legend()
    plt.subplot(339)
    plt.plot(tvec, Im_NaP, 'r', label='-1')
    plt.legend()

 
plt.savefig('remme_reproduced.png')
