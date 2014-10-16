__author__ = 'torbjone'

from os.path import join
import sys
import neuron
from neuron import h as nrn
import LFPy
import numpy as np
import pylab as plt


def insert_debug():

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.e_pas = -80
        sec.g_pas = 1./30000
        sec.Ra = 150
        sec.cm = 1
        sec.insert("debug_BG")
        sec.gbar_debug_BG = 1.0


def plot_dynamics():


    e_rev = 30
    vhalfn = -47.
    z = 6.5
    gamma = 0.5
    tau0 = 1.
    K = 0.0
    R = 8315.
    F = 9.648e4
    celsius = 33
    v = np.linspace(-100, 0, 1000)

    alpn = np.exp(z * gamma * (v-vhalfn) * F / (R * (273.16+celsius)))
    betn = np.exp(-z * (1 - gamma) * (v-vhalfn) * F / (R * (273.16+celsius)))

    ninf = alpn / (alpn + betn)

    plt.plot(v, ninf)

    plt.plot(v, alpn)
    plt.plot(v, betn)
    plt.show()

if __name__ == '__main__':

    timeres = 2**-6
    tstopms = 100
    tstartms = 0
    model_path = '.'

    cell_params = {
        'morphology': join('..', 'example_morphology.hoc'),
        #'rm' : 30000,               # membrane resistance
        #'cm' : 1.0,                 # membrane capacitance
        #'Ra' : 100,                 # axial resistance
        'v_init': -70,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': timeres,
        'tstartms': tstartms,          # start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_fun': [insert_debug],  # will execute this function
        'custom_fun_args': [{}],
    }

    cell = LFPy.Cell(**cell_params)

    sim_params = {'rec_vmem': True,
                  'rec_imem': True,
                  'rec_variables': ['n_debug_BG', 'ninf_debug_BG', 'taun_debug_BG',
                                    'alpn_debug_BG', 'betn_debug_BG']}

    cell.simulate(**sim_params)


    # [plt.plot(cell.tvec, cell.vmem[idx, :]) for idx in xrange(len(cell.xmid))]
    # [plt.plot(cell.tvec, cell.rec_variables['gna_INaP_BK'][idx, :]) for idx in xrange(len(cell.xmid))
    #
    #                  if 'gna_INaP_BK' in cell.rec_variables]
    print cell.rec_variables['n_debug_BG']

    plt.figure(figsize=[12,8])
    plt.subplot(131)
    [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], lw=2, color='k')
                for idx in xrange(cell.totnsegs)]

    plt.subplot(232, title='n', xlim=[-5, 100])
    plt.plot(cell.tvec, cell.rec_variables['n_debug_BG'][0,:])
    plt.subplot(233, title='n_inf', xlim=[-5, 100])
    plt.plot(cell.tvec, cell.rec_variables['ninf_debug_BG'][0,:])
    plt.subplot(235, title='alpha and beta', xlim=[-5, 100])
    plt.plot(cell.tvec, cell.rec_variables['alpn_debug_BG'][0,:])
    plt.plot(cell.tvec, cell.rec_variables['betn_debug_BG'][0,:])
    plt.subplot(236, title='tau_n', xlim=[-5, 100])
    plt.plot(cell.tvec, cell.rec_variables['taun_debug_BG'][0,:])

    #plt.ylim(-100, 100)
    plt.show()
