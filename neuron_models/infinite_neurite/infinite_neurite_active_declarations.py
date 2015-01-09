

import neuron
import numpy as np

def active_declarations(**kwargs):
    neuron.h.distance(0, 0)
    for sec in neuron.h.allsec():
        sec.nseg = 101
        sec.insert("QA")
        sec.V_r_QA = -80.
        v = -80.
        mAlpha = 0.001 * 6.43 * (v + 154.9)/(np.exp((v + 154.9) / 11.9) - 1.)
        mBeta = 0.001 * 193. * np.exp(v / 33.1)
        tau_w = 1/(mAlpha + mBeta)


        sec.tau_w_QA = tau_w
        sec.Ra = 150.
        sec.cm = 1.0
        sec.g_pas_QA = 0.0001
        sec.g_w_QA = 0.0001 * 5
        for seg in sec:
            if neuron.h.distance(seg.x) <= 50:
                seg.mu_QA = sec.g_w_QA / sec.g_pas_QA * kwargs['mu_factor_1']
            else:
                seg.mu_QA = sec.g_w_QA / sec.g_pas_QA * kwargs['mu_factor_2']
