__author__ = 'torbjone'

from os.path import join
import neuron
import LFPy


def insert_Ih_prox():

    for sec in neuron.h.apic:
        sec.insert("Ih_BK")
        sec.ghbar_Ih_BK = 1.

        #for seg in sec:
        #    xdist = neuron.h.distance(seg.x)
        #    seg.ghdbar_hd = ghd * (1+3.*xdist/100)
        #    if xdist > 100:
        #        seg.gkabar_kad = ka * (1+xdist/100)
        #    else:
        #        seg.gkabar_kap = ka * (1+xdist/100)


def insert_km():

    ka = 1
    gkm = 1

    for sec in neuron.h.axon:
        sec.insert("km")
        sec.gbar_km = gkm

    for sec in neuron.h.soma:
            sec.insert("km")
            sec.gbar_km = gkm

    for sec in neuron.h.apical_dendrite:
        for seg in sec:
            xdist = neuron.h.distance(seg.x)
            if xdist > 100:
                seg.gkabar_kad = ka * (1+xdist/100)
            else:
                seg.gkabar_kap = ka * (1+xdist/100)


def init(Vrest):
    neuron.h.t = 0

    neuron.h.finitialize(Vrest)
    neuron.h.fcurrent()

    for sec in neuron.h.allsec():
        for seg in sec:
            seg.e_pas = seg.v
            if neuron.h.ismembrane("na_ion"):
                seg.e_pas += seg.ina/seg.g_pas
            if neuron.h.ismembrane("k_ion"):
                seg.e_pas += seg.ik/seg.g_pas
            if neuron.h.ismembrane("Ih_BK"):
                seg.e_pas += seg.ih_Ih_BK/seg.g_pas
            if neuron.h.ismembrane("ca_ion"):
                seg.e_pas += seg.ica/seg.g_pas


def biophys_active(**kwargs):

    Vrest = -80 if not 'hold_potential' in kwargs else kwargs['hold_potential']

    Rm = 90000.
    Cm = 1.5
    Ra = 100.

    # for sec in neuron.h.axon:
    #     sec.insert('pas')
    #     sec.e_pas = Vrest
    #     sec.g_pas = 1/Rm
    #     sec.Ra = Ra
    #     sec.cm = Cm

    for sec in neuron.h.soma:
        sec.insert("pas")
        sec.e_pas = Vrest
        sec.g_pas = 1./Rm
        sec.Ra = Ra
        sec.cm = Cm

    for sec in neuron.h.dend:
        sec.insert("pas")
        sec.e_pas = Vrest
        sec.g_pas = 1./Rm
        sec.Ra = Ra
        sec.cm = Cm

    for sec in neuron.h.apic:
        sec.insert("pas")
        sec.e_pas = Vrest
        sec.g_pas = 1./Rm
        sec.Ra = Ra
        sec.cm = Cm

    #neuron.h.distance()
    insert_Ih_prox()
    #init(Vrest)


def active_declarations(**kwargs):
    ''' Set active conductances for modified CA1 cell'''

    #neuron.h.geom_nseg()
    #neuron.h.define_shape()
    exec('biophys_%s(**kwargs)' % kwargs['conductance_type'])


if __name__ == '__main__':

    import pylab as plt
    timeres = 2**-4
    cut_off = 0
    tstopms = 1000
    tstartms = -cut_off
    model_path = '.'

    cell_params = {
        'morphology': join(model_path, 'n120.hoc'),
        #'rm' : 30000,               # membrane resistance
        #'cm' : 1.0,                 # membrane capacitance
        #'Ra' : 100,                 # axial resistance
        'v_init': -60,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': timeres,
        'tstartms': tstartms,          # start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'conductance_type': 'active'}],
    }

    cell = LFPy.Cell(**cell_params)
    cell.simulate(rec_vmem=True, rec_imem=True)
    plt.subplot(121)
    plt.scatter(cell.xmid, cell.ymid, c=cell.vmem[:, -1])
    print cell.vmem
    plt.colorbar()
    plt.subplot(122)
    plt.plot(cell.tvec, cell.somav)
    plt.show()