__author__ = 'torbjone'

from os.path import join
import sys
import neuron
import LFPy
import numpy as np
import pylab as plt


def insert_Ih(apic_trunk, basal, apic_tuft):

    for sec in basal:
        sec.insert("Ih_BK_prox")
        sec.ghbar_Ih_BK_prox = 0.2

    neuron.h.distance()
    for sec in apic_trunk:
        sec.insert("Ih_BK_prox")
        for seg in sec:
            dist = neuron.h.distance(seg.x)
            seg.ghbar_Ih_BK_prox = 0.2 + (2.0 - 0.2)/(1 + np.exp((250. - dist)/30))

    for sec in apic_tuft:
        sec.insert("Ih_BK_dist")
        sec.ghbar_Ih_BK_dist = 20.

        #for seg in sec:
        #    xdist = neuron.h.distance(seg.x)
        #    seg.ghdbar_hd = ghd * (1+3.*xdist/100)
        #    if xdist > 100:
        #        seg.gkabar_kad = ka * (1+xdist/100)
        #    else:
        #        seg.gkabar_kap = ka * (1+xdist/100)


def insert_km(apic_trunk, basal, apic_tuft):

    ka = 1
    gkm = 1

    for sec in neuron.h.axon:
        sec.insert("km")
        sec.gbar_km = gkm

    for sec in neuron.h.somatic:
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


def make_section_lists():

    apic_trunk = neuron.h.SectionList()
    basal = neuron.h.SectionList()
    apic_tuft = neuron.h.SectionList()
    #oblique_dendrites = neuron.h.SectionList()

    for sec in neuron.h.allsec():
        sec_type = sec.name().split('[')[0]
        sec_idx = int(sec.name().split('[')[1][:-1])
        if sec_type == 'dend':
            basal.append(sec)
        elif sec_type == 'apic' and sec_idx > 0:
            apic_tuft.append(sec)
        elif sec_type == 'apic' and sec_idx == 0:
            apic_trunk.append(sec)
    return apic_trunk, basal, apic_tuft


def modify_morphology(apic_trunk, basal, apic_tuft):

    for sec in basal:
        sec.diam = 0.76

    neuron.h.distance()
    apic_tuft_root_diam = None
    apic_tuft_root_dist = None
    for sec in apic_trunk:
        npts = int(neuron.h.n3d())
        for i in xrange(npts):
            dist_calc = np.sqrt(neuron.h.x3d(i)**2 + neuron.h.y3d(i)**2 + neuron.h.z3d(i)**2)
            diam = 3.5 - 4.7e-3 * dist_calc
            neuron.h.pt3dchange(i, neuron.h.x3d(i), neuron.h.y3d(i), neuron.h.z3d(i), diam)
        apic_tuft_root_diam = neuron.h.diam3d(npts - 1)
        apic_tuft_root_dist = np.sqrt(neuron.h.x3d(i)**2 + neuron.h.y3d(i)**2 + neuron.h.z3d(i)**2)

    if apic_tuft_root_diam == None:
        raise RuntimeError, "Found no apic_tuft_root_diam"
    neuron.h.distance()

    # TODO: THE FOLLOWING MAKES NO SENSE AT THE MOMENT
    for sec in apic_tuft:
        npts = int(neuron.h.n3d())
        print sec.name(), npts
        for i in xrange(npts):
            dist_calc = np.sqrt(neuron.h.x3d(i)**2 + neuron.h.y3d(i)**2 + neuron.h.z3d(i)**2)
            dist_from_root = np.abs(dist_calc - apic_tuft_root_dist)
            #print dist_from_root, dist_calc
            diam = apic_tuft_root_diam - 18e-3 * (dist_from_root)
            # neuron.h.pt3dchange(i, neuron.h.x3d(i), neuron.h.y3d(i), neuron.h.z3d(i), diam)
    return apic_tuft_root_diam

def biophys_passive(apic_trunk, basal, apic_tuft, **kwargs):
    Vrest = -80 if not 'hold_potential' in kwargs else kwargs['hold_potential']

    rm = 90000.
    rm_apic_tuft = 20000.
    rm_myel_ax = 1e9

    cm = 1.5
    cm_myel_ax = 0.04

    ra = 100.

    for sec in neuron.h.axon_hillock:
         sec.insert('pas')
         sec.e_pas = Vrest
         sec.g_pas = 1./rm
         sec.Ra = ra
         sec.cm = cm

    for sec in neuron.h.axon_IS:
         sec.insert('pas')
         sec.e_pas = Vrest
         sec.g_pas = 1./rm
         sec.Ra = ra
         sec.cm = cm

    for sec in neuron.h.myelinated_axon:
         sec.insert('pas')
         sec.e_pas = Vrest
         sec.g_pas = 1./rm_myel_ax
         sec.Ra = ra
         sec.cm = cm_myel_ax

    for sec in neuron.h.somatic:
        sec.insert("pas")
        sec.e_pas = Vrest
        sec.g_pas = 1./rm
        sec.Ra = ra
        sec.cm = cm

    for sec in apic_tuft:

        sec.insert("pas")
        sec.e_pas = Vrest
        sec.g_pas = 1./rm_apic_tuft
        sec.Ra = ra
        sec.cm = cm

    for sec in basal:
        sec.insert("pas")
        sec.e_pas = Vrest
        sec.g_pas = 1./rm
        sec.Ra = ra
        sec.cm = cm

    for sec in apic_trunk:
        sec.insert("pas")
        sec.e_pas = Vrest
        sec.g_pas = 1./rm
        sec.Ra = ra
        sec.cm = cm

def create_axon():
    neuron.h('''
        create axon_hillock[4], axon_IS[1], myelinated_axon[1]
        connect axon_hillock[0](0), soma(0.5)
        connect axon_hillock[1](0), axon_hillock[0](1)
        connect axon_hillock[2](0), axon_hillock[1](1)
        connect axon_hillock[3](0), axon_hillock[2](1)
        connect axon_IS[0](0), axon_hillock[3](1)
        connect myelinated_axon[0](0), axon_IS[0](1)
    ''')

    for idx, sec in enumerate(neuron.h.axon_hillock):
        #print sec.name()
        sec.L = 2.5
        sec.diam = 4 - idx

    for idx, sec in enumerate(neuron.h.axon_IS):
        #print sec.name()
        sec.L = 20.
        sec.diam = 1.

    for idx, sec in enumerate(neuron.h.myelinated_axon):
        #print sec.name()
        sec.L = 1000.
        sec.diam = 1.


def active_declarations(**kwargs):
    ''' Set active conductances for modified CA1 cell

    '''

    # TODO: Test if Ih_prox makes sensible results
    # TODO: Add more channels

    #neuron.h.geom_nse# g()
    create_axon()
    apic_trunk, basal, apic_tuft = make_section_lists()
    modify_morphology(apic_trunk, basal, apic_tuft)
    biophys_passive(apic_trunk, basal, apic_tuft, **kwargs)
    insert_Ih(apic_trunk, basal, apic_tuft)



if __name__ == '__main__':

    timeres = 2**-4
    cut_off = 0
    tstopms = 200
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

    # ax1 = plt.subplot(221, aspect='equal')
    # ax2 = plt.subplot(223, aspect='equal')
    # ax3 = plt.subplot(133, aspect='equal')
    #
    # [ax1.plot([cell.xstart[i], cell.xend[i]], [cell.zstart[i], cell.zend[i]],
    #           'k', lw=cell.diam[i]**0.5) for i in xrange(len(cell.xmid))]
    # [ax3.plot([cell.zstart[i], cell.zend[i]], [-cell.xstart[i], -cell.xend[i]], 'k')
    #         for i in xrange(len(cell.xmid))]
    # [ax2.plot([cell.xstart[i], cell.xend[i]], [-cell.ystart[i], -cell.yend[i]], 'k')
    #         for i in xrange(len(cell.xmid))]

    #print cell.xmid.shape
    #plt.show()
    sys.exit()
    neuron.h.celsius = 33
    cell.simulate(rec_vmem=True, rec_imem=True)

    #clr = lambda idx: plt.cm.jet(int(256. * idx/(len(conductance_list ) - 1)))

    plt.figure(figsize=[10,5])
    plt.subplot(121, aspect='equal')
    #[plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], color='gray')
    #          for idx in xrange(cell.totnsegs)]
    plt.scatter(cell.xmid, cell.zmid, c=cell.vmem[:, -1], vmin=np.min(cell.vmem), vmax=np.max(cell.vmem))
    print cell.vmem
    #
    # K = 0.006
    # R = 8.315
    # F = 96.48
    # vhalfn = -82.
    # z = -3.
    # gamma = 0.5
    # betn = K * np.exp(-z * (1 - gamma) * (cell.vmem-vhalfn) * F / (R * (273.16+neuron.h.celsius)))
    # alpn = K * np.exp(z * gamma * (cell.vmem-vhalfn) * F / (R * (273.16+neuron.h.celsius)))
    # print betn
    # print alpn
    plt.colorbar()
    plt.subplot(122)
    plt.plot(cell.tvec, cell.somav)

    plt.savefig('test.png')