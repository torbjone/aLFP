__author__ = 'torbjone'

from os.path import join
import sys
import neuron
from neuron import h as nrn
import LFPy
import numpy as np
import pylab as plt


def insert_Ih(apic_trunk, basal, apic_tuft):

    for sec in basal:
        sec.insert("Ih_BK_prox")
        sec.ghbar_Ih_BK_prox = 0.2

    nrn.distance(0, 1)
    for sec in apic_trunk:
        sec.insert("Ih_BK_prox")
        for seg in sec:
            dist = nrn.distance(seg.x)
            seg.ghbar_Ih_BK_prox = 0.2 + (2.0 - 0.2)/(1 + np.exp((250. - dist)/30))

    for sec in apic_tuft:
        sec.insert("Ih_BK_dist")
        sec.ghbar_Ih_BK_dist = 20.


def insert_Im():

    gkm = 12
    max_dist = 40  # Changed because distance to soma is much less than distance to soma center.

    # Specify the origin close to the start of the axon
    nrn.distance(0, 1)
    sec_lists = [nrn.axonal_hillock, nrn.axonal_IS, nrn.myelinated_axonal, nrn.somatic]
    for clr_idx, sec_list in enumerate(sec_lists):
        for sec in sec_list:
            sec.insert("Im_BK")
            for seg in sec:
                if nrn.distance(seg.x) < max_dist:
                    print sec.name(), "gets INaP"
                    seg.gkbar_Im_BK = gkm
                else:
                    seg.gkbar_Im_BK = 0


def insert_INaP():

    gna = 50
    max_dist = 40  # Changed because distance to soma is much less than distance to soma center.
    nrn.distance(0, 1)
    sec_lists = [nrn.axonal_hillock, nrn.axonal_IS, nrn.myelinated_axonal]
    for sec_list in sec_lists:
        for sec in sec_list:
            sec.insert("INaP_BK")
            for seg in sec:
                if nrn.distance(seg.x) < max_dist:
                    print sec.name(), "gets INaP"
                    seg.gnabar_INaP_BK = gna
                else:
                    seg.gnabar_INaP_BK = 0


def init(Vrest):
    nrn.t = 0

    nrn.finitialize(Vrest)
    nrn.fcurrent()

    for sec in nrn.allsec():
        for seg in sec:
            seg.e_pas = seg.v
            if nrn.ismembrane("na_ion"):
                seg.e_pas += seg.ina/seg.g_pas
            if nrn.ismembrane("k_ion"):
                seg.e_pas += seg.ik/seg.g_pas
            if nrn.ismembrane("Ih_BK"):
                seg.e_pas += seg.ih_Ih_BK/seg.g_pas
            if nrn.ismembrane("ca_ion"):
                seg.e_pas += seg.ica/seg.g_pas


def make_section_lists():

    apic_trunk = nrn.SectionList()
    basal = nrn.SectionList()
    apic_tuft = nrn.SectionList()
    #oblique_dendrites = nrn.SectionList()

    for sec in nrn.allsec():
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
        for i in xrange(int(nrn.n3d())):
            nrn.pt3dchange(i, 0.76)

    nrn.distance()
    apic_tuft_root_diam = None
    apic_tuft_root_dist = None
    for sec in apic_trunk:

        npts = int(nrn.n3d())
        cummulative_L = 0
        for i in xrange(npts - 1):
            delta_x = (nrn.x3d(i + 1) - nrn.x3d(i))**2
            delta_y = (nrn.y3d(i + 1) - nrn.y3d(i))**2
            delta_z = (nrn.z3d(i + 1) - nrn.z3d(i))**2
            cummulative_L += np.sqrt(delta_x + delta_y + delta_z)
            diam = 3.5 - 4.7e-3 * cummulative_L
            nrn.pt3dchange(i, diam, sec=sec)

        apic_tuft_root_diam = nrn.diam3d(npts - 1)
        apic_tuft_root_dist = cummulative_L


    # THE FOLLOWING RETURNS NEGATIVE PARAMETERS!
    # for sec in nrn.somatic:
    #     if sec.name() == 'soma[2]':
    #         nrn.distance(0, 1)
    # for sec in apic_tuft:
    #
    #     npts = int(nrn.n3d())
    #     cummulative_L = 0
    #     start_dist_from_soma = nrn.distance(0)
    #     start_dist_from_tuft_root = start_dist_from_soma - apic_tuft_root_dist
    #     for i in xrange(npts - 1):
    #         delta_x = (nrn.x3d(i + 1) - nrn.x3d(i))**2
    #         delta_y = (nrn.y3d(i + 1) - nrn.y3d(i))**2
    #         delta_z = (nrn.z3d(i + 1) - nrn.z3d(i))**2
    #         cummulative_L += np.sqrt(delta_x + delta_y + delta_z)
    #         dist_from_root = start_dist_from_tuft_root + cummulative_L
    #         diam = apic_tuft_root_diam - 18e-3 * dist_from_root
    #         print diam, nrn.diam3d(i)
    #         # nrn.pt3dchange(i, diam, sec=sec)

    return apic_tuft_root_diam


def biophys_passive(apic_trunk, basal, apic_tuft, **kwargs):
    Vrest = -80 if not 'hold_potential' in kwargs else kwargs['hold_potential']

    rm = 90000.
    rm_apic_tuft = 20000.
    rm_myel_ax = 1e9

    cm = 1.5
    cm_myel_ax = 0.04

    ra = 100.

    for sec in nrn.allsec():
        sec.insert('pas')
        sec.e_pas = Vrest
        sec.g_pas = 1./rm
        sec.Ra = ra
        sec.cm = cm


    for sec in nrn.axonal_hillock:
         sec.e_pas = Vrest
         sec.g_pas = 1./rm
         sec.Ra = ra
         sec.cm = cm

    for sec in nrn.axonal_IS:
         sec.e_pas = Vrest
         sec.g_pas = 1./rm
         sec.Ra = ra
         sec.cm = cm

    for sec in nrn.myelinated_axonal:
         sec.e_pas = Vrest
         sec.g_pas = 1./rm_myel_ax
         sec.Ra = ra
         sec.cm = cm_myel_ax

    for sec in nrn.somatic:
        sec.e_pas = Vrest
        sec.g_pas = 1./rm
        sec.Ra = ra
        sec.cm = cm

    for sec in apic_tuft:
        sec.e_pas = Vrest
        sec.g_pas = 1./rm_apic_tuft
        sec.Ra = ra
        sec.cm = cm

    for sec in basal:
        sec.e_pas = Vrest
        sec.g_pas = 1./rm
        sec.Ra = ra
        sec.cm = cm

    for sec in apic_trunk:
        sec.e_pas = Vrest
        sec.g_pas = 1./rm
        sec.Ra = ra
        sec.cm = cm

# def create_axon():
#
#     neuron.h('''
#         create axon_hillock[4], axon_IS[1], myelinated_axon[1]
#         connect axon_hillock[0](0), soma[1](1)
#         connect axon_hillock[1](0), axon_hillock[0](1)
#         connect axon_hillock[2](0), axon_hillock[1](1)
#         connect axon_hillock[3](0), axon_hillock[2](1)
#         connect axon_IS[0](0), axon_hillock[3](1)
#         connect myelinated_axon[0](0), axon_IS[0](1)
#     ''')
#
#     #neuron.h('access soma[1]')
#     npts = int(nrn.n3d())
#     soma_end_point = nrn.x3d(npts - 1), nrn.y3d(npts - 1), nrn.z3d(npts - 1)
#     print soma_end_point
#
#     #neuron.h('access axon_hillock')
#     # Make axon in positive x-direction
#     for idx, sec in enumerate(nrn.axon_hillock):
#         L = 2.5
#         diam = 4. - idx
#         print soma_end_point[0] + (idx + 1)*L/2
#         nrn.pt3dchange(idx, soma_end_point[0] + (idx + 1)*L/2.,
#                                 soma_end_point[1], soma_end_point[2], diam)
#
#
#     #nrn.pt3dstyle(1)
#     for idx, sec in enumerate(nrn.axon_hillock):
#         npts = int(nrn.n3d())
#         for i in xrange(npts):
#             print nrn.x3d(i), nrn.y3d(i), nrn.z3d(i), nrn.diam3d(i)
#     #
#     #
#     # for idx, sec in enumerate(nrn.axon_IS):
#     #     #print sec.name()
#     #     sec.L = 200.
#     #     sec.diam = 1.
#     #
#     # for idx, sec in enumerate(nrn.myelinated_axon):
#     #     #print sec.name()
#     #     sec.L = 1000.
#     #     sec.diam = 1.


def active_declarations(**kwargs):
    ''' Set active conductances for modified CA1 cell

    '''

    # TODO: Test channels give reasonable results
    # TODO: Documents methods and code better.
    # TODO: Diameter modifications give negative diameters?

    #create_axon()
    apic_trunk, basal, apic_tuft = make_section_lists()
    modify_morphology(apic_trunk, basal, apic_tuft)
    biophys_passive(apic_trunk, basal, apic_tuft, **kwargs)
    #insert_Ih(apic_trunk, basal, apic_tuft)
    #insert_Im()
    insert_INaP()


def plot_cell(cell):

    for t in xrange(200, 300):
        plt.close('all')
        plt.subplot(111, aspect='equal')

        plt.scatter(cell.ymid, -cell.xmid, c=cell.vmem[:, t], vmin=-100, vmax=20,
                    edgecolor='none')

        plt.colorbar()
        plt.savefig('test_%04d.png' % t)


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

    timeres = 2**-4
    cut_off = 0
    tstopms = 100
    tstartms = -cut_off
    model_path = '.'

    cell_params = {
        'morphology': join(model_path, 'n120.hoc'),
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
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'conductance_type': 'active'}],
    }

    cell = LFPy.Cell(**cell_params)
    plot_dynamics()


    #cell.simulate(rec_vmem=True, rec_imem=True)


    [plt.plot(cell.tvec, cell.vmem[idx, :]) for idx in xrange(len(cell.xmid))]
    plt.ylim(-100, 100)
    plt.show()
    #plot_cell(cell)

    # K = 0.00
    # R = 8315.
    # F = 9.648e4
    # vhalfn = -47.
    # z = 6.5
    # gamma = 0.5

    # print -z * (1 - gamma) * (np.linspace(-100, 20)-vhalfn) * F / (R * (273.16+nrn.celsius))

    # betn = np.exp(-z * (1 - gamma) * (cell.vmem-vhalfn) * F / (R * (273.16+nrn.celsius)))
    # alpn = np.exp(z * gamma * (cell.vmem-vhalfn) * F / (R * (273.16+nrn.celsius)))
    #print betn
    #print alpn