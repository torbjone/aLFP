__author__ = 'torbjone'

import os
from os.path import join
import sys
import neuron
from neuron import h as nrn
import LFPy
import numpy as np
import pylab as plt
import scipy.fftpack as ff
import aLFP


def insert_Ih(apic_trunk, basal, apic_tuft):

    for sec in basal:
        sec.insert("Ih_BK_prox")
        sec.ghbar_Ih_BK_prox = 1e-4 * 0.2

    nrn.distance(0, 1)
    for sec in apic_trunk:
        sec.insert("Ih_BK_prox")
        for seg in sec:
            dist = nrn.distance(seg.x)
            seg.ghbar_Ih_BK_prox = 1e-4*(0.2 + (2.0 - 0.2)/(1 + np.exp((250. - dist)/30)))

    for sec in apic_tuft:
        sec.insert("Ih_BK_dist")
        sec.ghbar_Ih_BK_dist = 1e-4 * 20.


def insert_Im():

    gkm = 1e-4 * 12.
    max_dist = 40.  # Changed because distance to soma is much less than distance to soma center.

    # Specify the origin close to the start of the axon
    nrn.distance(0, 1)
    sec_lists = [nrn.axonal_hillock, nrn.axonal_IS, nrn.myelinated_axonal, nrn.somatic]
    for clr_idx, sec_list in enumerate(sec_lists):
        for sec in sec_list:
            sec.insert("Im_BK")
            for seg in sec:
                if nrn.distance(seg.x) < max_dist:
                    #print sec.name(), "gets IM"
                    seg.gkbar_Im_BK = gkm
                else:
                    seg.gkbar_Im_BK = 0


def insert_INaP():

    gna = 1e-4 * 50.
    max_dist = 40  # Changed because distance to soma is much less than distance to soma center.
    nrn.distance(0, 1)
    sec_lists = [nrn.axonal_hillock, nrn.axonal_IS, nrn.myelinated_axonal]
    for sec_list in sec_lists:
        for sec in sec_list:
            sec.insert("INaP_BK")
            for seg in sec:
                if nrn.distance(seg.x) < max_dist:
                    #print sec.name(), "gets INaP"
                    seg.gnabar_INaP_BK = gna
                else:
                    seg.gnabar_INaP_BK = 0


def make_uniform(Vrest):
    """ Makes the cell uniform. Doesn't really work for INaP yet,
    since it is way to strong it seems
    """
    nrn.t = 0

    nrn.finitialize(Vrest)
    nrn.fcurrent()

    for sec in nrn.allsec():
        for seg in sec:
            seg.e_pas = seg.v
            if nrn.ismembrane("na_ion"):
                # print sec.name(), seg.e_pas, seg.ina, seg.g_pas, seg.ina/seg.g_pas
                seg.e_pas += seg.ina/seg.g_pas
            if nrn.ismembrane("k_ion"):
                seg.e_pas += seg.ik/seg.g_pas
            if nrn.ismembrane("Ih_BK_prox"):
                seg.e_pas += seg.ih_Ih_BK_prox/seg.g_pas
            if nrn.ismembrane("Ih_BK_dist"):
                seg.e_pas += seg.ih_Ih_BK_dist/seg.g_pas
            if nrn.ismembrane("ca_ion"):
                seg.e_pas += seg.ica/seg.g_pas



def make_section_lists():

    # Apical tuft root is at end of apic[92]
    apic_trunk = nrn.SectionList()
    basal = nrn.SectionList()
    apic_tuft = nrn.SectionList()
    oblique_dendrites = nrn.SectionList()


    for sec in nrn.allsec():
        if sec.name() == 'apic[92]':
            apic_tuft.subtree()
            apic_tuft.remove()
        if 'dend' in sec.name():
            basal.append()

    for sec in apic_tuft:
        for i in xrange(int(nrn.n3d())):
            plt.plot(nrn.x3d(i), nrn.y3d(i), '.', color='r')

    for sec in basal:
        for i in xrange(int(nrn.n3d())):
            plt.plot(nrn.x3d(i), nrn.y3d(i), '.', color='g')

    plt.axis('equal')
    plt.show()



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
    rm_myel_ax = 1.e9

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


def active_declarations(**kwargs):
    ''' Set active conductances for modified CA1 cell

    '''

    # TODO: Documents methods and code better.
    # TODO: Diameter modifications give negative diameters?
    # TODO: Do we see the resonance properties we expect?


    apic_trunk, basal, apic_tuft = make_section_lists()

    # modify_morphology(apic_trunk, basal, apic_tuft)
    # biophys_passive(apic_trunk, basal, apic_tuft, **kwargs)
    # added_channels = 0
    # if 'use_channels' in kwargs:
    #     if 'Ih' in kwargs['use_channels']:
    #         print "Inserting Ih"
    #         insert_Ih(apic_trunk, basal, apic_tuft)
    #         added_channels += 1
    #     if 'Im' in kwargs['use_channels']:
    #         print "Inserting Im"
    #         insert_Im()
    #         added_channels += 1
    #     if 'INaP' in kwargs['use_channels']:
    #         print "Inserting INaP"
    #         insert_INaP()
    #         added_channels += 1
    #     if not added_channels == len(kwargs['use_channels']):
    #         raise RuntimeError("The right number of channels was not inserted!")
    #
    # if 'hold_potential' in kwargs:
    #     make_uniform(kwargs['hold_potential'])


def simple_test(input_idx, hold_potential):

    timeres = 2**-4
    cut_off = 0
    tstopms = 10
    tstartms = -cut_off
    model_path = 'c12861'

    cell_params = {
        'morphology': join(model_path, 'c12861.hoc'),
        #'rm' : 30000,               # membrane resistance
        #'cm' : 1.0,                 # membrane capacitance
        #'Ra' : 100,                 # axial resistance
        'v_init': hold_potential,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': timeres,
        'tstartms': tstartms,          # start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'use_channels': [],
                             'hold_potential': hold_potential}],
    }

    cell = LFPy.Cell(**cell_params)
    sys.exit()

    plt.scatter(cell.xmid, cell.ymid)
    plt.show()

    apic_stim_idx = cell.get_idx('apic[5]')[1]
    figfolder = join(model_path, 'verifications')
    if not os.path.isdir(figfolder): os.mkdir(figfolder)

    plt.seed(1234)
    apic_tuft_idx = cell.get_closest_idx(-400, 0, -50)
    trunk_idx = cell.get_closest_idx(-100, 0, 0)
    axon_idx = cell.get_idx('axon_IS')[0]
    basal_idx = cell.get_closest_idx(100, 100, 0)
    soma_idx = 0

    print input_idx, hold_potential
    idx_list = np.array([soma_idx, apic_stim_idx, apic_tuft_idx,
                         trunk_idx, axon_idx, basal_idx])

    print idx_list
    input_scaling = .01

    sim_params = {'rec_vmem': True,
                  'rec_imem': True,
                  'rec_variables': []}
    cell.simulate(**sim_params)

    simfolder = join(model_path, 'simresults')
    if not os.path.isdir(simfolder): os.mkdir(simfolder)

    simname = join(simfolder, 'simple_%d_%1.3f' % (input_idx, input_scaling))
    if 'use_channels' in cell_params['custom_fun_args'][0] and \
                    len(cell_params['custom_fun_args'][0]['use_channels']) > 0:
        for ion in cell_params['custom_fun_args'][0]['use_channels']:
            simname += '_%s' % ion
    else:
        simname += '_passive'

    if 'hold_potential' in cell_params['custom_fun_args'][0]:
        simname += '_%+d' % cell_params['custom_fun_args'][0]['hold_potential']


    plt.plot(cell.tvec, cell.somav)
    plt.show()
    #plot_cell_steady_state(cell)

if __name__ == '__main__':
    #aLFP.explore_morphology(join('c12861', 'c12861.hoc'))

    simple_test(0, -80)