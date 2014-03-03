__author__ = 'torbjone'

import os
from os.path import join
import sys
import neuron
from neuron import h as nrn
import LFPy
import numpy as np
import pylab as plt
import aLFP


def insert_Ih(section_dict):

    for sec in section_dict['basal']:
        sec.insert("Ih_BK_prox")
        sec.ghbar_Ih_BK_prox = 1e-4 * 0.2

    nrn.distance()
    for sec in section_dict['apic_trunk']:
        sec.insert("Ih_BK_prox")
        for seg in sec:
            dist = nrn.distance(seg.x)
            seg.ghbar_Ih_BK_prox = 1e-4*(0.2 + (2.0 - 0.2)/(1 + np.exp((250. - dist)/30)))

    for sec in section_dict['apic_tuft']:
        sec.insert("Ih_BK_dist")
        sec.ghbar_Ih_BK_dist = 1e-4 * 20.


def insert_Im(section_dict):

    gkm = 1e-4 * 12.
    max_dist = 40.  # Changed because distance to soma is much less than distance to soma center.

    # Specify the origin close to the start of the axon
    nrn.distance()
    key_list = ['axonal_hillock', 'axonal_IS', 'myelinated_axonal', 'somatic']

    for key, sec_list in section_dict.items():
        if not key in key_list:
            continue
        for sec in sec_list:
            sec.insert("Im_BK")
            for seg in sec:
                if nrn.distance(seg.x) < max_dist:
                    print sec.name(), "gets IM"
                    seg.gkbar_Im_BK = gkm
                else:
                    print sec.name(), "doesn't get IM"
                    seg.gkbar_Im_BK = 0


def insert_INaP(section_dict):

    gna = 1e-4 * 50.
    max_dist = 40  # Changed because distance to soma is much less than distance to soma center.
    nrn.distance()
    key_list = ['axonal_hillock', 'axonal_IS', 'myelinated_axonal']
    for key, sec_list in section_dict.items():
        if not key in key_list:
            continue
        for sec in sec_list:
            sec.insert("INaP_BK")
            for seg in sec:
                if nrn.distance(seg.x) < max_dist:
                    print sec.name(), "gets INaP"
                    seg.gnabar_INaP_BK = gna
                else:
                    print sec.name(), "doesn't get INaP"
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
    section_dict = {'somatic': nrn.SectionList(),
                    'myelinated_axonal': nrn.SectionList(),
                    'axonal_hillock': nrn.SectionList(),
                    'axonal_IS': nrn.SectionList(),
                    'basal': nrn.SectionList(),
                    'apic_trunk': nrn.SectionList(),
                    'oblique_dendrites': nrn.SectionList(),
                    'apic_tuft': nrn.SectionList(),
                   }

    apic_trunk_numbs = [0, 2, 8, 12, 24, 28, 30, 34, 38,
                        40, 46, 52, 54, 58, 64, 66, 68,
                        72, 74, 78, 80, 82, 86, 88, 90, 92]

    for sec in nrn.allsec():
        sec_type = sec.name().split('[')[0]
        sec_numb = int(sec.name().split('[')[1].split(']')[0])
        if sec_type == 'apic':
            if sec_numb > 92:
                section_dict['apic_tuft'].append()
            elif sec_numb in apic_trunk_numbs:
                section_dict['apic_trunk'].append()
            else:
                section_dict['oblique_dendrites'].append()
        elif sec_type == 'dend':
            section_dict['basal'].append()
        elif sec_type == 'soma':
            section_dict['somatic'].append()
        elif sec_type == 'myelinated_axon':
            section_dict['myelinated_axonal'].append()
        elif sec_type == 'axon_IS':
            section_dict['axonal_IS'].append()
        elif sec_type == 'axon_hillock':
            section_dict['axonal_hillock'].append()
        else:
            raise RuntimeError("%s not placed anywhere!" % sec_type)

        # if sec_type == 'apic' and sec_numb == 'apic[92]':
        #     apic_tuft.subtree()
        #     apic_tuft.remove()
    if 0:
        for sec in section_dict['apic_tuft']:
            for i in xrange(int(nrn.n3d())):
                plt.scatter(nrn.x3d(i), nrn.y3d(i), c='b', s=10*nrn.diam3d(i), edgecolor='none')

        for sec in section_dict['basal']:
            for i in xrange(int(nrn.n3d())):
                plt.scatter(nrn.x3d(i), nrn.y3d(i), c='g', s=10*nrn.diam3d(i), edgecolor='none')

        for sec in section_dict['oblique_dendrites']:
            for i in xrange(int(nrn.n3d())):
                plt.scatter(nrn.x3d(i), nrn.y3d(i), c='m', s=10*nrn.diam3d(i), edgecolor='none')
        for sec in section_dict['apic_trunk']:
            for i in xrange(int(nrn.n3d())):
                plt.scatter(nrn.x3d(i), nrn.y3d(i), c='y', s=10*nrn.diam3d(i), edgecolor='none')
        for sec in section_dict['somatic']:
            for i in xrange(int(nrn.n3d())):
                plt.scatter(nrn.x3d(i), nrn.y3d(i), c='k', s=10*nrn.diam3d(i), edgecolor='none')
        for sec in section_dict['axonal_IS']:
            for i in xrange(int(nrn.n3d())):
                plt.scatter(nrn.x3d(i), nrn.y3d(i), c='r', s=10*nrn.diam3d(i), edgecolor='none')
        for sec in section_dict['axonal_hillock']:
            for i in xrange(int(nrn.n3d())):
                plt.scatter(nrn.x3d(i), nrn.y3d(i), c='r', s=10*nrn.diam3d(i), edgecolor='none')
        for sec in section_dict['myelinated_axonal']:
            for i in xrange(int(nrn.n3d())):
                plt.scatter(nrn.x3d(i), nrn.y3d(i), c='r', s=10*nrn.diam3d(i), edgecolor='none')

        plt.axis('equal')
        plt.show()

    return section_dict


def modify_morphology(section_dict):

    for sec in section_dict['basal']:
        for i in xrange(int(nrn.n3d())):
            nrn.pt3dchange(i, 0.76)

    for sec in section_dict['oblique_dendrites']:
        for i in xrange(int(nrn.n3d())):
            nrn.pt3dchange(i, 0.73)

    nrn.distance()
    apic_tuft_root_diam = None
    apic_tuft_root_dist = None

    for sec in section_dict['apic_trunk']:
        npts = int(nrn.n3d())
        # cummulative_L = 0
        for i in xrange(npts):
            # delta_x = (nrn.x3d(i + 1) - nrn.x3d(i))**2
            # delta_y = (nrn.y3d(i + 1) - nrn.y3d(i))**2
            # delta_z = (nrn.z3d(i + 1) - nrn.z3d(i))**2
            # cummulative_L += np.sqrt(delta_x + delta_y + delta_z)
            # dist_from_soma = nrn.distance(0) + cummulative_L
            # diam = 3.5 - 4.7e-3 * dist_from_soma
            diam = 3.5 - 4.7e-3 * nrn.distance(0.5)
            # print nrn.diam3d(i), diam
            nrn.pt3dchange(i, diam)
        # print sec.name(), nrn.distance(0), nrn.distance(1), nrn.distance(0) + cummulative_L
        if sec.name() == 'apic[92]':
            apic_tuft_root_diam = nrn.diam3d(npts-2)
            apic_tuft_root_dist = nrn.distance(1.)

    for sec in section_dict['apic_tuft']:
        npts = int(nrn.n3d())
        # cummulative_L = 0
        # start_dist_from_soma = nrn.distance(0.5)
        # start_dist_from_tuft_root = start_dist_from_soma - apic_tuft_root_dist
        for i in xrange(npts):
            # delta_x = (nrn.x3d(i + 1) - nrn.x3d(i))**2
            # delta_y = (nrn.y3d(i + 1) - nrn.y3d(i))**2
            # delta_z = (nrn.z3d(i + 1) - nrn.z3d(i))**2
            # cummulative_L += np.sqrt(delta_x + delta_y + delta_z)
            # dist_from_root = start_dist_from_tuft_root + cummulative_L
            diam = apic_tuft_root_diam - 18e-3 * (nrn.distance(0.5) - apic_tuft_root_dist)
            if diam < 0.5:
                diam = 0.5
            nrn.pt3dchange(i, diam, sec=sec)


def biophys_passive(section_dict, **kwargs):
    Vrest = -80 if not 'hold_potential' in kwargs else kwargs['hold_potential']

    rm_dict = {'somatic': 90000.,
               'axonal_IS': 90000.,
               'axonal_hillock': 90000.,
               'myelinated_axonal': 1.0e6,
               'basal': 90000.,
               'apic_trunk': 90000.,
               'oblique_dendrites': 90000.,
               'apic_tuft': 20000.,
    }
    
    cm_dict = {'somatic': 1.5,
               'axonal_IS': 1.5,
               'axonal_hillock': 1.5,
               'myelinated_axonal': 0.04,
               'basal': 1.5,
               'apic_trunk': 1.5,
               'oblique_dendrites': 1.5,
               'apic_tuft': 1.5,
    }
    ra = 100.

    for key, sec_list in section_dict.items():
        for sec in sec_list:
            sec.insert('pas')
            sec.e_pas = Vrest
            sec.g_pas = 1./rm_dict[key]
            sec.Ra = ra
            sec.cm = cm_dict[key]


def active_declarations(**kwargs):
    ''' Set active conductances for modified CA1 cell

    '''

    # TODO: Documents methods and code better.
    # TODO: Diameter modifications give negative diameters?
    # TODO: Do we see the resonance properties we expect?

    section_dict = make_section_lists()

    modify_morphology(section_dict)
    biophys_passive(section_dict, **kwargs)
    added_channels = 0
    if 'use_channels' in kwargs:
        if 'Ih' in kwargs['use_channels']:
            print "Inserting Ih"
            insert_Ih(section_dict)
            added_channels += 1
        if 'Im' in kwargs['use_channels']:
            print "Inserting Im"
            insert_Im(section_dict)
            added_channels += 1
        if 'INaP' in kwargs['use_channels']:
            print "Inserting INaP"
            insert_INaP(section_dict)
            added_channels += 1
        if not added_channels == len(kwargs['use_channels']):
            raise RuntimeError("The right number of channels was not inserted!")

    if 'hold_potential' in kwargs:
        make_uniform(kwargs['hold_potential'])


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
        'custom_fun_args': [{'use_channels': ['Ih', 'Im', 'INaP'],
                             'hold_potential': hold_potential}],
    }

    cell = LFPy.Cell(**cell_params)
    # TODO: Check if channels work, is resonance frequency corrected? Start populations for both hay and Hu

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
    # aLFP.explore_morphology(join('c12861', 'c12861.hoc'))

    simple_test(0, -80)