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

    for sec in section_dict['oblique_dendrites']:
        sec.insert("Ih_BK_prox")
        for seg in sec:
            dist = nrn.distance(seg.x)
            seg.ghbar_Ih_BK_prox = 1e-4*(0.2 + (2.0 - 0.2)/(1 + np.exp((250. - dist)/30)))

    for sec in section_dict['apic_tuft']:
        sec.insert("Ih_BK_dist")
        sec.ghbar_Ih_BK_dist = 1e-4 * 20.

def insert_Ih_frozen(section_dict):

    for sec in section_dict['basal']:
        sec.insert("Ih_BK_prox_frozen")
        sec.ghbar_Ih_BK_prox_frozen = 1e-4 * 0.2

    nrn.distance()
    for sec in section_dict['apic_trunk']:
        sec.insert("Ih_BK_prox_frozen")
        for seg in sec:
            dist = nrn.distance(seg.x)
            seg.ghbar_Ih_BK_prox_frozen = 1e-4*(0.2 + (2.0 - 0.2)/(1 + np.exp((250. - dist)/30)))

    for sec in section_dict['oblique_dendrites']:
        sec.insert("Ih_BK_prox_frozen")
        for seg in sec:
            dist = nrn.distance(seg.x)
            seg.ghbar_Ih_BK_prox_frozen = 1e-4*(0.2 + (2.0 - 0.2)/(1 + np.exp((250. - dist)/30)))

    for sec in section_dict['apic_tuft']:
        sec.insert("Ih_BK_dist_frozen")
        sec.ghbar_Ih_BK_dist_frozen = 1e-4 * 20.

def insert_Im(section_dict):

    gkm = 1e-4 * 12.
    max_dist = 40.

    nrn.distance()
    for sec in section_dict['somatic']:
        if sec.name() == 'soma[1]':
            nrn.distance(0, 1)

    key_list = ['axonal_hillock', 'axonal_IS', 'myelinated_axonal', 'somatic']

    for key, sec_list in section_dict.items():
        if not key in key_list:
            continue
        for sec in sec_list:
            sec.insert("Im_BK")
            for seg in sec:
                if nrn.distance(seg.x) < max_dist:
                    # print sec.name(), "gets IM"
                    seg.gkbar_Im_BK = gkm
                else:
                    # print sec.name(), "doesn't get IM"
                    seg.gkbar_Im_BK = 0

def insert_Im_frozen(section_dict):

    gkm = 1e-4 * 12.
    max_dist = 40.

    nrn.distance()
    for sec in section_dict['somatic']:
        if sec.name() == 'soma[1]':
            nrn.distance(0, 1)

    key_list = ['axonal_hillock', 'axonal_IS', 'myelinated_axonal', 'somatic']

    for key, sec_list in section_dict.items():
        if not key in key_list:
            continue
        for sec in sec_list:
            sec.insert("Im_BK_frozen")
            for seg in sec:
                if nrn.distance(seg.x) < max_dist:
                    # print sec.name(), "gets IM"
                    seg.gkbar_Im_BK_frozen = gkm
                else:
                    # print sec.name(), "doesn't get IM"
                    seg.gkbar_Im_BK_frozen = 0


def insert_INaP(section_dict):

    gna = 1e-4 * 50.
    max_dist = 40.  # Changed because distance to soma is much less than distance to soma center.
    nrn.distance()
    key_list = ['axonal_hillock', 'axonal_IS', 'myelinated_axonal']
    for key, sec_list in section_dict.items():
        if not key in key_list:
            continue
        for sec in sec_list:
            sec.insert("INaP_BK")
            for seg in sec:
                if nrn.distance(seg.x) < max_dist:
                    # print sec.name(), "gets INaP"
                    seg.gnabar_INaP_BK = gna
                else:
                    # print sec.name(), "doesn't get INaP"
                    seg.gnabar_INaP_BK = 0

def insert_INaP_frozen(section_dict):

    gna = 1e-4 * 50.
    max_dist = 40.  # Changed because distance to soma is much less than distance to soma center.
    nrn.distance()
    key_list = ['axonal_hillock', 'axonal_IS', 'myelinated_axonal']
    for key, sec_list in section_dict.items():
        if not key in key_list:
            continue
        for sec in sec_list:
            sec.insert("INaP_BK_frozen")
            for seg in sec:
                if nrn.distance(seg.x) < max_dist:
                    # print sec.name(), "gets INaP"
                    seg.gnabar_INaP_BK_frozen = gna
                else:
                    # print sec.name(), "doesn't get INaP"
                    seg.gnabar_INaP_BK_frozen = 0


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
                seg.e_pas += seg.ina/seg.g_pas
            if nrn.ismembrane("k_ion"):
                seg.e_pas += seg.ik/seg.g_pas
            if nrn.ismembrane("ca_ion"):
                seg.e_pas += seg.ica/seg.g_pas
            if nrn.ismembrane("Ih_BK_prox"):
                seg.e_pas += seg.ih_Ih_BK_prox/seg.g_pas
            if nrn.ismembrane("Ih_BK_dist"):
                seg.e_pas += seg.ih_Ih_BK_dist/seg.g_pas
            if nrn.ismembrane("Ih_BK_prox_frozen"):
                seg.e_pas += seg.ih_Ih_BK_prox_frozen/seg.g_pas
            if nrn.ismembrane("Ih_BK_dist_frozen"):
                seg.e_pas += seg.ih_Ih_BK_dist_frozen/seg.g_pas

def make_section_lists(cellname):

    section_dict = {'somatic': nrn.SectionList(),
                    'myelinated_axonal': nrn.SectionList(),
                    'axonal_hillock': nrn.SectionList(),
                    'axonal_IS': nrn.SectionList(),
                    'basal': nrn.SectionList(),
                    'apic_trunk': nrn.SectionList(),
                    'oblique_dendrites': nrn.SectionList(),
                    'apic_tuft': nrn.SectionList(),
                   }

    if cellname == 'n120':
        apic_trunk_numbs = np.arange(10)
        apic_tuft_numbs = np.arange(10, 26)
        oblique_tuft_numbs = np.arange(26, 52)

        for sec in nrn.allsec():
            sec_type = sec.name().split('[')[0]
            sec_numb = int(sec.name().split('[')[1].split(']')[0])
            if sec_type == 'apic':
                if sec_numb in apic_trunk_numbs:
                    section_dict['apic_trunk'].append()
                elif sec_numb in apic_tuft_numbs:
                    section_dict['apic_tuft'].append()
                elif sec_numb in oblique_tuft_numbs:
                    section_dict['oblique_dendrites'].append()
                else:
                    raise RuntimeError("%s in apic not sorted right!" % sec.name())
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

    elif cellname == 'c12861':
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
    return section_dict


def plot_cell_sections(section_dict):
    clr_dict = {'somatic': 'k',
               'axonal_IS': 'r',
               'axonal_hillock': 'r',
               'myelinated_axonal': 'r',
               'basal': 'g',
               'apic_trunk': 'y',
               'oblique_dendrites': 'm',
               'apic_tuft': 'b',
    }
    for key, sec_list in section_dict.items():
        for sec in sec_list:
            for i in xrange(int(nrn.n3d())):
                plt.scatter(nrn.x3d(i), nrn.y3d(i), c=clr_dict[key],
                            s=10*nrn.diam3d(i), edgecolor='none')
    plt.axis('equal')
    plt.show()


def modify_morphology(section_dict, cellname):

    for key, sec_list in section_dict.items():
        for sec in sec_list:
            sec.nseg = 11

    for sec in section_dict['basal']:
        for i in xrange(int(nrn.n3d())):
            nrn.pt3dchange(i, 0.76)

    for sec in section_dict['oblique_dendrites']:
        for i in xrange(int(nrn.n3d())):
            nrn.pt3dchange(i, 0.73)

    if cellname == 'n120':
        apic_root_segment = 'apic[9]'
    elif cellname == 'c12861':
        apic_root_segment = 'apic[92]'
    else:
        raise RuntimeError("Not known cellname!")

    nrn.distance()
    apic_tuft_root_diam = None
    apic_tuft_root_dist = None

    for sec in section_dict['apic_trunk']:
        npts = int(nrn.n3d())
        cummulative_L = 0
        for i in xrange(npts):
            if not i == 0:
                delta_x = (nrn.x3d(i) - nrn.x3d(i - 1))**2
                delta_y = (nrn.y3d(i) - nrn.y3d(i - 1))**2
                delta_z = (nrn.z3d(i) - nrn.z3d(i - 1))**2
                cummulative_L += np.sqrt(delta_x + delta_y + delta_z)
            dist_from_soma = nrn.distance(0) + cummulative_L
            diam = 3.5 - 4.7e-3 * dist_from_soma
            # print diam, nrn.diam3d(i)
            nrn.pt3dchange(i, diam)
        if sec.name() == apic_root_segment:
            apic_tuft_root_diam = nrn.diam3d(npts - 1)
            apic_tuft_root_dist = nrn.distance(1.)

    longest_tuft_branch = find_longest_tuft_branch(section_dict, apic_tuft_root_dist)

    tuft_smallest_diam = 0.3
    for sec in section_dict['apic_tuft']:
        npts = int(nrn.n3d())
        cummulative_L = 0
        start_dist_from_tuft_root = nrn.distance(0.0) - apic_tuft_root_dist
        for i in xrange(npts):
            if not i == 0:
                delta_x = (nrn.x3d(i) - nrn.x3d(i - 1))**2
                delta_y = (nrn.y3d(i) - nrn.y3d(i - 1))**2
                delta_z = (nrn.z3d(i) - nrn.z3d(i - 1))**2
                cummulative_L += np.sqrt(delta_x + delta_y + delta_z)
            dist_from_root = start_dist_from_tuft_root + cummulative_L
            diam = apic_tuft_root_diam - dist_from_root/longest_tuft_branch * (apic_tuft_root_diam - tuft_smallest_diam)
            # print nrn.diam3d(i), diam
            nrn.pt3dchange(i, diam, sec=sec)


def find_longest_tuft_branch(section_dict, apic_tuft_root_dist):
    longest_branch = 0
    for sec in section_dict['apic_tuft']:
        npts = int(nrn.n3d())
        cummulative_L = 0
        start_dist_from_tuft_root = nrn.distance(0.0) - apic_tuft_root_dist
        for i in xrange(npts):
            if not i == 0:
                delta_x = (nrn.x3d(i) - nrn.x3d(i - 1))**2
                delta_y = (nrn.y3d(i) - nrn.y3d(i - 1))**2
                delta_z = (nrn.z3d(i) - nrn.z3d(i - 1))**2
                cummulative_L += np.sqrt(delta_x + delta_y + delta_z)
        longest_branch = np.max([longest_branch, start_dist_from_tuft_root + cummulative_L])
    return longest_branch


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

    area_of_spine = lambda diam_head, diam_neck, length_neck: np.pi * (diam_head**2 +
                                                                       diam_neck * length_neck -
                                                                       0.25 * diam_neck**2)
    spine_factors = {'somatic': 1,
                    'axonal_IS': 1,
                    'axonal_hillock': 1,
                    'myelinated_axonal': 1,
                    'basal': 1 + 1.26 * area_of_spine(0.45, 0.15, 0.45),
                    'apic_trunk': 1 + 1.27 * area_of_spine(0.45, 0.15, 0.45),
                    'oblique_dendrites': 1 + 1.43 * area_of_spine(0.45, 0.15, 0.45),
                    'apic_tuft': 1 + 0.6 * area_of_spine(0.56, 0.15, 0.45)}
    nrn.distance()
    for key, sec_list in section_dict.items():
        for sec in sec_list:
            sec.insert('pas')
            sec.e_pas = Vrest
            sec.Ra = 100.
            sec.g_pas = 1./rm_dict[key]
            sec.cm = cm_dict[key]
            if key == 'apic_trunk':
                for seg in sec:
                    spine_corr = spine_factors[key] if nrn.distance(seg.x) > 100 else 1.
                    # print sec.name(), nrn.distance(seg.x), spine_corr
            elif key == 'basal':
                for seg in sec:
                    spine_corr = spine_factors[key] if nrn.distance(seg.x) > 20 else 1.
                    # print sec.name(), nrn.distance(seg.x), spine_corr
            else:
                spine_corr = spine_factors[key]
            sec.g_pas *= spine_corr
            sec.cm *= spine_corr


def area_study(cell):

    soma_area = 0
    total_area = 0

    comp = 0
    for sec in cell.allseclist:
        for seg in sec:
            if 'soma' in sec.name():
                soma_area += cell.area[comp]
            total_area += cell.area[comp]
            comp += 1
    print comp, cell.totnsegs
    print "Cell total area: %1.3f\n Soma area: %1.3f" % (total_area, soma_area)


def active_declarations(**kwargs):
    ''' Set active conductances for modified CA1 cell
    '''
    section_dict = make_section_lists(kwargs['cellname'])
    modify_morphology(section_dict, kwargs['cellname'])
    biophys_passive(section_dict, **kwargs)
    added_channels = 0
    if 'use_channels' in kwargs:
        if 'Ih' in kwargs['use_channels']:
            print "Inserting Ih"
            insert_Ih(section_dict)
            added_channels += 1
        if 'Ih_frozen' in kwargs['use_channels']:
            print "Inserting frozen Ih"
            insert_Ih_frozen(section_dict)
            added_channels += 1
        if 'Im' in kwargs['use_channels']:
            print "Inserting Im"
            insert_Im(section_dict)
            added_channels += 1
        if 'Im_frozen' in kwargs['use_channels']:
            print "Inserting frozen Im"
            insert_Im_frozen(section_dict)
            added_channels += 1
        if 'INaP' in kwargs['use_channels']:
            print "Inserting INaP"
            insert_INaP(section_dict)
            added_channels += 1
        if 'INaP_frozen' in kwargs['use_channels']:
            print "Inserting frozen INaP"
            insert_INaP_frozen(section_dict)
            added_channels += 1
        if not added_channels == len(kwargs['use_channels']):
            raise RuntimeError("The right number of channels was not inserted!")
    else:
        raise RuntimeError("Not specified how to initialize cell!")
    if 'hold_potential' in kwargs:
        make_uniform(kwargs['hold_potential'])


def make_syaptic_stimuli(cell, input_idx):
    # Define synapse parameters
    synapse_parameters = {
        'idx': input_idx,
        'e': 0.,                   # reversal potential
        'syntype': 'ExpSyn',       # synapse type
        'tau': 10.,                # syn. time constant
        'weight': 0.001,            # syn. weight
        'record_current': True,
    }
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([5.]))
    return cell, synapse

def test_steady_state(input_idx, hold_potential, cellname):

    timeres = 2**-4
    cut_off = 0
    tstopms = 500
    tstartms = -cut_off
    model_path = cellname

    cell_params = {
        'morphology': join(model_path, '%s.hoc' % cellname),
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
                             'cellname': cellname,
                             'hold_potential': hold_potential}],
    }

    cell = LFPy.Cell(**cell_params)
    area_study(cell)
    plt.seed(1234)
    print input_idx, hold_potential
    sim_params = {'rec_vmem': True,
                  'rec_imem': True}
    cell.simulate(**sim_params)
    [plt.plot(cell.tvec, cell.vmem[idx, :]) for idx in xrange(len(cell.xmid))]
    plt.show()
    img = plt.scatter(cell.xmid, cell.zmid, c=cell.vmem[:, -1], edgecolor='none')
    plt.axis('equal')
    plt.colorbar(img)
    plt.show()


def simulate_synaptic_input(input_idx, holding_potential, use_channels, cellname):

    timeres = 2**-4
    cut_off = 0
    tstopms = 100
    tstartms = -cut_off
    model_path = cellname

    cell_params = {
        'morphology': join(model_path, '%s.hoc' % cellname),
        #'rm' : 30000,               # membrane resistance
        #'cm' : 1.0,                 # membrane capacitance
        #'Ra' : 100,                 # axial resistance
        'v_init': holding_potential,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': 100,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': timeres,
        'tstartms': tstartms,          # start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'use_channels': use_channels,
                             'cellname': cellname,
                             'hold_potential': holding_potential}],
    }

    cell = LFPy.Cell(**cell_params)
    plt.seed(1234)
    print input_idx, holding_potential
    sim_params = {'rec_vmem': True,
                  'rec_imem': True}
    make_syaptic_stimuli(cell, input_idx)
    cell.simulate(**sim_params)

    plt.subplot(211, title='Soma')
    plt.plot(cell.tvec, cell.vmem[0, :], label='%d %d mV %s' % (input_idx, holding_potential, str(use_channels)))

    plt.subplot(212, title='Input idx %d' % input_idx)
    plt.plot(cell.tvec, cell.vmem[input_idx, :], label='%d %d mV %s' % (input_idx, holding_potential, str(use_channels)))


def test_frozen_currents(input_idx, holding_potential, cellname):

    plt.close('all')
    simulate_synaptic_input(input_idx, holding_potential, [], cellname)
    simulate_synaptic_input(input_idx, holding_potential, ['Ih_frozen', 'Im_frozen', 'INaP_frozen'], cellname)
    simulate_synaptic_input(input_idx, holding_potential, ['Ih', 'Im', 'INaP'], cellname)

    plt.legend(frameon=False)
    plt.savefig('frozen_test_%d_%d_%s.png' % (input_idx, holding_potential, cellname))


def test_morphology(hold_potential, cellname):

    timeres = 2**-4
    cut_off = 0
    tstopms = 100
    tstartms = -cut_off
    model_path = cellname

    cell_params = {
        'morphology': join(model_path, '%s.hoc' % cellname),
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
        'custom_fun_args': [{'use_channels': ['Ih_frozen', 'Im_frozen', 'INaP_frozen'],
                             'cellname': model_path,
                             'hold_potential': hold_potential}],
    }

    cell = LFPy.Cell(**cell_params)

    if 1:
        [plt.plot([cell.xstart[i], cell.xend[i]], [cell.zstart[i], cell.zend[i]], 'k') for i in xrange(cell.totnsegs)]
        [plt.text(cell.xmid[i], cell.zend[i], '%1.2f' % cell.diam[i], color='r') for i in xrange(cell.totnsegs)]

        plt.axis('equal')
        plt.show()


def _make_WN_input(cell, max_freq):
    """ White Noise input ala Linden 2010 is made """
    tot_ntsteps = round((cell.tstopms - cell.tstartms)/\
                  cell.timeres_NEURON + 1)
    I = np.zeros(tot_ntsteps)
    tvec = np.arange(tot_ntsteps) * cell.timeres_NEURON
    for freq in xrange(1, max_freq + 1):
        I += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())
    return I

def _make_white_noise_stimuli(cell, input_idx):

    input_scaling = 0.0005
    max_freq = 1000
    input_array = input_scaling * _make_WN_input(cell, max_freq)

    noise_vec = neuron.h.Vector(input_array)
    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print "Input i" \
                      "nserted in ", sec.name()
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if syn is None:
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noise_vec.play(syn._ref_amp, cell.timeres_NEURON)
    return cell, syn, noise_vec


def test_white_noise_input():

    input = 'apic'
    timeres = 2**-4
    cut_off = 0
    tstopms = 1000
    tstartms = 0
    cellname = 'c12861'
    holding_potential = -60
    model_path = cellname
    use_channels = ['Im']
    lambda_f = 100

    cell_params = {
        'morphology': join(model_path, '%s.hoc' % cellname),
        #'rm' : 30000,               # membrane resistance
        #'cm' : 1.0,                 # membrane capacitance
        #'Ra' : 100,                 # axial resistance
        'v_init': holding_potential,             # initial crossmembrane potential
        'passive': False,           # switch on passive mechs
        'nsegs_method': 'lambda_f',  # method for setting number of segments,
        'lambda_f': lambda_f,           # segments are isopotential at this frequency
        'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
        'timeres_python': timeres,
        'tstartms': tstartms,          # start time, recorders start at t=0
        'tstopms': tstopms,
        'custom_fun': [active_declarations],  # will execute this function
        'custom_fun_args': [{'use_channels': use_channels,
                             'cellname': cellname,
                             'hold_potential': holding_potential}],
    }

    cell = LFPy.Cell(**cell_params)

    apic_idx = cell.get_closest_idx(x=0, y=0, z=350)
    soma_idx = 0

    if input is 'apic':
        input_idx = apic_idx
    else:
        input_idx = soma_idx
    plt.seed(1234)
    print input_idx, holding_potential
    sim_params = {'rec_vmem': True,
                  'rec_imem': True}
    import aLFP
    num_elecs = 5
    electrode_parameters = {
            'sigma': 0.3,
            'x': np.ones(num_elecs) * 100,
            'y': np.zeros(num_elecs),
            'z': np.linspace(-200, 600, num_elecs)
    }
    electrode = LFPy.RecExtElectrode(**electrode_parameters)
    elec_clr = ['cyan', 'orange', 'pink', 'g', 'y']

    cell, syn, noise_vec = _make_white_noise_stimuli(cell, input_idx)
    cell.simulate(electrode=electrode, **sim_params)

    freqs, [psd_s] = aLFP.return_freq_and_psd(cell.tvec, cell.imem[soma_idx, :])
    freqs, [psd_a] = aLFP.return_freq_and_psd(cell.tvec, cell.imem[apic_idx, :])
    freqs, psd_e = aLFP.return_freq_and_psd(cell.tvec, electrode.LFP)

    plt.subplot(121, title="Number of compartments: %d" % cell.totnsegs)
    [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], c='k') for idx in xrange(cell.totnsegs)]
    plt.plot(cell.xmid[soma_idx], cell.zmid[soma_idx], 'rD')
    plt.plot(cell.xmid[apic_idx], cell.zmid[apic_idx], 'bD')
    [plt.plot(electrode.x[idx], electrode.z[idx], 'o', c=elec_clr[idx]) for idx in xrange(num_elecs)]

    plt.subplot(322, xlim=[1, 1000], title='Soma imem psd', ylim=[1e-8, 1e-4])
    plt.loglog(freqs, psd_s)
    plt.grid(True)

    plt.subplot(324, xlim=[1, 1000], title='Apic imem psd', ylim=[1e-8, 1e-3])
    plt.loglog(freqs, psd_a)
    plt.grid(True)

    plt.subplot(326, xlim=[1, 1000], title='LFP', ylim=[1e-10, 1e-5])
    [plt.loglog(freqs, psd_e[idx], c=elec_clr[idx]) for idx in xrange(num_elecs)]
    plt.grid(True)

    plt.savefig('wn_test_Im_apic_whole.png')

if __name__ == '__main__':
    # aLFP.explore_morphology(join('c12861', 'c12861.hoc'))
    # test_morphology(-60, 'c12861')
    # test_steady_state(0, -80, 'c12861')

    test_white_noise_input()

    # test_frozen_currents(0, -80, 'c12861')
    # test_frozen_currents(0, -70, 'c12861')
    # test_frozen_currents(0, -60, 'c12861')
    #
    # test_frozen_currents(500, -80, 'c12861')
    # test_frozen_currents(500, -70, 'c12861')
    # test_frozen_currents(500, -60, 'c12861')
    #
    # test_frozen_currents(0, -80, 'n120')
    # test_frozen_currents(0, -70, 'n120')
    # test_frozen_currents(0, -60, 'n120')
    #
    # test_frozen_currents(500, -80, 'n120')
    # test_frozen_currents(500, -70, 'n120')
    # test_frozen_currents(500, -60, 'n120')
