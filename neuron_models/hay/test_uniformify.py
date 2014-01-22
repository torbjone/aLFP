__author__ = 'torbjone'

""" Test if hay model can be made uniform by tampering with the static currents
"""

from os.path import join

import LFPy
import neuron


timeres = 2**-6
cut_off = 200
tstopms = 1000

model_path = join('lfpy_version')
neuron.load_mechanisms(join('mod'))

cell_params_n = {
    'morphology': join(model_path, 'morphologies', 'cell1.hoc'),
    #'rm' : 30000,               # membrane resistance
    #'cm' : 1.0,                 # membrane capacitance
    #'Ra' : 100,                 # axial resistance
    'v_init': -77,             # initial crossmembrane potential
    'passive': False,           # switch on passive mechs
    'nsegs_method': 'lambda_f',  # method for setting number of segments,
    'lambda_f': 100,           # segments are isopotential at this frequency
    'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
    'timeres_python': 1,
    'tstartms': -cut_off,          # start time, recorders start at t=0
    'tstopms': tstopms,
    'custom_code': [join(model_path, 'custom_codes.hoc'),
                    join(model_path, 'biophys3_active.hoc')],
}


cell_params_u = {
    'morphology': join(model_path, 'morphologies', 'cell1.hoc'),
    #'rm' : 30000,               # membrane resistance
    #'cm' : 1.0,                 # membrane capacitance
    #'Ra' : 100,                 # axial resistance
    'v_init': -77,             # initial crossmembrane potential
    'passive': False,           # switch on passive mechs
    'nsegs_method': 'lambda_f',  # method for setting number of segments,
    'lambda_f': 100,           # segments are isopotential at this frequency
    'timeres_NEURON': timeres,   # dt of LFP and NEURON simulation.
    'timeres_python': timeres,
    'tstartms': -cut_off,          # start time, recorders start at t=0
    'tstopms': tstopms,
    'custom_code': [join(model_path, 'custom_codes.hoc'),
                    join(model_path, 'biophys3_active_mod.hoc')],
}


def normal_cell():
    cell = LFPy.Cell(**cell_params_n)
    cell.simulate(rec_vmem=True, rec_imem=True)
    return cell


def uniform_cell():
    cell = LFPy.Cell(**cell_params_u)
    cell.simulate(rec_vmem=True, rec_imem=True)
    return cell


if __name__ == '__main__':
    #cell_n = normal_cell()
    cell_u = uniform_cell()