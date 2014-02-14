__author__ = 'torbjone'

""" Test if hay model can be made uniform by tampering with the static currents
"""

from os.path import join

import LFPy
import neuron
import pylab as plt


timeres = 2**-6
cut_off = 1000
tstopms = 1000

model_path = join('lfpy_version')
neuron.load_mechanisms(join('mod'))

cell_params_n = {
    'morphology': join(model_path, 'morphologies', 'cell1.hoc'),
    #'rm' : 30000,               # membrane resistance
    #'cm' : 1.0,                 # membrane capacitance
    #'Ra' : 100,                 # axial resistance
    'v_init': -60,             # initial crossmembrane potential
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


def normal_cell():
    cell = LFPy.Cell(**cell_params_n)
    cell.simulate(rec_vmem=True, rec_imem=True)
    return cell


if __name__ == '__main__':
    cell = normal_cell()
    plt.scatter(cell.xmid, cell.zmid, c=cell.vmem[:, -1])
    plt.colorbar()
    plt.figure()
    plt.plot(cell.tvec, cell.vmem[0, :])
    plt.show()