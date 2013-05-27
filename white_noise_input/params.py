import numpy as np

elec_x = np.array([-50, 0, 50, 0, 0])
elec_y = np.array([-100, 0, 400, 800, 1100])
elec_z = np.zeros(len(elec_x)) + 5.

timeres = 2**-6
tstopms = 1000
