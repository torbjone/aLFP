import numpy as np
import os
if not os.environ.has_key('DISPLAY'):
    at_stallo = True
else:
    at_stallo = False


elec_x = np.array([-50, 0, 50, 0, 0])
elec_y = np.array([-100, 0, 400, 800, 1100])
elec_z = np.zeros(len(elec_x)) + 5.

if at_stallo:
    timeres = 2**-5
else:
    timeres = 2**-4
tstopms = 1000
