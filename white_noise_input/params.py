import numpy as np
import os
import sys
from ipdb import set_trace
import pylab as pl
from os.path import join


elec_x = np.array([-100, 0, 100, 0, 0])
elec_y = np.array([-100, 0, 400, 800, 1100])
elec_z = np.zeros(len(elec_x))

input_scaling_BS = np.array([0.001, 0.01, 0.1])

timeres = 2**-5
tstopms = 1000
