import numpy as np
import os
if not os.environ.has_key('DISPLAY'):
    at_stallo = True
else:
    at_stallo = False
    
if at_stallo:
    timeres = 2**-5
else:
    timeres = 2**-4
