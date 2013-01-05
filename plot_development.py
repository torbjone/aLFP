import pylab as pl
import numpy as np
import os
import sys

filelist = os.listdir(os.path.join('renormalized', 'development'))

v_r = [x for x in filelist if ('v_renormalized' in x or 'v_active' in x)]
idx = 4800

for v in v_r:
    name = v[6:-4]

    print name
    foo = np.load(os.path.join('renormalized', 'development', v))
    marker = '-'
    if 'active' in name:
        marker = '--'
    pl.subplot(121)
    pl.plot(foo - foo[4800], marker)
    pl.subplot(122)
    pl.plot(foo, marker, label = name)
    
pl.legend()
pl.show()
