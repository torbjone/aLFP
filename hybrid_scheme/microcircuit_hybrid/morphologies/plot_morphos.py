#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import LFPy
import neuron
from glob import glob


def plot_morpho(ax, cell, color):
    #create linecollection
    line_segments = LineCollection(
        [zip([cell.xstart[i], cell.xend[i]], [cell.zstart[i], cell.zend[i]])
         for i in xrange(cell.totnsegs)],
        linewidths=(1),
        linestyles='solid',
        color=color,
        rasterized=True
    )
    ax.add_collection(line_segments)    



layers = ['L1', 'L23', 'L4', 'L5', 'L6']

#set the boundaries of each layer, L1->L6
zaxisLayerBounds = np.array([[    0.0,   -81.6],
                             [  -81.6,  -587.1],
                             [ -587.1,  -922.2],
                             [ -922.2, -1170.0],
                             [-1170.0, -1491.7]])

zpos = zaxisLayerBounds.mean(axis=1)
xpos = 0

fig = plt.figure(figsize=(15, 6), frameon=False)
fig.subplots_adjust(left=0.075, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
ax = fig.add_subplot(111)

for i, layer in enumerate(layers):
    for fil in glob(layer+'*.hoc'):
        neuron.h('forall delete_section()')
        cell = LFPy.Cell(fil, pt3d=True)
        
        cell.set_pos(xpos, 0, zpos[i])
        
        for x, z in cell.get_pt3d_polygons():
            ax.fill(x, z, color='k', rasterized=True)        
        
        ax.text(xpos, 100, fil, ha='center', fontsize='x-small')
        
        xpos += 500

ax.hlines(zaxisLayerBounds[:, 0], 0, xpos)
ax.hlines(zaxisLayerBounds[-1, -1], 0, xpos)
ax.set_ylabel(r'depth ($\mu$m)')
ax.set_yticks(np.r_[zaxisLayerBounds[:, 0], zaxisLayerBounds[-1, -1]])
ax.set_xticks([])

for loc, spine in ax.spines.iteritems():
    #if loc in ['right','top', 'bottom']:
    spine.set_color('none') # don't draw spine
ax.yaxis.set_ticks_position('left')

        
ax.axis(ax.axis('equal'))

fig.savefig('plot_morphos.pdf', dpi=150)
plt.show()
