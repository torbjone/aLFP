#!/usr/bin/env python
'''just a test script for stretching morphos. create a cell object, apply
stretchYoMama(), import gui from neuron, use cellbuilder to export stretched
morphology'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import LFPy
import neuron
from glob import glob

np.random.seed(1)

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



def stretchYoMama(cell, upperbound):
    for secname in cell.allsecnames:
        if secname.rfind('apic[0]') >= 0:
            z0 = neuron.h.z3d(0, sec=neuron.h.apic[0])
                
            zmax = np.array([])
            #for h, sec in enumerate(cell.allseclist):
            #        #if sec.name().rfind('apic') >= 0:
            #        zmax = np.r_[zmax, cell.z3d[h].max()]
            for z3d in cell.z3d:
                zmax = np.r_[zmax, z3d.max()]
            zmax = zmax.max()
            
            
            #stretchfactor = (upperbound - cell.somapos[2]) / (zmax - cell.somapos[2])
            stretchfactor = (upperbound - z0) / (zmax - z0)
            print stretchfactor, z0, zmax
        
            
            for h, sec in enumerate(cell.allseclist):
                if sec.name().rfind('apic') >= 0:
                    n3d = int(neuron.h.n3d())
                    for n in xrange(n3d):
                        neuron.h.pt3dchange(n,
                                    cell.x3d[h][n],
                                    cell.y3d[h][n],
                                    (cell.z3d[h][n] - cell.z3d[h][0]) * stretchfactor + cell.z3d[h][0],
                                    cell.diam3d[h][n])
                    neuron.h.define_shape()
                else:
                    n3d = int(neuron.h.n3d())
                    for n in xrange(n3d):
                        neuron.h.pt3dchange(n,
                                    cell.x3d[h][n],
                                    cell.y3d[h][n],
                                    cell.z3d[h][n],
                                    cell.diam3d[h][n])
                    neuron.h.define_shape()
            cell.x3d, cell.y3d, cell.z3d, cell.diam3d = cell._collect_pt3d()
            cell._collect_geometry()
            
            return



fils = ['L6E_oi15rpy4Copy.hoc']
#fils = ['L23E_oi24rpy1.hoc']
##fils = ['L4E_oi53rpy1.hoc']
##fils = ['L5E_oi15rpy4.hoc']
##fils = ['L5E_j4a.hoc']
##fils = ['L6E_51-2a.CNG.hoc']
#
#
#
##
for i, layer in enumerate(layers):
    #for fil in glob(layer+'*oi24rpy1.hoc'):
    #for fil in glob(layer+'*oi53rpy1.hoc'):
    for fil in glob(layer+'*51-2a.CNG.hoc'):
        #for fil in fils:
        for j in xrange(1):
            neuron.h('forall delete_section()')
            cell = LFPy.Cell(fil, pt3d=True)
            
            localzpos = zpos[i] #+ np.random.randn()*100
            
            cell.set_pos(xpos, j*100, localzpos)
            

            upperbound = zaxisLayerBounds[1:3, 0].mean()
            stretchYoMama(cell, upperbound)
            #cell.set_pos(xpos, j*100, localzpos)
            
            
            
            #for x, z in cell.get_pt3d_polygons():
            #    ax.fill(x, z, color='k')
            #
            #ax.plot(cell.xmid, cell.zmid, '.')
            
            plot_morpho(ax, cell, 'k')
            
            ax.text(xpos, 100, fil, ha='center')
            
            #xpos += 500
#cell.set_pos(0, 0, 0)


#for i, layer in enumerate(layers):
#    #for fil in glob(layer+'*oi24rpy1.hoc'):
#    for fil in glob('stretched/' + layer + '*.hoc'):
#        for j in xrange(1):
#            neuron.h('forall delete_section()')
#            cell = LFPy.Cell(fil, pt3d=True)
#            
#            localzpos = zpos[i] #+ np.random.randn()*100
#            
#            cell.set_pos(xpos, j*100, localzpos)
#            
#
#            upperbound = zaxisLayerBounds[0, 0]
#            #stretchYoMama(cell, upperbound)
#            #cell.set_pos(xpos, j*100, localzpos)
#            
#            
#            
#            #for x, z in cell.get_pt3d_polygons():
#            #    ax.fill(x, z, color='k')
#            #
#            #ax.plot(cell.xmid, cell.zmid, '.')
#            
#            plot_morpho(ax, cell, 'k')
#            
#            ax.text(xpos, 100, fil.split('/')[-1], ha='center', fontsize='x-small')
#            
#            xpos += 500




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
fig.savefig('stretcha.pdf', dpi=150)
plt.show()
