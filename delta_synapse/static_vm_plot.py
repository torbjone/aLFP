
import pylab as plt
import numpy as np
import sys, os
from os.path import join
from ipdb import set_trace
folder = 'hay'

filelist = os.listdir(folder)
xmid = np.load(join(folder, 'xmid.npy'))
ymid = np.load(join(folder, 'ymid.npy'))
#zmid = np.load(join(folder, 'zmid.npy'))

input_pos = ''

for filename in filelist:
    if not ('vmem' in filename) or not ('.npy' in filename) or not ('active' in filename):
        continue
    print filename
    plt.close('all')
    fig = plt.figure()
    ax1 = fig.add_subplot(131, aspect='equal', frameon=False, xticks=[], yticks=[], title='Active')
    ax2 = fig.add_subplot(132, aspect='equal', frameon=False, xticks=[], yticks=[], title='Ih_linearized')
    ax3 = fig.add_subplot(133, aspect='equal', frameon=False, xticks=[], yticks=[], title='Passive_vss')

    
    name_root = 'vmem_%s%s_sim_%d.npy'
    sim = int(filename.split('_')[-1][:-4])
    
    vmem_a = np.load(join(folder, name_root %('active', input_pos, sim)))
    vmem_I = np.load(join(folder, name_root %('Ih_linearized', input_pos, sim)))
    vmem_p = np.load(join(folder, name_root %('passive_vss', input_pos, sim)))
    
    print "Active:\t\t", np.max(vmem_a), np.min(vmem_a)
    print "Ih_linearized:\t", np.max(vmem_I), np.min(vmem_I)
    print "Passive_vss:\t", np.max(vmem_p), np.min(vmem_p)
    img1 = ax1.scatter(xmid, ymid, c=vmem_a[:,0], vmax=-65, vmin=-78, edgecolor='none', cmap='spectral')
    plt.colorbar(img1, ax=ax1)
    
    img2 = ax2.scatter(xmid, ymid, c=vmem_I[:,0], vmax=-65, vmin=-78, edgecolor='none', cmap='spectral')
    plt.colorbar(img2, ax=ax2)
    
    img3 = ax3.scatter(xmid, ymid, c=vmem_p[:,0], vmax=-65, vmin=-78, edgecolor='none', cmap='spectral')
    plt.colorbar(img3, ax=ax3)
    
    for tidx in xrange(vmem_a.shape[1]):
        if tidx < 100:
            continue
        if tidx > 110:
            break
        img1.set_array(vmem_a[:,tidx])
        img2.set_array(vmem_I[:,tidx])
        img3.set_array(vmem_p[:,tidx])

        plt.savefig(join(folder, 'vmem_%s_%d_%05d.png' %(input_pos, sim, tidx)))
                   
