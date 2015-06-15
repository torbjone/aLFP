'''
Documentation:

This file reformats NEST output in a convinient way.

'''
import os
import numpy as np
from glob import glob
#import h5py_wrapper
from cellsim16popsParams import point_neuron_network_params
#from analysis_params import params
from hybridLFPy import helpers
from mpi4py import MPI

###################################
# Initialization of MPI stuff     #
###################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


flattenlist = lambda lst: sum(sum(lst, []),[])


def get_raw_gids(model_params):
    '''
    Reads text file containing gids of neuron populations as created within the NEST simulation. 
    These gids are not continuous as in the simulation devices get created in between.
    '''
    gidfile = open(os.path.join(model_params.raw_nest_output_path, model_params.GID_filename),'r') 
    gids = [] 
    for l in gidfile :
        a = l.split()
        gids.append([int(a[0]),int(a[1])])
    return gids 
    

def merge_gdf(model_params, raw_label='spikes_', file_type='gdf',
              fileprefix='spikes'):
    '''
    NEST produces one file per virtual process containing voltages. 
    This function gathers and combines them into one single file per hybridLFPy.
    '''
    #some preprocessing
    raw_gids = get_raw_gids(model_params)
    pop_sizes = [raw_gids[i][1]-raw_gids[i][0]+1 for i in np.arange(model_params.Npops)]
    raw_first_gids =  [raw_gids[i][0] for i in np.arange(model_params.Npops)]
    converted_first_gids = [int(1 + np.sum(pop_sizes[:i])) for i in np.arange(model_params.Npops)]

    for pop_idx in np.arange(model_params.Npops):
        if pop_idx % SIZE == RANK:
            files = glob(os.path.join(model_params.raw_nest_output_path,
                                      raw_label + str(pop_idx) + '*.' + file_type))
            gdf = [] # init
            for f in files:
                new_gdf = helpers.read_gdf(f)
                for line in new_gdf:
                    line[0] = line[0] - raw_first_gids[pop_idx] + converted_first_gids[pop_idx]
                    gdf.append(line)
            
            print 'writing: %s' % os.path.join(model_params.spike_output_path,
                                               fileprefix + '_%s.gdf' % model_params.X[pop_idx])
            helpers.write_gdf(gdf, os.path.join(model_params.spike_output_path,
                                                fileprefix + '_%s.gdf' % model_params.X[pop_idx]))
    
    COMM.Barrier()

    return
    

#def create_spatial_input_spikes_hdf5(model_params, fileprefix='depth_res_input_spikes-'):
#    '''
#    NEST produces one input spike file per virtual process, per layer, per cell-type.
#    This functions reads and calculates 'derived' layer and cell type specific 
#    input currents and saves them into hdf5 files.
#
#    Derived currents:
#    1) Total input current = sum of excitatory and inhibitory input currents
#    2) Sum of absolute values of excitatory and inhibitory input current (see Mazzoni 2008,2010)
#    '''
#    # Dictionary of total cell-type and layer specific input currents
#    data_dict={}
#
#    # Dictionary of cell-type and layer specific input currents a la Mazzoni
#    Mazzoni_data_dict={}
#
#    # Loop over all cell types
#    for cell_type in flattenlist(model_params.y_in_Y):
#        # Create empty sub-dictionary for each cell type
#        data_dict[cell_type] = {}
#        Mazzoni_data_dict[cell_type] = {}
#
#        # Loop over input layers
#        for input_layer in np.arange(model_params.num_input_layers):
#            # Initialize arrays for currents
#            current = np.zeros((model_params.tstop-model_params.tstart)/model_params.dt)  
#            Mazzoni_current = np.zeros((model_params.tstop-model_params.tstart)/model_params.dt)  
#
#            # File pattern of files to be read from and files
#            file_pattern = os.path.join(model_params.raw_nest_output_path,
#                        fileprefix + cell_type + '-' + str(input_layer) + '*.dat')
#            files = glob(file_pattern)
#            if files:
#                #Loop over all files (one file per virtual process)
#                for i, f in enumerate(files):
#                    if i % SIZE == RANK:
#                        data = open(f,'r')
#                        
#                        # Go through all lines of data
#                        for l in data :
#                            a = l.split()
#                            
#                            # Calculate derived currents
#                            this_current = (float(a[2])+float(a[3]))*1e-12 # pA -> nA
#                            this_Mazzoni_current = (np.abs(float(a[2]))+np.abs(float(a[3])))*1e-12 # pA -> nA
#    
#                            # Only update array if there is some current
#                            if this_current != 0.0:
#                                # Note: each file contains 'num_readout_cells' neurons -> calculate mean current
#                                current[int(float(a[1])/model_params.dt)] += this_current/ model_params.n_rec_depth_resolved_input 
#                            if this_Mazzoni_current != 0.0:
#                                Mazzoni_current[int(float(a[1])/model_params.dt)] += this_Mazzoni_current/ model_params.n_rec_depth_resolved_input 
#                        print f + ' done'
#                        data.close()
#
#            # Add derived currents in dictionaries
#            if RANK == 0:
#                total_current = np.zeros_like(current)
#                total_M_current = np.zeros_like(Mazzoni_current)
#            else:
#                total_current = None
#                total_M_current = None
#            
#            #sum to rank 0
#            COMM.Reduce([current, MPI.DOUBLE], [total_current, MPI.DOUBLE],
#                                                                    op=MPI.SUM, root=0)
#            COMM.Reduce([Mazzoni_current, MPI.DOUBLE], [total_M_current, MPI.DOUBLE],
#                                                                            op=MPI.SUM, root=0)
#            
#            #fill in dicts
#            data_dict[cell_type][str(input_layer)] = total_current
#            Mazzoni_data_dict[cell_type][str(input_layer)] = total_M_current
#            
#            
#            
#    if RANK == 0:
#        # Save data in hdf5 files
#        h5py_wrapper.add_to_h5(os.path.join(model_params.spike_output_path,
#                                            'cell_type_and_layer_resolved_currents.h5'),
#                               data_dict)
#        h5py_wrapper.add_to_h5(os.path.join(model_params.spike_output_path,
#                                            'cell_type_and_layer_resolved_Mazzoni_currents.h5'),
#                               Mazzoni_data_dict)
#
#    COMM.Barrier()
#    
#    return 


if __name__ == '__main__':    
    # Load model and analysis parameters
    networkParams = point_neuron_network_params()
     
    #test functions
    merge_gdf(networkParams,
            raw_label=networkParams.spike_detector_label,
            file_type='gdf',
            fileprefix=params.networkSimParams['label'])
    merge_gdf(networkParams,
            raw_label=networkParams.voltmeter_label,
            file_type='dat',
            fileprefix='voltages')
    merge_gdf(networkParams,
            raw_label=networkParams.weighted_input_spikes_label,
            file_type='dat',
            fileprefix='population_input_spikes')
    
    #create_spatial_input_spikes_hdf5(networkParams,
    #                                 fileprefix='depth_res_input_spikes-')

