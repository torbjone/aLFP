#!/bin/sh

#PBS -lnodes=5:ppn=16
#PBS -lwalltime=24:00:00
#PBS -A nn4661k

cd $PBS_O_WORKDIR
mpirun -np 80 python cellsim16pops.py --quiet
#wait
#mpirun -np 20 -bynode -bind-to-core -cpus-per-proc 4 python introfig.py
#wait
#mpirun -np 80 python redo_postproc.py