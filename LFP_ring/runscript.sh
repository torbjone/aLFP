#!/bin/bash

#PBS -lnodes=1:ppn=1
#PBS -lwalltime=10:00:00
#PBS -lpmem=2000MB
#PBS -m abe
#PBS -A nn4661k

cd /home/torbness/work/aLFP/LFP_ring
filename=ring_simulation.py

#mkdir /global/work/torbness/aLFP
#mkdir /global/work/torbness/aLFP/LFP_ring

cp $filename /global/work/torbness/aLFP/LFP_ring
cd /global/work/torbness/aLFP/LFP_ring

python $filename average_circle

#maxpartasks=64
#CELLS=100

#tasks=$(seq 0 $(($CELLS-1)))
#
#for t in $tasks; do
#        python $filename simulate_multiple_WN $t &		
#        activetasks=$(jobs | wc -l)
#        while [ $activetasks -ge $maxpartasks ]; do
#               sleep 1
#               activetasks=$(jobs | wc -l)
#        done
#done
#wait