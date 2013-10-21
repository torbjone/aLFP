#!/bin/bash

#PBS -lnodes=4:ppn=16
#PBS -lwalltime=40:00:00
#PBS -lpmem=2000MB
#PBS -m abe
#PBS -A nn4661k

cd /home/torbness/work/aLFP/delta_synapse
filename=delta_synapse.py

mkdir /global/work/torbness/aLFP/delta_synapse
mkdir /global/work/torbness/aLFP/delta_synapse/hay

cp $filename /global/work/torbness/aLFP/delta_synapse
cd /global/work/torbness/aLFP/delta_synapse/

maxpartasks=64
CELLS=500

python $filename initialize_cell

tasks=$(seq 0 $(($CELLS-1)))
for t in $tasks; do
        python $filename simulate_single_cell $t &		
        activetasks=$(jobs | wc -l)
        while [ $activetasks -ge $maxpartasks ]; do
               sleep 1
               activetasks=$(jobs | wc -l)
        done
done
wait
