#!/bin/bash

#PBS -lnodes=1:ppn=16
#PBS -lwalltime=40:00:00
#PBS -lpmem=2000MB
#PBS -m abe
#PBS -A nn4661k

folder=correlated_population
filename=correlated_pop_smaller.py

stimuli_pos=apic
correlation=0.0

cd /home/torbness/work/aLFP/$folder/

mkdir /global/work/torbness/aLFP/$folder
mkdir /global/work/torbness/aLFP/$folder/hay

cp $filename /global/work/torbness/aLFP/$folder
cd /global/work/torbness/aLFP/$folder/

maxpartasks=16
CELLS_EACH=500
START_CELL=$(($PBS_ARRAYID*$CELLS_EACH))
END_CELL=$((($PBS_ARRAYID + 1)*$CELLS_EACH - 1))

echo $START_CELL to $END_CELL
tasks=$(seq $START_CELL $END_CELL)
for t in $tasks; do
        python $filename simulate_single_cell $correlation $stimuli_pos $t &	
        activetasks=$(jobs | wc -l)
        while [ $activetasks -ge $maxpartasks ]; do
               sleep 1
               activetasks=$(jobs | wc -l)
        done
done
wait
