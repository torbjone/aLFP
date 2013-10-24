#!/bin/bash

#PBS -lnodes=8:ppn=16
#PBS -lwalltime=100:00:00
#PBS -lpmem=2000MB
#PBS -m abe
#PBS -A nn4661k

folder=correlated_population
filename=correlated_pop.py

stimuli_pos=dend
correlation=1.0

cd /home/torbness/work/aLFP/$folder/

mkdir /global/work/torbness/aLFP/$folder
mkdir /global/work/torbness/aLFP/$folder/hay

cp $filename /global/work/torbness/aLFP/$folder
cd /global/work/torbness/aLFP/$folder/

maxpartasks=128
CELLS=10000

python $filename make_population

tasks=$(seq 0 $(($CELLS-1)))
for t in $tasks; do
        python $filename simulate_single_cell $correlation $stimuli_pos $t &	
        activetasks=$(jobs | wc -l)
        while [ $activetasks -ge $maxpartasks ]; do
               sleep 1
               activetasks=$(jobs | wc -l)
        done
done
wait

python $filename sum_all_signals $correlation $stimuli_pos active &
python $filename sum_all_signals $correlation $stimuli_pos Ih_linearized &
python $filename sum_all_signals $correlation $stimuli_pos passive_vss &

wait