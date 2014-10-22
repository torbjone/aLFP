#!/bin/bash

#PBS -lnodes=1:ppn=16
#PBS -lwalltime=36:00:00
#PBS -lpmem=2000MB
#PBS -m abe
#PBS -A nn4661k

folder=generic_study
filename=generic_study.py

cd /home/$USER/work/aLFP/trunk/aLFP
#mkdir /global/work/torbness/aLFP/$folder
#mkdir /global/work/torbness/aLFP/$folder/hay
#cp $filename /global/work/torbness/aLFP/$folder
#cd /global/work/torbness/aLFP/$folder/

maxpartasks=4

WEIGHTS=(0.0001 0.0001 0.0005 0.001 0.005 0.01)
MUS=(-0.5 0.0 2.0)

for mu in ${MUS[@]}; do
    for w in ${WEIGHTS[@]}; do
        echo $mu, $w
        #python $filename $mu $w &
        activetasks=$(jobs | wc -l)
        while [ $activetasks -ge $maxpartasks ]; do
               sleep 1
               activetasks=$(jobs | wc -l)
        done
    done
done
wait

echo "****PLOTTING****"

for w in ${WEIGHTS[@]}; do
    python $filename $w &
    while [ $activetasks -ge $maxpartasks ]; do
          sleep 1
          activetasks=$(jobs | wc -l)
    done
done
wait