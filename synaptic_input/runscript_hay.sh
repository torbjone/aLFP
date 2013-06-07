#!/bin/bash

#PBS -lnodes=1:ppn=2
#PBS -lwalltime=10:00:00
#PBS -lpmem=2000MB
#PBS -m abe
#PBS -A nn4661k

filename=hay_synaptic.py
mkdir /global/work/torbness/aLFP/
mkdir /global/work/torbness/aLFP/synaptic_input/
cd /global/work/torbness/aLFP/synaptic_input/
cp /home/torbness/work/aLFP/synaptic_input/$filename .
cp /home/torbness/work/aLFP/synaptic_input/params.py .

python $filename simulate
python $filename plot_synaptic
