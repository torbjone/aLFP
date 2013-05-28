#!/bin/bash

#PBS -lnodes=1:ppn=16
#PBS -lwalltime=20:00:00
#PBS -lpmem=2000MB
#PBS -m abe
#PBS -A nn4661k

filename=WN_hay.py
mkdir /global/work/torbness/aLFP/
mkdir /global/work/torbness/aLFP/white_noise_input/
cd /global/work/torbness/aLFP/white_noise_input/
cp /home/torbness/work/aLFP/white_noise_input/$filename .
cp /home/torbness/work/aLFP/white_noise_input/params.py .

python $filename simulate
python $filename plot_active
