#!/bin/bash

#PBS -lnodes=1:ppn=16
#PBS -lwalltime=36:00:00
#PBS -lpmem=2000MB
#PBS -m abe
#PBS -A nn4661k

folder=generic_study
filename=generic_study.py

cd /home/$USER/work/aLFP/trunk/aLFP
mkdir /global/work/torbness/aLFP/$folder
mkdir /global/work/torbness/aLFP/$folder/hay
cp $filename /global/work/torbness/aLFP/$folder
cd /global/work/torbness/aLFP/$folder/

mpirun -np 8 python generic_study.py