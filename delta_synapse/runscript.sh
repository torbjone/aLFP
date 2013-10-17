#!/bin/bash

tasks=$(seq 66 99)

for t in $tasks; do
        python delta_synapse.py simulate_single_cell $t
done


