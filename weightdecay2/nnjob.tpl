#!/bin/bash

#SBATCH -n 8
#SBATCH -t 60
#SBATCH -p general
#SBATCH --mem-per-cpu=500

module load hpc/python-2.7.1_full_stack

source ~/CS542/env/bin/activate
python ~/CS542/nettalk_network.py -l info -N 1000 \
    -p 5 \
    -s results2/weightdecay_^^.pkl \
    -W ^^ \
    -t firsthalf_random.data \
    -T secondhalf_random.data \
    > results2/weightdecay_^^.out \
    2> results2/weightdecay_^^.err