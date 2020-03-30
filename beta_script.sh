#!/bin/bash

## otherwise the default shell would be used
#$ -S /bin/bash
exec > "$TMPDIR"/stdout.txt 2>"$TMPDIR"/stderr.txt

## Use current working directory
#$ -cwd 

## Pass environment variables of workstation to GPU node 
#$ -V
 
## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue
#$ -q gpu.long.q@* 

## the maximum memory usage of this job, (below 4G does not make much sense)
#$ -l h_vmem=50G

## specify number of gpus to use
#$ -l gpu=1

## stderr and stdout are merged together to stdout
#$ -j y

# logging directory. preferrably on your scratch
#$ -o logs_and_checkpoints/jobs_log/
#
# Send mail at submission and completion of script
#$ -m be

# call your calculation executable, redirect output
python beta_script.py --repr_type $1
