#!/bin/sh
#PBS -W group_list=vaccine
#PBS -A vaccine
#PBS -e /home/projects/vaccine/people/s143849/alternative_data/log_err/cl_auc_partition.err
#PBS -o /home/projects/vaccine/people/s143849/alternative_data/log_err/cl_auc_partition.log
#PBS -N cl_auc_partition
#PBS -M asbjorn.skaarup@gmail.com
#PBS -m ae
#PBS -l nodes=1:ppn=1:thinnode
#PBS -l walltime=01:00:00:00
#PBS -l mem=50gb
#PBS -d /home/projects/vaccine/people/s143849/alternative_script/
python3 /home/projects/vaccine/people/s143849/alternative_script/partition/partition_2.py
