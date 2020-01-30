#!/bin/sh
#PBS -W group_list=vaccine
#PBS -A vaccine
#PBS -e /home/projects/vaccine/people/s143849/alternative_data/log_err/auc_nnalign.err
#PBS -o /home/projects/vaccine/people/s143849/alternative_data/log_err/auc_nnalign.log
#PBS -N auc_nnalign
#PBS -M asbjorn.skaarup@gmail.com
#PBS -m ae
#PBS -l nodes=1:ppn=1:thinnode
#PBS -l walltime=00:02:00:00
#PBS -l mem=30gb
#PBS -d /home/projects/vaccine/people/s143849/alternative_script/
python3 /home/projects/vaccine/people/s143849/alternative_script/auc/calc_auc.py cl_input_ mean
