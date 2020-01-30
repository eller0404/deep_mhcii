#!/bin/sh
#PBS -W group_list=vaccine
#PBS -A vaccine
#PBS -N in_all_training
#PBS -e /home/projects/vaccine/people/s143849/alternative_data/log_err/in_all_training.err
#PBS -o /home/projects/vaccine/people/s143849/alternative_data/log_err/in_all_training.log
#PBS -M asbjorn.skaarup@gmail.com
#PBS -m e
#PBS -l nodes=1:ppn=1:thinnode
#PBS -l walltime=00:01:00:00
#PBS -l mem=70gb
python3 /home/projects/vaccine/people/s143849/alternative_script/extract_top/in_all_training.py ur_ single_allele/ mean
