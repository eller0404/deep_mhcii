#!/bin/sh
#PBS -W group_list=vaccine
#PBS -A vaccine
#PBS -N cl_cell_line_mean
#PBS -e /home/projects/vaccine/people/s143849/alternative_data/log_err/cl_input_mean.err
#PBS -o /home/projects/vaccine/people/s143849/alternative_data/log_err/cl_input_mean.log
#PBS -M asbjorn.skaarup@gmail.com
#PBS -m e
#PBS -l nodes=1:ppn=1:thinnode
#PBS -l walltime=00:03:00:00
#PBS -l mem=70gb
python3 /home/projects/vaccine/people/s143849/alternative_script/extract_top/df_nnalign.py cl_input_ mean
