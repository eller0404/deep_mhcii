#!/bin/sh
#PBS -W group_list=vaccine
#PBS -A vaccine
#PBS -e /home/projects/vaccine/people/s143849/data/log_err/similarity_matrix.err
#PBS -o /home/projects/vaccine/people/s143849/data/log_err/similarity_matrix.log
#PBS -M asbjorn.skaarup@gmail.com
#PBS -m abe
#PBS -l nodes=1:ppn=1:thinnode
#PBS -l walltime=00:00:10:00
#PBS -l mem=15gb
#PBS -d /home/projects/vaccine/people/s143849/script/
python3 /home/projects/vaccine/people/s143849/script/similarity/similarity_matrix.py
