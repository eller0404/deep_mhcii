#!/bin/sh
#PBS -W group_list=vaccine
#PBS -A vaccine
#PBS -l nodes=1:ppn=1:thinnode
#PBS -l walltime=01:00:00:00
#PBS -l mem=30gb
#PBS -d /home/projects/vaccine/people/s143849/alternative_script/
echo input 1: $1
echo input 2: $2
python3 /home/projects/vaccine/people/s143849/alternative_script/deep/LSTM.py $1 $2 100 10
