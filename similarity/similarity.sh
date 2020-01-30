#!/bin/sh
#PBS -W group_list=vaccine
#PBS -A vaccine
#PBS -e /home/projects/vaccine/people/s143849/data/log_err/blast62_$1_$2.err
#PBS -o /home/projects/vaccine/people/s143849/data/log_err/blast62_$1_$2.log
#PBS -M asbjorn.skaarup@gmail.com
#PBS -m abe
#PBS -l nodes=1:ppn=1:thinnode
#PBS -l walltime=00:00:30:00
#PBS -l mem=5gb
#PBS -d /home/projects/vaccine/people/s143849/script/
module load tools
module load perl/5.24.0
module load ncbi-blast/2.8.1+
echo $1 $2
python3 /home/projects/vaccine/people/s143849/script/similarity/blastp.py $1 $2
