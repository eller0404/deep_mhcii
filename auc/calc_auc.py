#!/usr/bin/python3
import os
cwd = '/home/projects/vaccine/people/s143849/alternative_script/'
os.chdir(cwd)
import sys
sys.path.append(cwd)
import time
start_time = time.time()
print('start importing')
from my_func_computerome import *
print(round(time.time()-start_time,3),'s to import my_func')
import pandas as pd
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import copy
import collections
import json
import time
from subprocess import Popen, PIPE
import subprocess as process

folder = "/home/projects/vaccine/people/s143849/"
data_folder = folder+'alternative_data/'
birkir_folder = data_folder+'birkir/'
partition_folder = data_folder+'s_partition/'
nnalign_folder = data_folder+'s_nnalign/'
extract_top_folder = data_folder+'s_extract_top/'
input_folder = 'input/'
output_folder = 'output/'
single_allele_folder = 'single_allele/'
cell_line_folder = 'cell_line/'
random_folder = 'random/'
mean_folder = 'mean/'
jupyter_folder = data_folder+'jupyter/'
log_folder = data_folder+'log_err/'
auc_folder = data_folder+'s_auc/'
filter_folder = jupyter_folder+'s_filter/'

extension = sys.argv[1]
method = sys.argv[2]

nnalign_filename = extension+'nnalign.ft'
partition_filename = extension+'partition_9mer.ft'
peptide_filename = extension+'acc.txt'
random_pick_filename = extension+'random.ft'
mean_filename = extension+'mean.ft'
top_mhc_uc_filename = extension+method+'_top_mhc_uc.ft'
top_mhc_wc_filename = extension+method+'_top_mhc_wc.ft'
top_protein_uc_filename = extension+method+'_top_protein_uc.ft'
top_protein_wc_filename = extension+method+'_top_protein_wc.ft'

df_nnalign_filename = extract_top_folder+nnalign_filename
df_partition_filename = partition_folder+partition_filename
df_peptides_filename = filter_folder+peptide_filename
write_random_pick_filename = extract_top_folder+random_pick_filename
write_mean_filename = extract_top_folder+mean_filename
filename_mhc_uc = extract_top_folder+top_mhc_uc_filename
filename_mhc_wc = extract_top_folder+top_mhc_wc_filename
filename_protein_uc = extract_top_folder+top_protein_uc_filename
filename_protein_wc = extract_top_folder+top_protein_wc_filename

tmp_filename = 'tmp.csv'
auc_filename = 'auc'

print(write_mean_filename)
df_lig = load_df_feather(write_mean_filename)
print(df_lig.head())
print((df_lig['target']==1).sum())
acc_group = df_lig.groupby(['acc','MHC_mol'])
#acc_group = df_lig.groupby(['source'])

filename_tmp = auc_folder+tmp_filename
for context in ['wc','uc']:
    aucs = list()
    for c,acc in enumerate(acc_group):
        acc[1][['target',f'Prediction_{context}']].to_csv(filename_tmp,header=None,index=False,sep='\t')
        run = ['roc','-t','0.5',filename_tmp]
        process = Popen(run, universal_newlines=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        aucs.append(stdout.split('\n')[-2].split()[2])
        print(acc[0],aucs[-1])

        filename_auc = auc_folder+auc_filename
        aucs_np = np.array(aucs,dtype='float')
        print(context,np.mean(aucs_np))
        np.save(filename_auc+'_'+context,aucs_np)

