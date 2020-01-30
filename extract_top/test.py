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
peptide_filename = extension+'acc.csv'
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

filenames = [
    'cl_auc_5-wc.log',
    'cl_auc_4-wc.log',
    'cl_auc_1-wc.log',
    'cl_auc_3-wc.log',
    'cl_auc_2-wc.log',
    'cl_auc_4-uc.log',
    'cl_auc_5-uc.log',
    'cl_auc_3-uc.log',
    'cl_auc_2-uc.log',
    'cl_auc_1-uc.log']
translate_predictions = [
    'Prediction_5_wc',
    'Prediction_4_wc',
    'Prediction_1_wc',
    'Prediction_3_wc',
    'Prediction_2_wc',
    'Prediction_4_uc',
    'Prediction_5_uc',
    'Prediction_3_uc',
    'Prediction_2_uc',
    'Prediction_1_uc']

for c,filename in enumerate(filenames):
    filenames[c] = log_folder+filename

df = load_df_feather(write_mean_filename)
print(df.columns)
print(df.shape)
print(df.tail())