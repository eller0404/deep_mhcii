#!/usr/bin/python3
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
import sys
import os

folder = "/home/projects/vaccine/people/s143849/"
data_folder = 'data/'
birkir_folder = 'birkir/'
deep_folder = 'deep/'
input_folder = 'inputs/'
output_folder = 'outputs/'
loss_folder = 'loss/'
prediction_folder = 'predictions/'
param_folder = 'params/'
epoch_folder = 'epochs/'
donotknow_folder = 'DoNotKnow/'
inliximab_folder = 'infliximab/'
state_folder = 'intermidiate_state/'
peptide_folder = 'peptides/'
protein_folder = 'proteins/'
similarity_folder = 'similarity/'
nnalign_folder = 'nnalign/'
blast62_folder = 'blast62/'
partition_folder = 'partitions/'

ligand_filename = 'MasterMerger_context_pos_19_03_08.txt'
sequence_filename = 'MasterSourceProt_blastEntrez.fasta'
new_ligand_filename = 'log_eval_concat.txt'

df_seq_filename = 'df_seq.json'
df_seq_filter_filename = 'df_seq_filter.json'
df_seq_profile_filename = 'df_seq_profile.json'
filtered_fasta = 'filtered_fasta'
peptides_filename = 'cl_input_acc.csv'
partition_filename = 'ur_partition.ft'
partition_9mer_filename = 'ur_partition_9mer.ft'
top_mhc_uc_filename = 'top_mhc_uc.ft'
top_mhc_wc_filename = 'top_mhc_wc.ft'
top_protein_uc_filename = 'top_protein_uc.ft'
top_protein_wc_filename = 'top_protein_wc.ft'
input_filename = 'predicted_profiles.npy'
output_filename = 'experimental_profiles.npy'
acc_filename = 'acc.npy'
length_filename = 'length.npy'
nnalign_filename = 'nnalign.ft'
random_pick_filename = 'random_pick.ft'

filename = folder+data_folder+nnalign_folder+input_folder+peptides_filename
df_peptides = pd.read_csv(filename,sep='\t',header=None)
df_peptides.columns = ['acc','from_start','to_end','Peptide','affinity','MHC_mol','Context']

train_filenames = ['train_EL','train_BA']
for c,filename in enumerate(train_filenames):
    train_filenames[c] = folder+data_folder+birkir_folder+filename
df_partition = add_partition(df_peptides,train_filenames)

filename = folder+data_folder+state_folder+peptide_folder+partition_filename
print(filename)
write_df_feather(filename,df_partition)

df_partition = find_9mer(df_partition)

filename = folder+data_folder+state_folder+peptide_folder+partition_9mer_filename
print(filename)
write_df_feather(filename,df_partition)

# train_all = df_partition[(df_partition['train_1']==1)&(df_partition['train_2']==1)&(df_partition['train_3']==1)&(df_partition['train_4']==1)&(df_partition['train_5']==1)].shape[0]
# train_9mer_all = df_partition[(df_partition['train_1_9mer']==1)&(df_partition['train_2_9mer']==1)&(df_partition['train_3_9mer']==1)&(df_partition['train_4_9mer']==1)&(df_partition['train_5_9mer']==1)].shape[0]
# print(f'{train_all}\t{train_all/df_partition.shape[0]*100:.3f}%')
# print(f'{train_9mer_all}\t{train_9mer_all/df_partition.shape[0]*100:.3f}%')