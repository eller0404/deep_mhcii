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
import subprocess
from subprocess import Popen,PIPE

folder = "/home/projects/vaccine/people/s143849/"
data_folder = 'alternative_data/'
birkir_folder = 'birkir/'
partition_folder = 's_partition/'
nnalign_folder = 's_nnalign/'
extract_top_folder = 's_extract_top/'
input_folder = 'input/'
output_folder = 'output/'
single_allele_folder = 'single_allele/'
cell_line_folder = 'cell_line/'
random_folder = 'random/'
mean_folder = 'mean/'
jupyter_folder = 'jupyter/'

extension = sys.argv[1]
allele_folder = sys.argv[2]
method = sys.argv[3]

nnalign_filename = extension+'nnalign.ft'
partition_filename = extension+'partition_9mer.ft'
peptide_filename = extension+'13_21_acc.txt'
random_pick_filename = extension+'random.ft'
top_mhc_uc_filename = extension+method+'_top_mhc_uc.ft'
top_mhc_wc_filename = extension+method+'_top_mhc_wc.ft'
top_protein_uc_filename = extension+method+'_top_protein_uc.ft'
top_protein_wc_filename = extension+method+'_top_protein_wc.ft'
auc_input_filename = 'all_auc_score_target.txt'

df_nnalign_filename = folder+data_folder+extract_top_folder+allele_folder+nnalign_filename
df_partition_filename = folder+data_folder+partition_folder+allele_folder+partition_filename
df_peptides_filename = folder+data_folder+jupyter_folder+allele_folder+peptide_filename
write_random_pick_filename = folder+data_folder+extract_top_folder+allele_folder+random_folder+random_pick_filename
filename_mhc_uc = folder+data_folder+extract_top_folder+allele_folder+random_folder+top_mhc_uc_filename
filename_mhc_wc = folder+data_folder+extract_top_folder+allele_folder+random_folder+top_mhc_wc_filename
filename_protein_uc = folder+data_folder+extract_top_folder+allele_folder+random_folder+top_protein_uc_filename
filename_protein_wc = folder+data_folder+extract_top_folder+allele_folder+random_folder+top_protein_wc_filename
print(os.path.exists(df_nnalign_filename),df_nnalign_filename)
print(os.path.exists(df_partition_filename),df_partition_filename)
print(os.path.exists(df_peptides_filename),df_peptides_filename)
print(os.path.exists(os.path.dirname(write_random_pick_filename)),write_random_pick_filename)
print(os.path.exists(os.path.dirname(filename_mhc_wc)),filename_mhc_wc)
"""
# filenames = [
#     'cl_5-wc.log',
#     'cl_4-wc.log',
#     'cl_1-wc.log',
#     'cl_3-wc.log',
#     'cl_2-wc.log',
#     'cl_4-uc.log',
#     'cl_5-uc.log',
#     'cl_3-uc.log',
#     'cl_2-uc.log',
#     'cl_1-uc.log']
# translate_predictions = [
#     'Prediction_5_wc',
#     'Prediction_4_wc',
#     'Prediction_1_wc',
#     'Prediction_3_wc',
#     'Prediction_2_wc',
#     'Prediction_4_uc',
#     'Prediction_5_uc',
#     'Prediction_3_uc',
#     'Prediction_2_uc',
#     'Prediction_1_uc']

# for c,filename in enumerate(filenames):
#     filenames[c] = folder+data_folder+nnalign_folder+output_folder+filename

# df_nnalign = extract_nnalign(filenames,translate_predictions)
# print('extracted nnalign output')
"""
# write_df_feather(filename,df_nnalign)
df_nnalign = load_df_feather(df_nnalign_filename)
df_nnalign['target'] = 0.0

df_partition = load_df_feather(df_partition_filename)
print('partition loaded')

df_peptides = pd.read_csv(df_peptides_filename,sep='\t',header=None,usecols=[0])
df_peptides.columns = ['acc']
print('acc loaded')

df_merge = pd.concat([df_nnalign,df_peptides],axis=1)
df_merge = pd.concat([df_merge,df_partition[['train_1', 'train_2', 'train_3', 'train_4', 'train_5', 'train_1_9mer', 'train_2_9mer', 'train_3_9mer', 'train_4_9mer', 'train_5_9mer']]],axis=1)
# df_merge = merge_nn_partition(df_partition,df_nnalign,df_peptides)
# print('merged partition and nnalign')
print(df_merge.columns)
for i in range(6):
    print(i,'9mer', sum(df_merge[['train_1_9mer', 'train_2_9mer', 'train_3_9mer', 'train_4_9mer', 'train_5_9mer'] ].sum(axis=1)==i))
    print(i,'match', sum(df_merge[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].sum(axis=1)==i))
#df_merge = remove_present_in_all(df_merge,mer9=True)
print('len',len(df_merge[df_merge[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].sum(axis=1)==i]['Peptide'].unique()))
print(df_merge[df_merge[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].sum(axis=1)==i]['Peptide'].unique())

target_filter = df_merge[['train_1', 'train_2', 'train_3', 'train_4', 'train_5']].sum(axis=1)>0
df_merge.loc[target_filter,'target'] = 1.0


if method == 'mean':
    partition_cols = ['train_1_9mer', 'train_2_9mer', 'train_3_9mer', 'train_4_9mer', 'train_5_9mer']
    for context in ['wc', 'uc']:
        prediction_cols = [f'Prediction_1_{context}', f'Prediction_2_{context}', f'Prediction_3_{context}', f'Prediction_4_{context}', f'Prediction_5_{context}']
        df_bool = df_merge[partition_cols]
        df_bool[df_merge[partition_cols].sum(axis=1)==5]=0
        df_pred = df_merge[prediction_cols]
        df_pred[df_bool.to_numpy(dtype=bool)] = np.nan
        df_merge[f'Prediction_{context}'] = df_pred.mean(axis=1)

    df_lig = df_merge[['acc','Peptide','MHC_mol','Context','Prediction_uc','Prediction_wc','target']]

elif method == 'random':
    df_lig = extract_random_peptides(df_merge)
    try:
        write_df_feather(write_random_pick_filename,df_lig)
    except:
        df_lig = df_lig.reset_index()
        write_df_feather(write_random_pick_filename,df_lig)

    print('extracted random scores')

acc_group = df_lig.groupby(['acc','MHC_mol'])
for context in ['uc','wc']:
    for c,acc in enumerate(acc_group):
        acc[1][['target',f'Prediction_{context}']].to_csv('tmp.csv',header=None,index=False,sep='\t')
        process = Popen(run, universal_newlines=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        aucs.append(stdout.split('\n')[-2].split()[2])
        print(acc[0],aucs[-1])
        if c>50:
            break

    aucs = np.array(aucs,dtype='float')
    print(context,np.mean(aucs))


# df_protein_uc,df_protein_wc,df_mhc_uc,df_mhc_wc = extract_top_prediction(df_lig)

# try:
#     write_df_feather(filename_mhc_uc,df_mhc_uc)
#     write_df_feather(filename_mhc_wc,df_mhc_wc)
#     write_df_feather(filename_protein_uc,df_protein_uc)
#     write_df_feather(filename_protein_wc,df_protein_wc)
# except:
#     df_mhc_uc = df_mhc_uc.reset_index()
#     df_mhc_wc = df_mhc_wc.reset_index()
#     df_protein_uc = df_protein_uc.reset_index()
#     df_protein_wc = df_protein_wc.reset_index()
#     write_df_feather(filename_mhc_uc,df_mhc_uc)
#     write_df_feather(filename_mhc_wc,df_mhc_wc)
#     write_df_feather(filename_protein_uc,df_protein_uc)
#     write_df_feather(filename_protein_wc,df_protein_wc)


"""
train_counter = int(sys.argv[1])

merge_filename = folder+data_folder+state_folder+merge_nn_partition_filename
df_merge = load_df_feather(merge_filename)

df_merge = remove_present_in_all(df_merge,mer9=True)

mer9 = True
wc = '_uc'
if mer9:
    extension = '_9mer'
else:
    extension = ''

naughty_list = list()
print(f'train_{train_counter}{extension}\t{wc}')
l = pick_random_partition(df_merge,f'train_{train_counter}{extension}',train_counter,naughty_list)
naughty_list = naughty_list + l
df_protein_wc, df_mhc_wc = extract_top_peptides_split(df_merge,'_wc',train_counter,l)
df_protein_uc, df_mhc_uc = extract_top_peptides_split(df_merge,'_uc',train_counter,l)

start_folder = folder+data_folder+state_folder
filename_mhc_uc = folder+data_folder+state_folder+top_mhc_uc_filename
filename_mhc_wc = folder+data_folder+state_folder+top_mhc_wc_filename
filename_protein_uc = folder+data_folder+state_folder+top_protein_uc_filename
filename_protein_wc = folder+data_folder+state_folder+top_protein_wc_filename
filename_mhc_wc = f'{filename_mhc_wc[:-3]}_{train_counter}.ft'
filename_mhc_uc = f'{filename_mhc_uc[:-3]}_{train_counter}.ft'
filename_protein_wc = f'{filename_protein_wc[:-3]}_{train_counter}.ft'
filename_protein_uc = f'{filename_protein_uc[:-3]}_{train_counter}.ft'

write_df_feather(filename_mhc_uc,df_mhc_uc)
write_df_feather(filename_mhc_wc,df_mhc_wc)
write_df_feather(filename_protein_uc,df_protein_uc)
write_df_feather(filename_protein_wc,df_protein_wc)
"""
