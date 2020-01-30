#!/usr/bin/python3
from my_func_computerome import *
import pandas as pd
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import copy
import collections
import json
import time

pd.set_option('display.max_columns', None)

folder = "/home/projects/vaccine/people/s143849/"
data_folder = "data/"
state_folder = "intermidiate_state/"
protein_folder = "protein_peptides/"
partition_folder = "partitions/"
nnalign_folder = "nnalign/"
fasta_folder = 'one_entry/'
input_folder = 'inputs/'
blast_folder = 'blast/blosum62/'

ligand_filename = "MasterMerger_context_pos_19_03_08.txt"
sequence_filename = "MasterSourceProt_blastEntrez.fasta"

df_seq_filename = 'df_seq.json'
df_seq_filter_filename = 'df_seq_filter.json'
df_seq_profile_filename = 'df_seq_profile.json'
filtered_fasta = 'filtered_fasta'
peptides_filename = '13_21_small_acc.txt'
partition_filename = 'partition.ft'
partition_9mer_filename = 'partition_9mer.ft'
top_mhc_uc_filename = 'top_mhc_uc.ft'
top_mhc_wc_filename = 'top_mhc_wc.ft'
top_protein_uc_filename = 'top_protein_uc.ft'
top_protein_wc_filename = 'top_protein_wc.ft'
input_filename = 'predicted_profiles.npy'
output_filename = 'experimental_profiles.npy'
similarity_filename = 'sim_62.json'
length_filename = 'length_62.json'
score_filename = 'score_62.json'
evalue_filename = 'evalue_62.json'

filename = folder+data_folder+state_folder+df_seq_filter_filename
df_seq = load_df_json(filename)

size = df_seq['acc'].shape[0]
sim = np.zeros([size,size],dtype='float32')
length = np.zeros([size,size],dtype='int32')
score = np.zeros([size,size],dtype='float64')
evalue = np.zeros([size,size],dtype='float64')

accs = df_seq['acc'].to_numpy()
acc_pos = dict()
for c,acc in enumerate(accs):
    acc_pos[acc] = c
for acc in accs:
    df = pd.read_csv(folder+data_folder+blast_folder+acc+'.blast', sep='\t', header=None)
    df.columns = [
        'acc_query', 
        'acc_subject', 
        'identity', 
        'length', 
        'mismatch', 
        'gaps', 
        'start_pos_query', 
        'end_pos_query', 
        'start_pos_subject', 
        'end_pos_subject', 
        'evalue', 
        'bitscore']
    df = df.drop_duplicates(subset=['acc_query', 'acc_subject'],keep='first')
    pos = df.apply(lambda x: acc_pos[x['acc_subject']],axis=1).to_numpy()
    sim[acc_pos[acc],pos] = df['identity']
    length[acc_pos[acc],pos] = df['length']
    score[acc_pos[acc],pos] = df['bitscore']
    evalue[acc_pos[acc],pos] = df['evalue']

sim = pd.DataFrame(sim,index=accs,columns=accs)
length = pd.DataFrame(length,index=accs,columns=accs)
score = pd.DataFrame(score,index=accs,columns=accs)
evalue = pd.DataFrame(evalue,index=accs,columns=accs)

sim.to_json(folder+data_folder+state_folder+similarity_filename,orient='columns')
length.to_json(folder+data_folder+state_folder+length_filename,orient='columns')
score.to_json(folder+data_folder+state_folder+score_filename,orient='columns')
evalue.to_json(folder+data_folder+state_folder+evalue_filename,orient='columns')





