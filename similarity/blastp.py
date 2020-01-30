#!/usr/bin/python3
import pandas as pd
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import copy
import subprocess
from subprocess import Popen, PIPE
import collections
import json
import random
import sys
import time
from my_func_computerome import *


folder = "/home/projects/vaccine/people/s143849/"
data_folder = "data/"
state_folder = "intermidiate_state/"
protein_folder = "protein_peptides/"
partition_folder = "partitions/"
nnalign_folder = "nnalign/"
fasta_folder = 'one_entry/'
blast_folder = 'blast/'

df_seq_filename = 'df_seq.json'
df_seq_filter_filename = 'df_seq_filter.json'
filtered_fasta = 'filtered_fasta.fasta'
peptides_filename = '13_21_small_acc.txt'
partition_filename = 'partition.ft'
partition_9mer_filename = 'partition_9mer.ft'
top_mhc_uc_filename = 'top_mhc_uc_small.ft'
top_mhc_wc_filename = 'top_mhc_wc_small.ft'
top_protein_uc_filename = 'top_protein_uc_small.ft'
top_protein_wc_filename = 'top_protein_wc_small.ft'

filename = folder+data_folder+state_folder+df_seq_filter_filename
df_seq_filter = load_df_json(filename)
minimum = int(sys.argv[1])
maximum = int(sys.argv[2])

for c,row in df_seq_filter.iloc[minimum:maximum].iterrows():
    acc = row['acc']
    query = folder+data_folder+fasta_folder+acc+".fasta"
    subject = folder+data_folder+fasta_folder+filtered_fasta
    out = folder+data_folder+blast_folder+'blosum62/'+acc+".blast"
    print(out)
    evalue = "1000000000"
    matrix = "BLOSUM62"
    num_descriptions = "2000"
    num_alignments = "2000"
    outfmt = "6"
    string = ["blastp", "-query", query, "-subject", subject, "-out", out, "-evalue", evalue, "-matrix", matrix, "-num_alignments", num_alignments, "-outfmt", outfmt]
    process = Popen(string, universal_newlines=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

