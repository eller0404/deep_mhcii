#!/usr/bin/python3
import pandas as pd
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import copy
import collections
import json
import time
import random
import os 
import math
import theano
import theano.tensor as T
import lasagne
from scipy.stats import pearsonr, spearmanr

def extract_seq(seq_filename):
    """extract sequence from fasta to df"""
    seq = dict()
    fasta_sequences = SeqIO.parse(open(seq_filename),'fasta')
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        seq[name] = sequence 
    df = pd.DataFrame(list(seq.items()), columns=['acc', 'seq'])
    return df

def extract_ligands(lig_filename, col_num = None,header=0):
    """extract ligands from csv to df"""
    df = pd.read_csv(lig_filename, sep='\t',header=header)
    if col_num == None:
        df['length'] = pd.Series(df['Ligand'].str.len())
    else:
        accs = df.iloc[:,col_num-1].str.split('|').map(lambda x: x[1])
        df['UID'] = accs
        df['length'] = pd.Series(df.iloc[:,col_num].str.len())
        df = df.drop(columns=[1])
    return df

def combine_seq_and_lig(df_seq,df_lig,acc_col = None):
    """add coverage to df with of protein sequences"""
    df_seq.set_index('acc', inplace=True)
    df_seq['count'] = pd.Series(df_lig['UID'].value_counts())
    df_seq['length'] = pd.Series(df_seq['seq'].str.len())
    df_seq['coverage'] = pd.Series(df_seq['count']/df_seq['length']*100)
    df_seq.reset_index(inplace=True)
    return df_seq

def add_ligands_seq(df_seq,df_lig):
    group = df_lig.groupby('UID')
    source = dict()
    ligand = dict()
    allele = dict()
    source_uniq = dict()
    allele_uniq = dict()
    for c,i in enumerate(group):
        source[i[0]] = list(i[1]['Source'])
        ligand[i[0]] = list(i[1]['Ligand'])
        allele_list = i[1]['ALID'].str.split('__').map(lambda x: x[1:])
        allele[i[0]] = list(allele_list)

    df_seq['lig'] = df_seq['acc'].map(ligand)
    df_seq['source'] = df_seq['acc'].map(source)
    df_seq['allele'] = df_seq['acc'].map(allele)
    return df_seq

def remove_zero_ligands(df):
    zeros = df['count']>0
    print(sum(zeros), 'removed from df')
    df = df[zeros]
    return df

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def uniq(df_seq):
    df_seq['allele_unique'] = df_seq['allele'].map(lambda x: np.unique(list(flatten(x))))
    df_seq['source_unique'] = df_seq['source'].map(lambda x: np.unique(x))
    df_seq['allele_unique_count'] = df_seq['allele_unique'].str.len()
    df_seq['source_unique_count'] = df_seq['source_unique'].str.len()
    return df_seq

def write_df_json(filename,df):
    if filename[-5:]!='.json':
        print('add ".json" to filename')
        return None
    df.to_json(filename)

def write_df_feather(filename,df):
    if filename[-3:]!='.ft':
        print('add ".ft" to filename')
        return None
    df.to_feather(filename)

def load_df_json(filename):
    df = pd.read_json(filename)
    return df

def load_df_feather(filename,columns=None):
    if columns == None:
        df = pd.read_feather(filename, use_threads=True)
    else:
        df = pd.read_feather(filename,columns=columns, use_threads=True)
    return df

def element_to_list(e):
    if (type(e)==list)|(type(e)==tuple):
        return e
    elif type(e).__module__ == np.__name__:
        list(e)
    else:
        return [e]

def filter_df(df,cols,limits):
    for counter,col in enumerate(cols):
        limits_col = element_to_list(limits[counter])
        df = df[df[col]>limits_col[0]]
        if len(limits_col)>1:
            df = df[df[col]<limits_col[1]]
        elif len(limits_col)>2:
            print('Error: to many limits')
            print(limits_col,len(limits_col))
            return None
    return df

def plot_hist(df,col,bins=None,xlim=None,text_size=30,figsize=(30,10),xlabel=None,ylabel=None):
    plt.rc('xtick', labelsize=text_size)
    plt.rc('ytick', labelsize=text_size)
    plt.rc('axes', titlesize=text_size)
    f, ax = plt.subplots(figsize=figsize)
    if bins == None:
        bins = np.arange(0, df[col].max()+1,1)
    h = ax.hist(x=df[col], bins=bins)
    title = col
    ax.set_title(title)
    ax.grid(b=True)
    if xlim!=None:
        ax.set_xlim(0,xlim)
    if xlabel!=None:
        ax.set_xlabel(xlabel)
    if ylabel!=None:
        ax.set_ylabel(ylabel)
    return h,ax

def plot_bars(df,col,text_size=30,figsize=(30,10),acc=False,limits=None):
    plt.rc('xtick', labelsize=text_size)
    plt.rc('ytick', labelsize=text_size)
    plt.rc('axes', titlesize=text_size)
    f, ax = plt.subplots(figsize=figsize)
    if not acc:
        uniq = np.unique(list(flatten(df[col])))
        heights = list()
        for c,i in enumerate(uniq):
            heights.append(sum(df[col].map(lambda x: i in x)))
        x = [x for y,x in sorted(zip(heights,uniq))]   
        y = sorted(heights)  
        title = col
    elif acc:
        df = df.sort_values(by=['coverage'],ascending=False)
        if limits!=None:
            df = df.iloc[limits[0]:limits[1],:]
        x = df['acc']
        y = df['coverage']
        title = 'ligands per protein'
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, rotation=90, ha='center')
    ax.bar(x=np.arange(len(x)), height = y)   
    ax.set_title(title)
    return x,y,ax

def line_profile(row):
    acc = row['acc']
    acc = row['acc']
    seq = row['seq']
    ligands = row['lig']
    profile = np.zeros(len(seq))
    no_match = list()
    for lig in ligands:
        try:
            start = seq.index(lig)
        except:
            if acc not in no_match:
                print(acc)
            no_match.append(acc)
            continue
        end = start+len(lig)
        profile[start:end] = profile[start:end]+1
    return profile

def profile(df):
    df['profile'] = df.apply(lambda x: line_profile(x))
    return df

def write_fasta(df,path,seperate=False,filename=None):
    if not seperate:
        if filename == None:
            print('You must combine "seperate=False" and "filename"')
            return None
        if ('.' in filename):
            print(f'Do not include file format: "{filename[filename.index("."):]}"')
            return None
        with open(f'{path}{filename}.fasta', 'w') as outfile:
            print(len(df['acc']))
            outfile.write('\n'.join(flatten(list(zip('>'+df['acc'],df['seq'])))))
    else:
        if filename != None:
            print('Do not combine "seperate=True" and "filename"')
        for counter,row in enumerate(zip('>'+df['acc'],df['seq'])):
            acc = row[0]
            seq = row[1]
            with open(f'{path}{acc[1:]}.fasta', 'w') as outfile:
                outfile.write(f'{acc}\n{seq}\n')

def filter_lig(df_lig,df_seq_filter):
    df_lig_filter = df_lig[df_lig['UID'].isin(df_seq_filter['acc'].values)]
    return df_lig_filter


def make_mers(acc,seq,alleles,limits):
    #acc = row['acc']
    #pritn(acc)
    #seq = row['seq']
    #alleles = row['allele_unique']
    peptides = dict()
    first = True
    for allele in alleles:
        for aa_pos in range(len(seq)):
            for length in range(limits[0],limits[1]+1):
                if aa_pos+length>len(seq):
                    continue
                if aa_pos-3<0:
                    Xs = 'X'*(3-aa_pos)
                    context = Xs+seq[0:aa_pos+3]+seq[aa_pos+length-3:aa_pos+length+3]
                elif aa_pos+length+3>len(seq):
                    Xs = 'X'*(3-(len(seq)-(aa_pos+length)))
                    context = seq[aa_pos-3:aa_pos+3]+seq[aa_pos+length-3:len(seq)]+Xs
                else:
                    context = seq[aa_pos-3:aa_pos+3]+seq[aa_pos+length-3:aa_pos+length+3]


                if first:
                    peptides[acc] = [[acc,aa_pos,len(seq)-aa_pos-length,seq[aa_pos:aa_pos+length],1.0,allele,context]]
                    first = False
                else:
                    peptides[acc].append([acc,aa_pos,len(seq)-aa_pos-length,seq[aa_pos:aa_pos+length],1.0,allele,context])
    return peptides[acc]

def list_to_string(l):
    l = [str(i) for i in l]
    return l

def write_mers(df,filename,limits=(13,21)):
    with open(filename+'_acc.txt', 'w') as outfile_acc:
        with open(filename+'.txt', 'w') as outfile:
            lines = df.apply(lambda x: make_mers(x['acc'],x['seq'],x['allele_unique'],limits), axis=1)
            n_lines = df.shape[0]
            print('constucting peptides done')
            start_time = time.time()
            for c,line in enumerate(lines):
                if c%100==0:
                    print(f'{c/n_lines*100:.3f}%\t{time.time()-start_time:.3f}')
                    start_time = time.time()
                output_acc = '\n'.join(['\t'.join(list_to_string(j)) for j in line])
                output = '\n'.join(['\t'.join(list_to_string(j[3:])) for j in line])
                outfile_acc.write(output_acc+'\n')
                outfile.write(output+'\n')

def in_train(df_peptides,train_filename,counter):
    with open(f'{train_filename}{counter}.txt', 'r') as infile:
        df_train = pd.read_csv(infile, sep='\t', header=None)
    df_train.columns = ('Peptide', 'affinity', 'MHC_mol')
    train_col = f'{os.path.basename(train_filename)}_{counter}'
    print(train_col)
    df_peptides[train_col] = 0
    if os.path.basename(train_filename) == 'train_EL':
        MHC = df_train['MHC_mol'].str.split('__')
        df_train['MHC1'] = MHC.str[1]
        df_train['MHC2'] = MHC.str[2]
        df_train['MHC3'] = MHC.str[3]
        for j in range(1,4):
            print(j,'j out of 3')
            for i in range(1,4):
                start_time = time.time()
                print(i,'i out of 3')
                set1 = set([tuple(x) for x in df_peptides[['Peptide',f'MHC{i}']].to_numpy()])
                set2 = set([tuple(x) for x in df_train[['Peptide',f'MHC{j}']].to_numpy()])
                intersect = set1.intersection(set2)
                if len(intersect) != 0:
                    df_peptides.set_index(['Peptide',f'MHC{i}'], inplace=True)
                    df_peptides.loc[intersect,train_col] = 1
                    df_peptides.reset_index(inplace=True)
                else:
                    print('PROBLEM!!!!!!!!!!!!!')
                    print('no intersection between', i,j)
                print(time.time()-start_time)
                start_time = time.time()
    else:
        for i in range(1,4):
            display(df_peptides.head())
            set1 = set([tuple(x) for x in df_peptides[['Peptide',f'MHC{i}']].to_numpy()])
            set2 = set([tuple(x) for x in df_train[['Peptide',f'MHC_mol']].to_numpy()])
            intersect = set1.intersection(set2)
            print(intersect)
            if len(intersect)==0:
                print(i)
                break
            df_peptides.set_index(['Peptide',f'MHC{i}'], inplace=True)
            df_peptides.loc[intersect,train_col] = 1
            df_peptides.reset_index(inplace=True)
    return df_peptides

def add_partition(df,train_filenames):
    for filename in train_filenames:
        for i in range(1,6):
            print(i)
            df = in_train(df,filename,i)
    return df

# def in_train(df_peptides,train_filename,counter,exist):
#     start_time = time.time()
#     with open(f'{train_filename}{counter}.txt', 'r') as infile:
#         df_train = pd.read_csv(infile, sep='\t', header=None)
#     df_train.columns = ('Peptide', 'affinity', 'MHC_mol')
#     train_col = f'{train_filename}_{counter}'
#     intersect = set(df_peptides['Peptide']).intersection(set(df_train['Peptide']))
#     if not exist:
#         df_peptides[train_col] = 0
#     df_peptides.set_index('Peptide', inplace=True)
#     df_peptides.loc[intersect,train_col] = 1
#     df_peptides.reset_index(inplace=True)
#     print(round(time.time()-start_time,3),'s')
#     return df_peptides

# def add_partition(df,train_filenames):
#     exist = False
#     for filename in train_filenames:
#         for i in range(1,6):
#             print(filename,i,'out of',5)
#             df = in_train(df,filename,i,exist)
#         exist = True
#     return df

def find_9mer(df):
    start_time = time.time()
    group = df.groupby('acc')
    for i in range(1,6):
        in_train = np.zeros(df.shape[0])
        for acc,df_acc in group:
            positive_EL = df_acc[f'train_EL_{i}']==1
            positive_BA = df_acc[f'train_BA_{i}']==1
            positive = positive_EL|positive_BA
            peptides = df_acc[positive]['Peptide'].unique()
            for peptide in peptides:
                for pos in range(0,len(peptide)-9+1):
                    in_train_idx = df_acc[df_acc['Peptide'].str.contains(peptide[pos:pos+9],regex=False)].index
                    in_train[in_train_idx] = 1
        df[f'train_{i}_9mer']=in_train

    return df

# def find_9mer(df,train_filenames):
#     start_time = time.time()
#     for filename in train_filenames:
#         for i in range(1,6):
#             in_train = np.zeros(df.shape[0])
#             acc_uniq = np.unique(list(df['acc'].values))
#             for c,acc in enumerate(acc_uniq):
#                 acc_filter = df['acc']==acc
#                 positive_filter = df[f'{filename}_{i}']==1
#                 df_acc = df[acc_filter]
#                 train_uniq = df[acc_filter&positive_filter]['Peptide'].unique()
#                 for peptide in train_uniq:
#                     for pos in range(0,len(peptide)-9+1):
#                         in_train_idx = df_acc[df_acc['Peptide'].str.contains(peptide[pos:pos+9],regex=False)].index
#                         in_train[in_train_idx] = 1
#                 print(f'{c/len(acc_uniq)*100:.3f}%\t{i}\t{acc}\t{len(train_uniq)}\t{time.time()-start_time:.3f}')
#                 start_time = time.time()
#             df[f'{filename}_{i}_9mer']=in_train
#     return df

def extract_nnalign(filenames, translate_predictions):
    with open(filenames[0], 'r') as infile:
        for line in infile:
            if line[0] == '#':
                header = line
            else:
                break
    header = header.split()[1:]
    
    first = True
    for counter,filename in enumerate(filenames):
        print(f'{counter}\t{os.path.basename(filename)}\t{counter/len(filenames)*100:.3f}')
        if 'wc' in filename:
            extra_row = 1
            usecols = [2,3,4,11,14]
            cols = ['target',f'Prediction_{counter}','Peptide','MHC_mol','Context']
        else:
            extra_row = 0
            usecols = [2,3,4,11]
            cols = ['target',f'Prediction_{counter}','Peptide','MHC_mol']
        df = pd.read_csv(filename, sep=' ', header=None, skiprows=272+extra_row,usecols=usecols)
        #diff = df.shape[1]-len(header)
        #cols = header + list(np.arange(diff))
        df.columns = cols
        if first:
            df_nnalign = df
            #df_nnalign.rename(columns={'Prediction':f'Prediction_{counter}'},inplace=True)
            first = False
        else:
            df_nnalign[f'Prediction_{counter}'] = df[f'Prediction_{counter}']
    
    translate_pred = dict()
    for c,translate in enumerate(translate_predictions):
        translate_pred[f'Prediction_{c}'] = translate
        
    df_nnalign.rename(
        columns=translate_pred, inplace=True)
    #cols_float = translate_predictions
    #df_nnalign[cols_float] = df_nnalign[cols_float].astype('float32')
    #cols_all = df_nnalign.columns
    #cols_used = ['Peptide','MHC_mol','Context']+cols_float
    #df_nnalign = df_nnalign[cols_used]
    return df_nnalign

def merge_nn_partition(df_partition, df_nnalign,df_nnalign_input):
    if df_partition.shape[0] != df_nnalign.shape[0]:
        print('not same size')
        print('partition size',df_partition.shape[0])
        print('nnalign size',df_nnalign.shape[0])

    if df_nnalign.shape[0] != df_nnalign_input.shape[0]:
        print('acc cannot be added due to not same size')
        print('acc size',df_nnalign_input.shape[0])
        return None
        
    df_nnalign = pd.concat([df_nnalign,df_nnalign_input['acc']],axis=1)
    print('partition columns:',df_partition.columns)
    print('nnalign columns:',df_nnalign.columns)
    print('unique nnalign rows:',df_nnalign.drop_duplicates(['Peptide','acc']).shape[0])
    print('unique partition rows:',df_partition.drop_duplicates(['Peptide','acc']).shape[0])
    df = pd.merge(df_nnalign, df_partition, on=['Peptide','acc'], how='inner',suffixes=('','_partition')).drop(columns=['Context_partition', 'MHC_mol_partition'])
    print('merge df shape',df.shape)
    df.columns = df.columns.astype(str)

    return df

def remove_present_in_all(df,mer9=False):
    train_filter = df[f'train_1']!=3 # true series
    positive_filter = df[f'train_1']==3 # false series
    if mer9:
        extension = '_9mer'
    for i in range(1,6):
        train_filter = train_filter&(df[f'train_{i}{extension}']==1)
        if i < 3:
            positive_filter = positive_filter|((df[f'train_{i}{extension}']==1))
    print(f'peptides before: {df.shape[0]}')
    print(f'peptides removed: {sum(train_filter)}')
    print(f'peptides after: {df.shape[0]-sum(train_filter)}')
    train_all = df[train_filter].index
    df = df.drop(index=list(train_all))
    print(f'positive rows: {sum(positive_filter)}')
    return df

"""
def pick_random_partition(df,filter_col,counter,naughty_list):
    random.seed(0)
    nrows_all = df.shape[0]
    df_filter = df[df[filter_col]==0]
    if '1' in filter_col:
        other_col = filter_col.replace(str(counter),'2')
    else:
        other_col = filter_col.replace(str(counter),'1')
    positve_samples = list(df[(df[filter_col]==0)&(df[other_col]==1)].index)
    print(f'peptides present in other 4 partitions: {len(positve_samples)}')
    nrows_filter = df_filter.shape[0]
    print(nrows_filter, 'not present in partition')
    diff = list(set(df_filter.index)-set(naughty_list)-set(positve_samples))
    print('choose between', len(diff))
    fifth = int(nrows_all/5)-len(positve_samples)
    rest = nrows_all-len(naughty_list)-len(positve_samples)
    print('options',fifth,rest)
    minimum = min(int(nrows_all/5)-len(positve_samples),nrows_all-len(naughty_list)-len(positve_samples))
    minimum = max(0,minimum)
    #if len(positve_samples)+minimum > int(nrows_all/5):
    #    minimum = 0
    print(minimum,'peptides chosen', minimum+len(positve_samples), 'overall')
    
    l = positve_samples+random.sample(diff, k=minimum)
    return l

def extract_top_prediction(df,filter_col,sort_col,naughty_list,counter,wc):
    start_time = time.time()
    df_filter = df[df[filter_col]==0]
    l = pick_random_partition(df,filter_col,counter,naughty_list)
    df_filter = df_filter.loc[l,:]
    accs = df['acc'].unique()
    first_protein = True
    first_mhc = True
    for c, acc in enumerate(accs):
        df_acc = df_filter[df_filter['acc']==acc]
        df_acc = df_acc.rename(columns={sort_col:'Prediction'})
        if c % 10 == 0:
            print(f'{c}\t{c/len(accs)*100:.2f} %\t{c}\t{wc}\t{df_acc.shape[0]}\t{time.time()-start_time:.3f}')
            start_time = time.time()
        alleles = df_acc['MHC_mol'].unique()
        npeps = dict()
        for allele in alleles:
            df_allele = df_acc[df_acc['MHC_mol']==allele]
            npeps[allele] = df_allele.shape[0]
            df_allele = df_allele.sort_values('Prediction',ascending=False)[:max(round(df_allele.shape[0]/100),1)]
            df_allele['partition'] = counter
            if first_mhc:
                df_mhc = df_allele[['acc','Peptide','MHC_mol','Prediction','partition']]
                first_mhc=False
            else:
                df_mhc = df_mhc.append(df_allele[['acc','Peptide','MHC_mol','Prediction','partition']], ignore_index = True)
        #print(f'{df_acc.shape[0]}\t{acc}')
        #print(list(npeps.keys()))
        #print(list(npeps.values()))
        #print('****')
        df_acc = df_acc.sort_values('Prediction',ascending=False)[:max(int(df_acc.shape[0]/100),1)]
        df_acc['partition'] = counter
        if first_protein:
            df_protein = df_acc[['acc','Peptide','MHC_mol','Prediction','partition']]
            first_protein = False
        else:
            df_protein = df_protein.append(df_acc[['acc','Peptide','MHC_mol','Prediction','partition']], ignore_index = True)
    #df_protein[sort_col] = df_protein[sort_col].astype('float32')
    #df_protein['partition'] = df_protein['partition'].astype('int8')
    #df_mhc[sort_col] = df_mhc[sort_col].astype('float32')
    df_mhc['partition'] = df_mhc['partition'].astype('int8')
    return df_protein,df_mhc,l+naughty_list

def extract_top_prediction_split(df,filter_col,sort_col,l,counter,wc):
    start_time = time.time()
    df_filter = df[df[filter_col]==0]
    df_filter = df_filter.loc[l,:]
    accs = df['acc'].unique()
    first_protein = True
    first_mhc = True
    for c, acc in enumerate(accs):
        df_acc = df_filter[df_filter['acc']==acc]
        df_acc = df_acc.rename(columns={sort_col:'Prediction'})
        if c % 10 == 0:
            print(f'{c}\t{c/len(accs)*100:.2f} %\t{c}\t{wc}\t{df_acc.shape[0]}\t{time.time()-start_time:.3f}')
            start_time = time.time()
        alleles = df_acc['MHC_mol'].unique()
        npeps = dict()
        for allele in alleles:
            df_allele = df_acc[df_acc['MHC_mol']==allele]
            npeps[allele] = df_allele.shape[0]
            df_allele = df_allele.sort_values('Prediction',ascending=False)[:max(round(df_allele.shape[0]/100),1)]
            df_allele['partition'] = counter
            if first_mhc:
                df_mhc = df_allele[['acc','Peptide','MHC_mol','Prediction','partition']]
                first_mhc=False
            else:
                df_mhc = df_mhc.append(df_allele[['acc','Peptide','MHC_mol','Prediction','partition']], ignore_index = True)
        #print(f'{df_acc.shape[0]}\t{acc}')
        #print(list(npeps.keys()))
        #print(list(npeps.values()))
        #print('****')
        df_acc = df_acc.sort_values('Prediction',ascending=False)[:max(int(df_acc.shape[0]/100),1)]
        df_acc['partition'] = counter
        if first_protein:
            df_protein = df_acc[['acc','Peptide','MHC_mol','Prediction','partition']]
            first_protein = False
        else:
            df_protein = df_protein.append(df_acc[['acc','Peptide','MHC_mol','Prediction','partition']], ignore_index = True)
    #df_protein[sort_col] = df_protein[sort_col].astype('float32')
    #df_protein['partition'] = df_protein['partition'].astype('int8')
    #df_mhc[sort_col] = df_mhc[sort_col].astype('float32')
    df_mhc['partition'] = df_mhc['partition'].astype('int8')
    return df_protein,df_mhc

def extract_top_peptides_split(df,wc,train_counter,naughty_list,mer9=False):
    if mer9:
        extension = '_9mer'
    else:
        extension = ''

    df_protein,df_mhc = extract_top_prediction_split(df,f'train_{train_counter}{extension}',f'Prediction_{train_counter}{wc}',naughty_list,train_counter,wc)
    
    return df_protein, df_mhc

def extract_top_peptides(df,mer9=False):
    start_time = time.time()
    df_list = list()
    wc = '_uc'
    if mer9:
        extension = '_9mer'
    else:
        extension = ''
    for wc_counter in range(2):
        naughty_list = []
        first = True
        for train_counter in range(1,6):
            print(train_counter, '***')
            df_protein,df_mhc,naughty_list = extract_top_prediction(df,f'train_{train_counter}{extension}',f'Prediction_{train_counter}{wc}',naughty_list,train_counter,wc)
            if first:
                df_protein_top = df_protein
                df_mhc_top = df_mhc
                first = False
            else:
                df_protein_top = df_protein_top.append(df_protein, ignore_index = True)
                df_mhc_top = df_mhc_top.append(df_mhc, ignore_index = True)
            print(train_counter, '!!!')
            start_time = time.time()
        if wc_counter == 0:
            df_protein_uc = df_protein_top
            df_mhc_uc = df_mhc_top
        else:
            df_protein_wc = df_protein_top
            df_mhc_wc = df_mhc_top
        wc = '_wc'
    print(len(naughty_list),len(np.unique(naughty_list)),df.shape[0])
    return df_protein_uc, df_mhc_uc, df_protein_wc, df_mhc_wc

def extract_top_splits(df,filename_mhc_wc,filename_mhc_uc,filename_protein_wc,filename_protein_uc):
    mer9 = True
    wc = '_uc'
    if mer9:
        extension = '_9mer'
    else:
        extension = ''

    naughty_list = list()
    for train_counter in range(1,6):
        print(f'train_{train_counter}{extension}\t{wc}')
        l = pick_random_partition(df,f'train_{train_counter}{extension}',train_counter,naughty_list)
        naughty_list = naughty_list + l
        df_protein_wc, df_mhc_wc = extract_top_peptides_split(df,'_wc',train_counter,l)
        df_protein_uc, df_mhc_uc = extract_top_peptides_split(df,'_uc',train_counter,l)

        start_folder = folder+data_folder+state_folder
        filename_mhc_wc = f'{filename_mhc_wc[:-3]}_{train_counter}.ft'
        filename_mhc_uc = f'{filename_mhc_uc[:-3]}_{train_counter}.ft'
        filename_protein_wc = f'{filename_protein_wc[:-3]}_{train_counter}.ft'
        filename_protein_uc = f'{filename_protein_uc[:-3]}_{train_counter}.ft'

        write_df_feather(filename_mhc_uc,df_mhc_uc)
        write_df_feather(filename_mhc_wc,df_mhc_wc)
        write_df_feather(filename_protein_uc,df_protein_uc)
        write_df_feather(filename_protein_wc,df_protein_wc)
    if len(naughty_list)!=df.shape[0]:
        print('ERROR: ',len(naughty_list),df.shape[0])
    print('****')

def load_tops(filename_mhc_wc,filename_mhc_uc,filename_protein_wc,filename_protein_uc):
    first = True
    start_folder = folder+data_folder+state_folder
    for train_counter in range(1,6):
        filename_mhc_wc_count = f'{filename_mhc_wc[:-3]}_{train_counter}.ft'
        filename_mhc_uc_count = f'{filename_mhc_uc[:-3]}_{train_counter}.ft'
        filename_protein_wc_count = f'{filename_protein_wc[:-3]}_{train_counter}.ft'
        filename_protein_uc_count = f'{filename_protein_uc[:-3]}_{train_counter}.ft'
        df_mhc_uc_split = load_df_feather(filename_mhc_uc_count)
        df_mhc_wc_split = load_df_feather(filename_mhc_wc_count)
        df_protein_uc_split = load_df_feather(filename_protein_uc_count)
        df_protein_wc_split = load_df_feather(filename_protein_wc_count)
        if first:
            df_mhc_uc = df_mhc_uc_split
            df_mhc_wc = df_mhc_wc_split
            df_protein_uc = df_protein_uc_split
            df_protein_wc = df_protein_wc_split
        else:
            df_mhc_uc = df_mhc_uc.append(df_mhc_uc_split,ignore_index=True)
            df_mhc_wc = df_mhc_wc.append(df_mhc_wc_split,ignore_index=True)
            df_protein_uc = df_protein_uc.append(df_protein_uc_split,ignore_index=True)
            df_protein_wc = df_protein_wc.append(df_protein_wc_split,ignore_index=True)
        first = False
    return df_mhc_uc, df_mhc_wc, df_protein_uc, df_protein_wc
"""

def extract_random_peptides(df):
    if 'train_1_9mer' not in df.columns:
        print('must have columns with information about partitions')
        return None

    # find index of peptides which must use specific predictor
    print('finding index of positive peptides')
    idxs = list()
    for i in range(1,6):
        if i == 2:
            other_col = 1
        else:
            other_col = 2
        l = (df[f'train_{i}_9mer']==0)&(df[f'train_{other_col}_9mer']==1)
        idxs.append(list(l[l].index))
    idxs = np.array(idxs)
    
    # if some partitions contain more than 1/5 of the peptides the rest of the peptides is distributed equally
    print('find amount of peptides from each partition')
    lengths = np.array(list(map(len,idxs)))
    total = df.shape[0]
    counter = 5
    for i in lengths[lengths>math.ceil(df.shape[0]/5)]:
        total -= i
        counter -= 1
    total_sum = math.ceil(total/counter)
    
    # indexes to choose between
    print('find peptides to choose between')
    choose_between = set(df.index)
    for i in idxs:
        choose_between = choose_between-set(i)
        
    # find random index from each partition
    print('find random index for negative peptides for each partition')
    random_left = len(choose_between)
    for c,i in enumerate(idxs):
        if len(i) > total_sum:
            continue
        random_amount = total_sum-len(i)
        if random_amount > random_left:
            random_amount = random_left
        random_left -= random_amount
        chosen = random.sample(choose_between, k=random_amount)
        choose_between = choose_between-set(chosen)
        idxs[c] = i+chosen
        
    # extract indexes from each partition
    print('extract partitions')

    df['Prediction_uc'] = np.nan
    df['Prediction_wc'] = np.nan
    df['Partition'] = np.nan

    for i in range(1,6):
        print(i, 'out of', 5)
        df.loc[idxs[i-1],'Prediction_uc'] = df.loc[idxs[i-1],f'Prediction_{i}_uc']
        df.loc[idxs[i-1],'Prediction_wc'] = df.loc[idxs[i-1],f'Prediction_{i}_wc']
        df.loc[idxs[i-1],'Partition'] = i
        df = df.drop(columns=[f'Prediction_{i}_uc',f'Prediction_{i}_wc'])
    # df_predict = pd.DataFrame([])
    # for i in range(1,6):
    #     df_choose = df.loc[idxs[i-1],['Peptide','MHC_mol','Context','acc',f'Prediction_{i}_wc',f'Prediction_{i}_uc']]
    #     df_choose = df_choose.rename(columns={f'Prediction_{i}_wc':'Prediction_wc',f'Prediction_{i}_uc':'Prediction_uc'})
    #     df_predict = pd.concat([df_predict,df_choose],sort=True)
    #     df_predict.loc[idxs[i-1],'Partition'] = i
    return df

def extract_top_prediction(df):
    df_pep_wc_all = pd.DataFrame([])
    df_pep_uc_all = pd.DataFrame([])
    df_mhc_uc_all = pd.DataFrame([])
    df_mhc_wc_all = pd.DataFrame([])
    
    df_wc = df.sort_values(by='Prediction_wc',ascending=False)
    df_uc = df.sort_values(by='Prediction_uc',ascending=False)
    
    group_uc = df_uc.groupby(by='acc')
    group_wc = df_wc.groupby(by='acc')
    
    for acc,i in group_uc:
        df_pep_uc = i[:i.shape[0]//100]
        df_pep_uc_all = pd.concat([df_pep_uc_all,df_pep_uc],sort=False)
        group_mhc = i.groupby(by='MHC_mol')
        for mhc,j in group_mhc:
            df_mhc_uc = j[:j.shape[0]//100]
            df_mhc_uc_all = pd.concat([df_mhc_uc_all,df_mhc_uc],sort=False)
            
    for acc,i in group_wc:
        df_pep_wc = i[:i.shape[0]//100]
        df_pep_wc_all = pd.concat([df_pep_wc_all,df_pep_wc],sort=False)
        group_mhc = i.groupby(by='MHC_mol')
        for mhc,j in group_mhc:
            df_mhc_wc = j[:j.shape[0]//100]
            df_mhc_wc_all = pd.concat([df_mhc_wc_all,df_mhc_wc],sort=False)
    
    return df_pep_uc_all,df_pep_wc_all,df_mhc_uc_all,df_mhc_wc_all

def write_csv(df,filename):
    df.to_csv(filename, sep='\t', index=False, header=False)

def add_peptides(df_seq,df_peptide,extension,speed=True,score=False):
    """add peptides, their associated mhc,score and partition as columns to each protein"""
    df_seq['peptide'+extension] = df_seq['acc'].apply(lambda x: list(df_peptide[df_peptide['acc']==x]['Peptide']))
    df_seq['peptide_mhc'+extension] = df_seq['acc'].apply(lambda x: list(df_peptide[df_peptide['acc']==x]['MHC_mol']))
    if score:
        df_seq['peptide_score_mean'+extension] = df_seq['acc'].apply(lambda x: df_peptide[df_peptide['acc']==x]['Prediction'].mean())
        df_seq['peptide_score_mean'+extension] = df_seq['peptide_score_mean'+extension].astype('float32')
    if not speed:
        df_seq['peptide_sort'+extension] = df_seq['peptide'+extension].apply(lambda x: sorted(x))
        df_seq['peptide_score'+extension] = df_seq['acc'].apply(lambda x: np.array(df_peptide[df_peptide['acc']==x]['Prediction'],dtype='float32'))
        df_seq['peptide_partition'+extension] = df_seq['acc'].apply(lambda x: np.array(df_peptide[df_peptide['acc']==x]['partition'],dtype='int8'))
    return df_seq

def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append(set()) #different object reference each time
    return list_of_objects

def add_zero_positions(df,extensions):
    """add columns with positions"""
    for extension in extensions:
        df['peptide_counter'+extension] = df.seq.apply(lambda x: np.array(np.zeros(len(x),dtype='int16')))
        df['mhc_counter'+extension] = df.seq.apply(lambda x: np.array(np.zeros(len(x),dtype='int16')))
        df['mhcs'+extension] = df.seq.apply(lambda x: init_list_of_objects(len(x)))
    return df

def match_peptides(seq,peptides):
    """match peptides to protein sequence"""
    positions = np.array(np.zeros(len(seq),dtype='int16'))
    for peptide in peptides:
        start_pos = seq.find(peptide)
        while start_pos != -1:
            positions[start_pos:start_pos+len(peptide)]+=1
            start_pos = seq.find(peptide,start_pos+1)
    return positions

def match_mhc(seq,peptides,mhcs):
    """match mhc to protein sequence based on peptides"""
    positions = init_list_of_objects(len(seq))
    for counter,peptide in enumerate(peptides):
        start_pos = seq.find(peptide)
        while start_pos != -1:
            mhc = mhcs[counter]
            pos = start_pos
            while pos != start_pos+len(peptide):
                positions[pos].update(set(element_to_list(mhc)))
                pos += 1
            start_pos = seq.find(peptide,start_pos+1)
    return positions

def match_mhc_counter(position_mhcs):
    """count number of associated mhc for each position"""
    positions = np.array(np.zeros(len(position_mhcs),dtype='int16'))
    for counter,mhcs in enumerate(position_mhcs):
        positions[counter] = len(mhcs)
    return positions

def count_positions(df,extensions):
    """find peptide and mhc positions and count number of mhcs"""
    cols = list()
    for extension in extensions:
        if extension == '_train':
            peptide_col = 'lig'
            mhc_col = 'allele'
        else:
            peptide_col = 'peptide'+extension
            mhc_col = 'peptide_mhc'+extension
        df['peptide_counter'+extension] = df.apply(lambda x: np.array(match_peptides(x['seq'],x[peptide_col]),dtype='int16'), axis=1)
        df['mhcs'+extension] = df.apply(lambda x: match_mhc(x['seq'], x[peptide_col], x[mhc_col]), axis=1)
        df['mhc_counter'+extension] = df.apply(lambda x: np.array(match_mhc_counter(x['mhcs'+extension]),dtype='int16'), axis=1)
        #cols.extend(['peptide_counter'+extension,'mhcs'+extension,'mhc_counter'+extension])
    #df[cols] = df[cols].astype('int8')
    return df

def normalize_count(df,extensions):
    """normalize counters at each position in the protein"""
    for extension in extensions:
        df['mhc_counter_norm'+extension] = df['mhc_counter'+extension].apply(lambda x: (np.array(x,dtype='float32')-min(x))/(max(x)-min(x)))
        df['peptide_counter_norm'+extension] = df['peptide_counter'+extension].apply(lambda x: (np.array(x,dtype='float32')-min(x))/(max(x)-min(x)))
    return df

def pcc(train_norm, predict_norm):
    """pcc calculator"""
    return pearsonr(train_norm,predict_norm)[0]

def calculate_pcc(df,extensions):
    cols = list()
    for extension in extensions:
        df['pcc_peptide'+extension] = df.apply(lambda x: pcc(x['peptide_counter_norm_train'], x['peptide_counter_norm'+extension]), axis=1)
        df['pcc_mhc'+extension] = df.apply(lambda x: pcc(x['mhc_counter_norm_train'], x['mhc_counter_norm'+extension]), axis=1)
        cols.extend(['pcc_peptide'+extension,'pcc_mhc'+extension])
    df[cols] = df[cols].astype('float32')
    return df

def peaks(counts):
    """calculate amount of peaks in sequence"""
    previous = 0
    peaks = 0
    for i in counts:
        if (i != 0) & (previous == 0):
            peaks += 1
        previous = i
    return peaks

def add_peaks(df,extensions):
    for extension in extensions:
        df['peaks'+extension] = df['peptide_counter_norm'+extension].apply(lambda x: peaks(x))
        df['peaks'+extension] = df['peaks'+extension].astype('int8')
    return df

"""test intervals for removing bad permorming proteins"""
length = np.arange(100,131,5)
count = np.arange(5,31,1)
coverage = np.arange(5,11,1)
allele = np.arange(0,6,1)
peaks = np.arange(0,7,1)
source = np.arange(0,6,1)
(len(length)*len(count)*len(coverage)*len(allele)*len(peaks)*len(source))/(10**6)

def grid_search(df,pcc_col,length,count,coverage,allele,source,max_count=None):
    """make dict with combination of parameters"""
    if max_count == None:
        max_count = len(length)*len(count)*len(coverage)*len(allele)*len(peaks)*len(source)
    counter = 0
    negative_filter = df[pcc_col]<0
    pcc = dict()
    negative_removed = dict()
    total_remvoed = dict()
    start = time.time()
    for a in length:
        filtera = df['length']<a
        for b in count:
            filterb = df['count']<b
            filterab = filtera|filterb
            for c in coverage:
                filterc = df['coverage']<c
                filterabc = filterab|filterc
                for d in allele:
                    filterd = df['allele_unique_count']<d
                    filterabcd = filterabc|filterd
                    for e in source:
                        filterf = df['source_unique_count']<e
                        filterabcdef = filterabcd|filtere
                        negative[(a,b,c,d,e)] = sum(filterabcde&negative_filter)
                        total[(a,b,c,d,e)] = sum(filterabcde)
                        pcc[(a,b,c,d,e)] = df[~filterabcde][pcc_col].mean()
                        
                        counter += 1
                        if counter % 3000==0:
                            diff = time.time()-start
                            print(f'{counter/max_count*100:.3f}%\t{diff:.3f}sec')
                            start = time.time()
                        if counter > max_count:
                            break
                    if counter > max_count:
                        break
                if counter > max_count:
                    break
            if counter > max_count:
                break
        if counter > max_count:
            break
    return negative_removed,total_remvoed,pcc


def merge_grid(negative_removed,total_removed,pcc,pcc_col):
    """combine grid search results"""
    cols = ['length','count','coverage','allele','source']
    
    df_pcc = pd.Series(pcc).reset_index()
    df_pcc.columns = cols + [pcc_col]
    
    df_negative = pd.Series(negative_removed).reset_index()
    df_negative.columns = cols + ['negative_removed']
    
    df_total = pd.Series(total).reset_index()
    df_total.columns = cols + ['total_removed']
    
    df = pd.merge(df_total, df_negative, left_on=cols, right_on=cols, how='outer')
    df = pd.merge(df, df_pcc, left_on=cols, right_on=cols, how='outer')
    
    df['ratio'] = df['negative']/df['total']
    
    return df

def pcc_scatter(df,x_col,extensions,xmax,filter_cols=None,filter_limits=None,figsize=(12,12)):
    """pcc overview with different prediction methods"""
    
    print('(method/context/promiscuity)\ta*x+b')
    fig,ax=plt.subplots(2,len(extensions),figsize=figsize)
    fig.suptitle(f'pcc >< {x_col} (method/context/promiscuity)')
    y_cols = list()
    wcs = list()
    promiscuities = list()
    rows = list()
    cols = list()
    methods = list()
    for c,extension in enumerate(extensions):
        y_cols.extend(['pcc_peptide'+extension,'pcc_mhc'+extension])
        promiscuities.extend(['-','+'])
        rows.extend([c,c])
        cols.extend([0,1])
        if 'wc' in extension:
            wcs.extend(['+','+'])
        else:
            wcs.extend(['-','-'])
        #rows.extend([c,c])
        #cols.extend([0,1])
        if 'protein' in extension:
            methods.extend(['protein','protein'])
        else:
            methods.extend(['allele','allele'])
    cols = [0,1,2,3,0,1,2,3]
    rows = [0,0,0,0,1,1,1,1]
    
    for c,y_col in enumerate(y_cols):
        wc = wcs[c]
        promiscuity = promiscuities[c]
        row = rows[c]
        col = cols[c]
        method = methods[c]
        
        df_negative = df[df[y_col]<0]
        
        
        title = f'({method}/{wc}/{promiscuity})'
        ax[row,col].title.set_text(title)
        ax[row,col].set_ylim(-1,1)
        ax[row,col].plot(df[x_col],df[y_col],'.',label='positive')
        ax[row,col].plot(df_negative[x_col],df_negative[y_col],'.',markersize=7,label='negative')
        ax[row,col].legend(loc='upper right')
        ax[row,col].axhline(0,ls='--',c='k')
        
        if filter_cols!=None:
            for counter,col in enumerate(filter_cols):
                limits = element_to_list(filter_limits[counter])
                df_filter = df[df[col]>limits[0]]
                if filter_limits[counter]>1:
                    df_filter = df[df[col]<limits[1]]
            ax[row,col].plot(df_filter[x_col],df_filter[y_col],'r+',markersize=4,label='filter')
            
        ax[row,col].set_xlim(0,xmax)
        model = LinearRegression().fit(np.array(df[x_col]).reshape((-1, 1)), list(df[y_col]))
        r_sq = model.score(np.array(df[x_col]).reshape((-1, 1)), list(df[y_col]))
        ax[row,col].plot(np.arange(xmax),model.intercept_+model.coef_*np.arange(xmax),'k-')
        print(f'({method}/{wc}/{promiscuity})\ta={model.coef_[0]:.5f}\tb={model.intercept_:.3f}\tR2={r_sq:.5f}')

    print('blue=all\norange=pcc<0 for best forming predictor\nred=filter')
    return ax
    

def pcc_boxplot(df,extensions,figsize=(10,5)):
    """boxplot of different prediction methods"""
    fig,ax=plt.subplots(2,4,figsize=figsize)
    fig.suptitle('pcc (method/context/promiscuity)')

    y_cols = list()
    wcs = list()
    promiscuities = list()
    cols = list()
    methods = list()
    for c,extension in enumerate(extensions):
        y_cols.extend(['pcc_peptide'+extension,'pcc_mhc'+extension])
        promiscuities.extend(['-','+'])
        #cols.extend([0+c*2,1+c*2])
        if 'wc' in extension:
            wcs.extend(['+','+'])
        else:
            wcs.extend(['-','-'])
        if 'protein' in extension:
            methods.extend(['protein','protein'])
        else:
            methods.extend(['allele','allele'])
    rows = [0,0,0,0,1,1,1,1]
    cols = [0,1,2,3,0,1,2,3]
    
    for c,y_col in enumerate(y_cols):
        wc = wcs[c]
        promiscuity = promiscuities[c]
        method = methods[c]
        col = cols[c]
        row = rows[c]
        
        title = f'({method}/{wc}/{promiscuity})'
        ax[row,col].title.set_text(title)
        ax[row,col].set_ylim(-1,1)
        ax[row,col].boxplot(df[y_col])
        ax[row,col].axhline(0,ls='--',c='k')
    return ax

def plot_profile(df,idx,col,figsize=(10,10)):
    fig,ax=plt.subplots(1,1,figsize=figsize)
    ax.plot(np.arange(len(df.loc[idx,'seq'])),df.loc[idx,col],'-',label='prediction')
    ax.plot(np.arange(len(df.loc[idx,'seq'])),df.loc[idx,'peptide_counter_norm_train'],'-',label='experiment')
    legend = ax.legend(loc='upper right')
    if 'peptide' in col:
        promiscuity = '-'
    else:
        promiscuity = '+'
    if 'protein' in col:
        method = 'protein'
    else:
        method = 'allele'
    if 'wc' in col:
        wc = '+'
    else:
        wc = '-'

    title = f'profile of {df.loc[idx,"acc"]} ({method}/{wc}/{promiscuity})'
    ax.set_title(title)
    ax.set_ylim(0,1)
    print(f'protein: {df.loc[idx,"acc"]}')
    print(f'length: {df.loc[idx,"length"]}')
    print(f'# ligands: {df.loc[idx,"count"]}')
    print(f'(method/context/promiscuity)')
    print(f'pcc (allele/-/-):\t{df.loc[idx,"pcc_peptide_mhc_uc"]:.3f}')
    print(f'pcc (allele/+/-):\t{df.loc[idx,"pcc_peptide_mhc_wc"]:.3f}')
    print(f'pcc (allele/+/+):\t{df.loc[idx,"pcc_mhc_mhc_wc"]:.3f}')
    print(f'pcc (allele/-/+):\t{df.loc[idx,"pcc_mhc_mhc_uc"]:.3f}')
    print(f'pcc (protein/-/-):\t{df.loc[idx,"pcc_peptide_protein_uc"]:.3f}')
    print(f'pcc (protein/+/-):\t{df.loc[idx,"pcc_peptide_protein_wc"]:.3f}')
    print(f'pcc (protein/+/+):\t{df.loc[idx,"pcc_mhc_protein_wc"]:.3f}')
    print(f'pcc (protein/-/+):\t{df.loc[idx,"pcc_mhc_protein_uc"]:.3f}')
    return ax

def extract_acc(df,acc):
    return df[df['acc']==acc]

def get_data(data_filename,acc_filename,partition_filenames,remove_filename=None,pcc_filename=None,output='data'):
    data = np.load(data_filename, allow_pickle=True)
    accs = np.load(acc_filename, allow_pickle=True)
    if remove_filename != None:
        remove_accs = np.load(remove_filename, allow_pickle=True)
    if pcc_filename != None:
        pcc = np.load(pcc_filename, allow_pickle=True)
    idxs = list()
    for filename in partition_filenames:
        partition = np.load(filename, allow_pickle=True)
        partition_idxs = np.where(np.in1d(accs,partition))[0]
        if remove_filename != None:
            remove_idx = np.where(np.in1d(accs,remove_accs))[0]
            mask = np.ones(partition_idxs.shape[0],np.bool)
            mask[np.in1d(partition_idxs,remove_idx)] = 0
            partition_idxs = partition_idxs[mask]
        idxs = idxs+list(partition_idxs)
        idxs = list(map(lambda x: int(x),idxs))
    if output == 'data':
        return data[idxs]
    elif output == 'acc':
        return accs[idxs]

def hot_aa(seq):
    all_aa=IUPAC.IUPACProtein.letters.upper()
    pos = dict()
    for c,i in enumerate(list(all_aa+'B'+'Z'+'X'+'*')):
        pos[i] = c
    one_hot = np.zeros([1000,len(pos)],dtype='float32')
    ones = list(map(lambda x: pos[x] if x in pos else len(pos)-1,seq))
    one_hot[range(len(seq)),ones] = 1
    return one_hot

def create_data_sparse(seq,profile):
    one_hot = hot_aa(seq)
    profile = np.array([[profile[c]] if c<len(profile) else [0.0] for c in range(1000)])
    out = np.concatenate((one_hot, profile), axis=1)
    return out

def create_data_blosum(seq,profile,length):
    blosum = blosum_encode(seq,length=length)
    profile = np.array([[profile[c]] if c<len(profile) else [0.0] for c in range(length)])
    out = np.concatenate((blosum, profile), axis=1)
    return out

def prediction_profile(acc,accs,prediction,df_profile,prediction_col,train_col,figsize,ax=None):
    if ax==None:
        fig,ax=plt.subplots(figsize=figsize)
    df_acc = df_profile[df_profile['acc']==acc]
    pos = np.where(accs==acc)[0][0]
    length = list(df_acc['length'])[0]
    pcc_deep = pcc(list(df_acc[train_col])[0],prediction[pos][:length])
    pcc_nnalign = pcc(list(df_acc[train_col])[0],list(df_acc[prediction_col])[0])
    title = f'protein profile for {acc} with pcc: {pcc_deep:.2f} with deep and {pcc_nnalign:.2f} for nnalign'
    ax.title.set_text(title)
    ax.plot(prediction[pos][:length],'-',label='prediction')
    ax.plot(list(df_acc[prediction_col])[0],'-',label='nnalign')
    ax.plot(list(df_acc[train_col])[0],'-',label='experiment')
    ax.set_ylim(-0.05,1.1)
    ax.legend(loc='upper right')
    return ax

def deep_pcc(accs, prediction, df_profile, train_col):
    pccs = df_profile.apply(lambda x: pcc(list(x[train_col]), prediction[accs==x['acc']][0][:len(x[train_col])]), axis=1)
    return pccs

def progress(percent=0, width=30, start_time=time.time()):
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']',
          f' {percent:.0f}%', f'\ttime: {time.time()-start_time:.1f} s',
          sep='', end='', flush=True)

def iterate_minibatches(inputs, targets, batch_size, seed=0, shuffle=True):
    assert len(inputs) == len(targets)
    num_seq = len(inputs)
    num_batches = num_seq // batch_size
    # if num_batches*batch_size != num_seq:
    #     num_batches = num_seq // batch_size + 1
    idxs = list(np.arange(num_seq))
    if shuffle:
        random.seed(seed)
        random.shuffle(idxs)
    for i in range(num_batches):
        idx = list(idxs[i*batch_size: (i+1)*batch_size])
        yield inputs[idx], targets[idx], idx

def blosum_encode(seq,show=False,length=None):
    blosum = ep.blosum62
    #encode a peptide into blosum features
    s=list(seq)
    x = np.array([blosum[i] if i in blosum else blosum['X'] for i in seq])
    if length!=None:
        x_length = np.zeros([length,x.shape[1]],dtype='float64')
        x_length[:x.shape[0],:x.shape[1]] = x
        x = x_length
    return x

def build_model(input_var,batch_size,n_feat,n_lstm):
#     batch_size = 100
    seq_len = 1000
#     n_feat = 25
    n_lstm_f = n_lstm
    n_lstm_b = n_lstm
    l_in = lasagne.layers.InputLayer(shape=(batch_size, seq_len,n_feat),input_var=input_var)
    l_forward = lasagne.layers.LSTMLayer(l_in, n_lstm_f)
    l_vertical = lasagne.layers.ConcatLayer([l_in, l_forward], axis=2)
    l_backward = lasagne.layers.LSTMLayer(l_vertical, n_lstm_b, backwards=True)
    l_sum = lasagne.layers.ConcatLayer(incomings=[l_forward, l_backward], axis=2)
    # l_reshape = lasagne.layers.ReshapeLayer(l_sum, (batch_size*seq_len, n_lstm_f+n_lstm_b))
    l_dense = lasagne.layers.DenseLayer(l_sum, num_units=1, nonlinearity=lasagne.nonlinearities.rectify, num_leading_axes=2)
    l_out = lasagne.layers.ReshapeLayer(l_dense, (batch_size,seq_len))
    all_layers = lasagne.layers.get_all_layers(l_out)
    print("  layer output shapes:")
    for layer in all_layers:
        name = str.ljust(layer.__class__.__name__, 32)
        print("    %s %s" % (name, lasagne.layers.get_output_shape(layer)))
    return l_in, l_out

# def build_model(input_var,batch_size,n_feat,n_lstm):
# #     batch_size = 100
#     seq_len = 1000
# #     n_feat = 25
#     n_lstm_f = n_lstm
#     n_lstm_b = n_lstm
#     l_in = lasagne.layers.InputLayer(shape=(batch_size, seq_len,n_feat),input_var=input_var)
#     l_forward = lasagne.layers.LSTMLayer(l_in, n_lstm_f)
#     l_vertical = lasagne.layers.ConcatLayer([l_in, l_forward], axis=2)
#     l_backward = lasagne.layers.LSTMLayer(l_vertical, n_lstm_b, backwards=True)
#     l_sum = lasagne.layers.ConcatLayer(incomings=[l_forward, l_backward], axis=2)
#     l_reshape = lasagne.layers.ReshapeLayer(l_sum, (batch_size*seq_len, n_lstm_f+n_lstm_b))
#     l_1 = lasagne.layers.DenseLayer(l_reshape, num_units=1, nonlinearity=lasagne.nonlinearities.rectify)
#     l_out = lasagne.layers.ReshapeLayer(l_1, (batch_size,seq_len))
#     all_layers = lasagne.layers.get_all_layers(l_out)
#     print("  layer output shapes:")
#     for layer in all_layers:
#         name = str.ljust(layer.__class__.__name__, 32)
#         print("    %s %s" % (name, lasagne.layers.get_output_shape(layer)))
#     return l_in, l_out

def build_model_CNN(input_var,batch_size,n_feat,n_lstm):
#     batch_size = 100
    seq_len = 1000
#     n_feat = 25
    n_lstm_f = n_lstm
    n_lstm_b = n_lstm
    l_in = lasagne.layers.InputLayer(shape=(batch_size, seq_len, n_feat),input_var=input_var)
    l_0 = lasagne.layers.DimshuffleLayer(l_in, (0,2,1))
    l_1 = lasagne.layers.Conv1DLayer(l_0, num_filters=8, filter_size=5, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(), pad=2)
    l_2 = lasagne.layers.Conv1DLayer(l_1, num_filters=8, filter_size=5, nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(), pad=2)
    l_3 = lasagne.layers.DimshuffleLayer(l_2, (0,2,1))
    l_dense = lasagne.layers.DenseLayer(l_3, num_units=1, nonlinearity=lasagne.nonlinearities.rectify, num_leading_axes=2)
    l_out = lasagne.layers.ReshapeLayer(l_dense, (batch_size,seq_len))
    all_layers = lasagne.layers.get_all_layers(l_out)
    print("  layer output shapes:")
    for layer in all_layers:
        name = str.ljust(layer.__class__.__name__, 32)
        print("    %s %s" % (name, lasagne.layers.get_output_shape(layer)))

    return l_in, l_out

def build_model_useq(input_var):
    batch_size = 100
    seq_len = 1000
    n_feat = 1
    n_lstm_f = 50
    n_lstm_b = 50
    lr = 0.25
    start_saving_at = 0
    save_every = 1
    filenumber = 0
    l_in = lasagne.layers.InputLayer(shape=(batch_size, seq_len,n_feat),input_var=input_var)
    l_forward = lasagne.layers.LSTMLayer(l_in, n_lstm_f)
    l_vertical = lasagne.layers.ConcatLayer([l_in, l_forward], axis=2)
    l_backward = lasagne.layers.LSTMLayer(l_vertical, n_lstm_b, backwards=True)
    l_sum = lasagne.layers.ConcatLayer(incomings=[l_forward, l_backward], axis=2)
    l_reshape = lasagne.layers.ReshapeLayer(l_sum, (batch_size*seq_len, n_lstm_f+n_lstm_b))
    l_1 = lasagne.layers.DenseLayer(l_reshape, num_units=1, nonlinearity=lasagne.nonlinearities.rectify)
    l_out = lasagne.layers.ReshapeLayer(l_1, (batch_size,seq_len))
    all_layers = lasagne.layers.get_all_layers(l_out)
    print("  layer output shapes:")
    for layer in all_layers:
        name = str.ljust(layer.__class__.__name__, 32)
        print("    %s %s" % (name, lasagne.layers.get_output_shape(layer)))
    return l_in, l_out

def confusion_matrix(train_profile, predicted_profile, threshold=None): 
    train_profile = np.array(train_profile)
    predicted_profile = np.array(predicted_profile)
    
    length = len(train_profile)
    
    train_negative = train_profile==0
    train_positive = train_profile!=0
    predicted_positive = predicted_profile==0
    predicted_negative = predicted_profile!=0
     
    if threshold==None:
        true_positive = sum(train_positive&predicted_positive)/length*100
        true_negative = sum(train_negative&predicted_negative)/length*100
        false_positive = sum(train_negative&predicted_positive)/length*100
        false_negative = sum(train_positive&predicted_negative)/length*100
    else:
        true_negative = sum(train_negative&(predicted_profile<threshold))/length*100
        false_negative = sum(train_negative&(predicted_profile>threshold))/length*100
        true_positive = sum(np.abs(train_profile[train_positive]-predicted_profile[train_positive])<threshold)/length*100
        false_positive = sum(np.abs(train_profile[train_positive]-predicted_profile[train_positive])>threshold)/length*100
        
    total = true_positive+true_negative+false_positive+false_negative
    
    if total < 99.9:
        print('does not match 100 %')
        print(total)
        return None
    elif total > 101:
        print('exceed 100 %')
        print(total)
        return None
    
    return [true_positive, true_negative, false_positive, false_negative]

def extend_input(profile, length):
    profile = np.array([[profile[c]] if c<len(profile) else [0.0] for c in range(length)])
    return profile
