#!/usr/bin/python3
print('start importing')
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
print('done importing')

folder = "/home/projects/vaccine/people/s143849/"
data_folder = "data/"
state_folder = "intermidiate_state/"
protein_folder = "protein_peptides/"
partition_folder = "partitions/"
nnalign_folder = "nnalign/"
fasta_folder = 'one_entry/'

def extract_seq(seq_filename):
    seq = dict()
    fasta_sequences = SeqIO.parse(open(seq_filename),'fasta')
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        seq[name] = sequence 
    df = pd.DataFrame(list(seq.items()), columns=['acc', 'seq'])
#     df.set_index('acc', inplace=True)
    return df

def extract_ligands(lig_filename):
    df = pd.read_csv(lig_filename, sep='\t')
    df['length'] = pd.Series(df['Ligand'].str.len())
    return df

def combine_seq_and_lig(df_seq,df_lig):
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
    df = df[df['count']>0]
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

def write_fasta(df,filename,seperate=False):
    if '.' in filename:
        idx = filename.index('.')
        print(f'Do not include file format: {filename[idx:]}')
        return None
    if not seperate:
        with open(f'{filename}.fasta', 'w') as outfile:
            outfile.write('\n'.join(flatten(list(zip('>'+df['acc'],df['seq'])))))
    else:
        for counter,row in enumerate(zip('>'+df['acc'],df['seq'])):
            acc = row[0]
            seq = row[1]
            with open(f'{filename}_{counter}.fasta', 'w') as outfile:
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
                    context = Xs+seq[0:aa_pos]+seq[aa_pos+length:aa_pos+length+3]
                elif aa_pos+length+3>len(seq):
                    Xs = 'X'*(3-(len(seq)-(aa_pos+length)))
                    context = seq[aa_pos-3:aa_pos]+seq[aa_pos+length:len(seq)]+Xs
                else:
                    context = seq[aa_pos-3:aa_pos]+seq[aa_pos+length:aa_pos+length+3]
                if first:
                    peptides[acc] = [[acc,aa_pos,len(seq)-aa_pos-length,seq[aa_pos:aa_pos+length],1.0,allele,context]]
                    first = False
                else:
                    peptides[acc].append([acc,aa_pos,len(seq)-aa_pos-length,seq[aa_pos:aa_pos+length],1.0,allele,context])
    return peptides[acc]

def list_to_string(l):
    l = [str(i) for i in l]
    return l

def write_mers(df,limits=(13,21),filename=None):
    if filename == None:
        if limits[1]-limits[0]>1:
            filename = f'{folder}{data_folder}{state_folder}{limits[0]}_{limits[1]}'
        else:
            filename = f'{folder}{data_folder}{state_folder}{limits[0]}'
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

def in_train(df_peptides,train_filename,counter,exist):
    with open(f'{train_filename}{counter}.txt', 'r') as infile:
        df_train = pd.read_csv(infile, sep='\t', header=None)
    df_train.columns = ('Peptide', 'affinity', 'MHC_mol')
    train_col = f'train_{counter}'
    intersect = set(df_peptides['Peptide']).intersection(set(df_train['Peptide']))
    if not exist:
        df_peptides[train_col] = 0
    df_peptides.set_index('Peptide', inplace=True)
    df_peptides.loc[intersect,train_col] = 1
    df_peptides.reset_index(inplace=True)
    df_peptides[train_col] = df_peptides[train_col].astype('int8')
    return df_peptides

def add_partition(df,train_filenames):
    exist = False
    for filename in train_filenames:
        for i in range(1,6):
            print(i)
            df = in_train(df,filename,i,exist)
        exist = True
    df['from_start'] = df['from_start'].astype('int16')
    df['to_end'] = df['to_end'].astype('int16')
    df['affinity'] = df['affinity'].astype('int8')
    return df

def find_9mer(df):
    start_time = time.time()
    for i in range(1,6):
        in_train = np.zeros(df.shape[0])
        acc_uniq = np.unique(list(df['acc'].values))
        for c,acc in enumerate(acc_uniq):
            acc_filter = df['acc']==acc
            positive_filter = df[f'train_{i}']==1
            df_acc = df[acc_filter]
            train_uniq = df[acc_filter&positive_filter]['Peptide'].unique()
            for peptide in train_uniq:
                for pos in range(0,len(peptide)-9+1):
                    in_train_idx = df_acc[df_acc['Peptide'].str.contains(peptide[pos:pos+9],regex=False)].index
                    in_train[in_train_idx] = 1
            print(f'{c/len(acc_uniq)*100:.3f}%\t{i}\t{acc}\t{len(train_uniq)}\t{time.time()-start_time:.3f}')
            start_time = time.time()
        df[f'train_{i}_9mer']=in_train.astype('int8')
    return df

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
        print(f'{counter}\t{counter/len(filenames)*100:.3f}')
        if 'wc' in filename:
            extra_row = 1
        else:
            extra_row = 0
        df = pd.read_csv(filename, sep=' ', header=None, skiprows=272+extra_row)
        diff = df.shape[1]-len(header)
        cols = header + list(np.arange(diff))
        df.columns = cols
        if first:
            df_nnalign = df
            df_nnalign.rename(columns={'Prediction':f'Prediction_{counter}', 2:'Context'},inplace=True)
            first = False
        else:
            df_nnalign[f'Prediction_{counter}'] = df['Prediction']
    
    translate_pred = dict()
    for c,translate in enumerate(translate_predictions):
        translate_pred[f'Prediction_{c}'] = translate
        
    df_nnalign.rename(
        columns=translate_pred, inplace=True)
    cols_float = translate_predictions
    #df_nnalign[cols_float] = df_nnalign[cols_float].astype('float16')
    cols_all = df_nnalign.columns
    cols_used = ['Peptide','MHC_mol','Context']+cols_float
    df_nnalign = df_nnalign[cols_used]
    return df_nnalign

def merge_nn_partition(df_partition, df_nnalign):
    if df_partition.shape[0] != df_nnalign.shape[0]:
        print('not same size')
        return None
    df_nnalign = pd.concat([df_nnalign,df_partition['acc']],axis=1)
    df = pd.merge(df_nnalign, df_partition, on=['Peptide', 'MHC_mol','acc','Context'], how='inner')
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
    #df_protein[sort_col] = df_protein[sort_col].astype('float16')
    #df_protein['partition'] = df_protein['partition'].astype('int8')
    #df_mhc[sort_col] = df_mhc[sort_col].astype('float16')
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
    #df_protein[sort_col] = df_protein[sort_col].astype('float16')
    #df_protein['partition'] = df_protein['partition'].astype('int8')
    #df_mhc[sort_col] = df_mhc[sort_col].astype('float16')
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
            #with open(folder+data_folder+state_folder+'extract_top.progress', 'a+') as progress_file:
            #   progress_file.write(f'{train_counter}\t{time.time()-start_time:.3f}\n')
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

def combine_splits(filename_mhc_wc,filename_mhc_uc,filename_protein_wc,filename_protein_uc):
    first = True
    start_folder = folder+data_folder+state_folder
    for train_counter in range(1,6):
        filename_mhc_wc = f'{filename_mhc_wc[:-3]}_{train_counter}.ft'
        filename_mhc_uc = f'{filename_mhc_uc[:-3]}_{train_counter}.ft'
        filename_protein_wc = f'{filename_protein_wc[:-3]}_{train_counter}.ft'
        filename_protein_uc = f'{filename_protein_uc[:-3]}_{train_counter}.ft'
        df_mhc_uc_split = load_df_feather(filename_mhc_uc)
        df_mhc_wc_split = load_df_feather(filename_mhc_wc)
        df_protein_uc_split = load_df_feather(filename_protein_uc)
        df_protein_wc_split = load_df_feather(filename_protein_wc)
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

def write_csv(df,filename):
    df.to_csv(filename, sep='\t', index=False, header=False)

def add_peptides(df_seq,df_peptide,extension,speed=True,score=False):
    """add peptides, their associated mhc,score and partition as columns to each protein"""
    df_seq['peptide'+extension] = df_seq['acc'].apply(lambda x: list(df_peptide[df_peptide['acc']==x]['Peptide']))
    df_seq['peptide_mhc'+extension] = df_seq['acc'].apply(lambda x: list(df_peptide[df_peptide['acc']==x]['MHC_mol']))
    if score:
        df_seq['peptide_score_mean'+extension] = df_seq['acc'].apply(lambda x: df_peptide[df_peptide['acc']==x]['Prediction'].mean())
        df_seq['peptide_score_mean'+extension] = df_seq['peptide_score_mean'+extension].astype('float16')
    if not speed:
        df_seq['peptide_sort'+extension] = df_seq['peptide'+extension].apply(lambda x: sorted(x))
        df_seq['peptide_score'+extension] = df_seq['acc'].apply(lambda x: np.array(df_peptide[df_peptide['acc']==x]['Prediction'],dtype='float16'))
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
        df['peptide_counter'+extension] = df.seq.apply(lambda x: np.array(np.zeros(len(x),dtype='int8')))
        df['mhc_counter'+extension] = df.seq.apply(lambda x: np.array(np.zeros(len(x),dtype='int8')))
        df['mhcs'+extension] = df.seq.apply(lambda x: init_list_of_objects(len(x)))
    return df

def match_peptides(seq,peptides):
    """match peptides to protein sequence"""
    positions = np.array(np.zeros(len(seq),dtype='int8'))
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
    positions = np.array(np.zeros(len(position_mhcs),dtype='int8'))
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
        df['peptide_counter'+extension] = df.apply(lambda x: np.array(match_peptides(x['seq'],x[peptide_col]),dtype='int8'), axis=1)
        df['mhcs'+extension] = df.apply(lambda x: match_mhc(x['seq'], x[peptide_col], x[mhc_col]), axis=1)
        df['mhc_counter'+extension] = df.apply(lambda x: np.array(match_mhc_counter(x['mhcs'+extension]),dtype='int8'), axis=1)
        #cols.extend(['peptide_counter'+extension,'mhcs'+extension,'mhc_counter'+extension])
    #df[cols] = df[cols].astype('int8')
    return df

def normalize_count(df,extensions):
    """normalize counters at each position in the protein"""
    for extension in extensions:
        df['mhc_counter_norm'+extension] = df['mhc_counter'+extension].apply(lambda x: (np.array(x,dtype='float16')-min(x))/(max(x)-min(x)))
        df['peptide_counter_norm'+extension] = df['peptide_counter'+extension].apply(lambda x: (np.array(x,dtype='float16')-min(x))/(max(x)-min(x)))
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
    df[cols] = df[cols].astype('float16')
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
    
    print('wc/promiscuity\ta*x+b')
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
        print(f'({wc}/{promiscuity})\ta={model.intercept_:.3f}\tb={model.coef_[0]:.5f}\tR2={r_sq:.5f}')

    print('blue=all\norange=pcc<0 for best forming predictor\nred=filter')
    return ax
    

def pcc_boxplot(df,extensions,figsize=(10,5)):
    """boxplot of different prediction methods"""
    fig,ax=plt.subplots(2,4,figsize=figsize)
    fig.suptitle('pcc (method/wc/promiscuity)')

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
            wcs.extend(['-','-'])
        else:
            wcs.extend(['+','+'])
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
    title = f'profile of {df.loc[idx,"acc"]}'
    ax.set_title(title)
    ax.set_ylim(0,1)
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

def get_train(filename):
    X = np.load(filename+'.npy')
    X_train = X[:900]
    X_valid = X[900:]
    y_train = X[:900,21]
    y_valid = X[900:,21]
    mask_train = np.zeros([900,4],dtype='int8')
    mask_train[:,2] = 1
    mask_test = np.zeros([100,4],dtype='int8')
    mask_test[:,2] = 1
    num_seq_train = np.size(X_train,0)
    return X_train, X_valid, y_train, y_valid, mask_train, mask_test, num_seq_train






