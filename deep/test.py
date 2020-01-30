#!/usr/bin/env python
import os
import sys

folder = '/home/projects/vaccine/people/s143849/'
data_folder = 'alternative_data/'
birkir_folder = 'birkir/'
jupyter_folder = 'jupyter/'
log_err_folder = 'log_err/'
deep_folder = 's_deep/'
extract_top_folder = 's_extract_top/'
nnalign_folder = 's_nnalign/'
partition_folder = 's_partition/'
similarity_folder = 's_similarity/'
filter_folder = 's_filter/'
profile_folder = 's_profile/'
loss_folder = 'loss/'
prediction_folder = 'prediction/'
single_allele_folder = 'single_allele/'
cell_line_folder = 'cell_line/'
random_folder = 'random/'
mean_folder = 'mean/'
all_folder = 'all/'
param_folder = 'params/'
top100_folder = 'top100/'

ligand_filename = 'MasterMerger_context_pos_19_03_08.txt'
sequence_filename = 'MasterSourceProt_blastEntrez.fasta'
new_ligand_filename = 'log_eval_concat.txt'

df_seq_filename = 'df_seq.json'
df_seq_filter_filename = 'df_seq_filter.json'
df_seq_profile_filename = 'df_seq_profile.json'
filtered_fasta = 'filtered_fasta'
peptides_filename = '13_21_acc.txt'
partition_filename = 'partition.ft'
partition_9mer_filename = 'partition_9mer.ft'
top_mhc_uc_filename = 'top_mhc_uc.ft'
top_mhc_wc_filename = 'top_mhc_wc.ft'
top_protein_uc_filename = 'top_protein_uc.ft'
top_protein_wc_filename = 'top_protein_wc.ft'
input_filename = 'predicted_profiles.npy'
output_filename = 'experimental_profiles.npy'
acc_filename = 'acc.npy'
length_filename = 'length.npy'

output_filename = 'experimental_profiles.npy'
acc_filename = 'acc.npy'
n_lstm = int(sys.argv[1])
lr = float(sys.argv[2])
input_filename = 'predicted_profiles.npy'
meta_filename = f'{n_lstm}_{lr}_'+'dump_'
best_params_filename = f'{n_lstm}_{lr}_'+'best_params.npy'
prediction_filename = f'{n_lstm}_{lr}_'+'prediction.npy'
best_epoch_filename = f'{n_lstm}_{lr}_'+'best_epoch.txt'
train_loss_filename = f'{n_lstm}_{lr}_'+'train_loss.npy'
valid_loss_filename = f'{n_lstm}_{lr}_'+'valid_loss.npy'

train_amount_folder = top100_folder
batch_size = 20
n_feat = 25
num_epochs = 5
encoding = 'blosum_'

train_counter = [0,1,2,3,4]
train_filenames = list()
for i in train_counter:
    train_filenames.append(folder+data_folder+jupyter_folder+similarity_folder+f'{i}.npy')

data_filename = folder+data_folder+jupyter_folder+deep_folder+train_amount_folder+encoding+input_filename
filename_acc = folder+data_folder+jupyter_folder+deep_folder+train_amount_folder+acc_filename

valid_filenames = list()
for counter in train_counter:
	valid_filename = [folder+data_folder+jupyter_folder+similarity_folder+f'{counter}.npy']
filename = folder+data_folder+jupyter_folder+deep_folder+train_amount_folder+output_filename

predictions_path = folder+data_folder+deep_folder+prediction_folder+f'{counter}_'+prediction_filename

for i in train_filenames:
	print(os.path.exists(i))
print(os.path.exists(data_filename))
print(os.path.exists(filename_acc))
for i in valid_filenames:
	print(os.path.exists(i))
print(os.path.exists(filename))
print(os.path.exists(folder+data_folder+deep_folder+prediction_folder))
