#!/usr/bin/python3
import os
import sys
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,optimizer=None,device=cpu,floatX=float32"
#sys.path.insert(0,'..')
import numpy as np
import theano
import theano.tensor as T
import lasagne
import matplotlib.pyplot as plt
import time
import itertools
start_time = time.time()
from my_func_computerome import *
print(round(time.time()-start_time,3),'s to import my_func')
import pickle
from datetime import datetime, timedelta

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
peptides_filename = '13_21_small_acc.txt'
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
meta_filename = f'{lr}_'+'dump_'
best_params_filename = f'{lr}_'+'best_params.npy'
prediction_filename = f'{lr}_'+'prediction.npy'
best_epoch_filename = f'{lr}_'+'best_epoch.txt'
train_loss_filename = f'{lr}_'+'train_loss.npy'
valid_loss_filename = f'{lr}_'+'valid_loss.npy'

batch_size = 10
n_feat = 25
num_epochs = 40

def main():
    # We use ftensor3 because the protein data is a 3D-matrix in float32 
    input_var = T.ftensor3('inputs')
    # ivector because the labels is a single dimensional vector of integers
    target_var = T.fvector('targets')
    # fmatrix because the masks to ignore the padded positions is a 2D-matrix in float32
    mask_var = T.fmatrix('masks')
    
    l_in, l_out = build_model(input_var=input_var,batch_size=batch_size,n_feat=n_feat,n_lstm=n_lstm)

    filename = folder+data_folder+deep_folder+input_folder+input_filename
    X_train, X_valid = get_data(filename)
    X = np.concatenate((X_train,X_valid),axis=0)
    print('X shape',X.shape)

    sym_x = T.ftensor3()
    inference = lasagne.layers.get_output(l_out, sym_x, deterministic=True)
    best_path = folder+data_folder+deep_folder+output_folder+param_folder+best_params_filename
    params = np.load(best_path,allow_pickle=True)
    lasagne.layers.set_all_param_values(l_out, params)
    predict = theano.function([sym_x], inference, allow_input_downcast=True)
    predictions = []
    num_batches = np.size(X,axis=0) // batch_size + 1
    for i in range(num_batches):
        x_batch = X[i*batch_size:(i+1)*batch_size]
        size = x_batch.shape[0]
        print('batch shape', x_batch.shape)
        if x_batch.shape[0] != batch_size:
            x_batch = X[-batch_size:]
        p = predict(x_batch)
        zero_test = p == np.zeros([*p.shape])
        if zero_test.all():
            print('all vere predicted 0')
            return None
        if p[]
        if size != 0:
            predictions.append(p[-size:])
    predictions = np.concatenate(predictions, axis = 0)
    predictions_path = folder+data_folder+deep_folder+output_folder+prediction_folder+prediction_filename
    print("Storing predictions in %s" % data_folder+deep_folder+output_folder+prediction_folder+prediction_filename)
    np.save(predictions_path, predictions)

if __name__ == '__main__':
    main()