import os
import sys
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.utils import compute_norms
from lasagne.regularization import regularize_network_params, l1, l2
import matplotlib.pyplot as plt
import time
import itertools
path = '/home/projects/vaccine/people/s143849/alternative_script/'
# path = '/Users/asbjornellerskaarup/Google_Drive/Skole/Master/script'
sys.path.append(path)
print('importing my_func...')
start_time = time.time()
from my_func_computerome import *
# from my_func import *
print(round(time.time()-start_time,3),'s to import my_func')
import pickle
from datetime import datetime, timedelta
import pandas as pd
from IPython.display import display, HTML
from Bio import SeqIO
import copy
import collections
import json
import os.path as path
import math
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
import subprocess
from subprocess import Popen,PIPE

extension = 'ur_random_'

n_lstm = int(sys.argv[1])
lr = float(sys.argv[2])
amount = sys.argv[3]
batch_size = int(sys.argv[4])
# n_lstm = 200
# lr = 0.0001
# amount = '100'
# batch_size = 10

encoding = 'blosum_'

folder = '/home/projects/vaccine/people/s143849/'
# folder = '/Users/asbjornellerskaarup/Google_Drive/Skole/Master/'
data_folder = folder+'alternative_data/'
birkir_folder = data_folder+'birkir/'
jupyter_folder = data_folder+'jupyter/'
log_err_folder = data_folder+'log_err/'
deep_folder = data_folder+'s_deep/'
extract_top_folder = data_folder+'s_extract_top/'
nnalign_folder = data_folder+'s_nnalign/'
partition_folder = data_folder+'s_partition/'
similarity_folder = data_folder+'s_similarity/'
filter_folder = jupyter_folder+'s_filter/'
profile_folder = jupyter_folder+'s_profile/'
jp_deep_folder = jupyter_folder+'s_deep/'
jp_partition_folder = jupyter_folder+'s_partition/'
jp_similarity_folder = jupyter_folder+'s_similarity/'
single_allele_folder = 'single_allele/'
cell_line_folder = 'cell_line/'
random_folder = 'random/'
mean_folder = 'mean/'
all_folder = 'all/'
top100_folder = '100/'
random150_folder = '150/'
if 'ur_' in extension:
    allele_folder = single_allele_folder
elif 'cl_' in extension:
    allele_folder = cell_line_folder
if 'random' in extension:
    method_folder = 'random/'
elif 'mean' in extension:
    method_folder = 'mean/'
if amount == '100':
    train_amount_folder = jp_deep_folder+top100_folder
elif amount == 'all':
    train_amount_folder = jp_deep_folder+all_folder
elif amount == '150':
    train_amount_folder = jp_deep_folder+random150_folder
deep_extension = f'{n_lstm}_{lr}_{amount}_{batch_size}_'
deep_output_folder = deep_folder+f'{deep_extension[:-1]}/'
if not os.path.exists(deep_output_folder):
    print('making folder',deep_output_folder)
    os.makedirs(deep_output_folder)

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
acc_filename = 'acc.npy'
bad_acc_filename = 'bad_acc.npy'
length_filename = 'length.npy'
output_filename = 'experimental_profiles.npy'
pcc_filename = 'pcc.npy'

input_filename = encoding+'predicted_profiles.npy'
meta_filename = deep_extension+'dump_'
best_params_filename = deep_extension+'best_params.npy'
prediction_train_filename = deep_extension+'train_prediction.npy'
prediction_valid_filename = deep_extension+'valid_prediction.npy'
best_epoch_filename = deep_extension+'best_epoch.txt'
train_loss_filename = deep_extension+'train_loss.npy'
valid_loss_filename = deep_extension+'valid_loss.npy'


n_feat = 25
num_epochs = 50

def main():
    np.random.seed(0)
    lasagne.random.set_rng(np.random)
    for counter in range(4,5):
        print('valid partition',counter)
        train_counter = [0,1,2,3,4]
        train_counter.remove(counter)

        train_filenames = list()
        for i in train_counter:
            train_filenames.append(jp_similarity_folder+f'{i}.npy')

        print('loading data...')
        filename_bad = train_amount_folder+bad_acc_filename
        data_filename = train_amount_folder+input_filename
        filename_acc = train_amount_folder+acc_filename
        filename_pcc = train_amount_folder+pcc_filename
        # X_train = get_data(data_filename,filename_acc,train_filenames,filename_bad,filename_pcc)
        X_train = get_data(data_filename,filename_acc,train_filenames)
        print('Train shape',X_train.shape)
        
        valid_filename = [jp_similarity_folder+f'{counter}.npy']
        X_valid = get_data(data_filename,filename_acc,valid_filename)
        print('Valid shape',X_valid.shape)

        filename = train_amount_folder+output_filename
        filename_pcc = train_amount_folder+pcc_filename
        # y_train = get_data(filename,filename_acc,train_filenames,filename_bad,filename_pcc)
        y_train = get_data(filename,filename_acc,train_filenames)
        print('Train target shape',y_train.shape)
        y_valid = get_data(filename,filename_acc,valid_filename)
        print('Valid target shape',y_valid.shape)

        filename_length = train_amount_folder+length_filename
        # lengths_train = get_data(filename_length,filename_acc,train_filenames,filename_bad,filename_pcc)
        lengths_train = get_data(filename_length,filename_acc,train_filenames)
        lengths_valid = get_data(filename_length,filename_acc,valid_filename)

        pccs_target_train = np.zeros(X_train.shape[0])
        for c,i in enumerate(X_train):
            length = lengths_train[c]
            pccs_target_train[c] = round(pcc(y_train[c][:int(length)],i[:int(length),24]),3)
        m_pcc_target_train = round(np.mean(pccs_target_train[(~np.isnan(pccs_target_train))&(pccs_target_train!=0)]),3)
        print('pcc',pccs_target_train,m_pcc_target_train)

        pccs_target_valid = np.zeros(X_valid.shape[0])
        for c,i in enumerate(X_valid):
            length = lengths_valid[c]
            pccs_target_valid[c] = round(pcc(y_valid[c][:int(length)],i[:int(length),24]),3)
        m_pcc_target_valid = round(np.mean(pccs_target_valid[(~np.isnan(pccs_target_valid))&(pccs_target_valid!=0)]),3)
        print('pcc',pccs_target_valid,m_pcc_target_valid)

        # We use ftensor3 because the protein data is a 3D-matrix in float32 
        input_var = T.ftensor3('inputs')
        # ivector because the labels is a single dimensional vector of integers
        target_var = T.fvector('targets')
        # fmatrix because the masks to ignore the padded positions is a 2D-matrix in float32
        mask_var = T.fmatrix('masks')
        
        l_in, l_out = build_model_CNN(input_var=input_var,batch_size=batch_size,n_feat=n_feat,n_lstm=n_lstm)
        
        # Get output training, deterministic=False is used for training
        prediction = lasagne.layers.get_output(l_out, inputs={l_in: input_var}, deterministic=False)

        # Calculate the categorical cross entropy between the labels and the prediction
        t_loss = lasagne.objectives.squared_error(prediction.flatten(), target_var)

        # rebroadcast = theano.tensor.TensorType('float32', broadcastable=())
        loss = T.mean(t_loss)
        # print(loss.type)
        # loss.type = rebroadcast
        # print(loss.type)
        # Training loss
        # weight_decay = 1e-2
        # l1_penalty = regularize_network_params(l_out, l2)
        # loss += l1_penalty * weight_decay

        # Parameters
        params = lasagne.layers.get_all_params([l_out], trainable=True)
        params_show = copy.deepcopy(params)
        params_show.insert(15,'')
        params_show.insert(15,'')
        params_show.insert(32,'')
        params_show.insert(32,'')
        param = lasagne.layers.get_all_param_values(l_out)
        # for c,i in enumerate(param):
        #     norms = compute_norms(i)
        #     print('norm',i.shape,norms.shape,'\t',np.around(np.mean(norms),6),params_show[c],sep='\t')
        # Update parameters using ADAM 
        updates = lasagne.updates.adam(loss, params, learning_rate=lr)
        
        # Get output validation, deterministic=True is only use for validation
        val_prediction = lasagne.layers.get_output(l_out, inputs={l_in: input_var}, deterministic=True)

        # Calculate the categorical cross entropy between the labels and the prediction
        t_val_loss = lasagne.objectives.squared_error(val_prediction.flatten(), target_var)

        # Validation loss 
        val_loss = T.mean(t_val_loss)
        
        print('making training function...')
        start_time = time.time()
        train_fn = theano.function([input_var, target_var], [loss, prediction], updates=updates, allow_input_downcast=True)
        print(round(time.time()-start_time,3),'s to make training function')
        
        print('making validation function...')
        val_fn = theano.function([input_var, target_var], [val_loss, val_prediction], allow_input_downcast=True)

        # Lists to save loss and accuracy of each epoch
        loss_training = np.zeros(num_epochs,dtype='float64')
        loss_validation = np.zeros(num_epochs,dtype='float64')
        start_time_est = time.time()
        prev_time = time.time()
        min_val_loss = float("inf")

        # Start training 
        for epoch in range(num_epochs):
            print("Epoch %s of %s" % (epoch + 1, num_epochs))

            # Full pass training set
            train_err = 0
            train_batches = 0

            num_seq_train = X_train.shape[0]//batch_size
            num_seq_valid = X_valid.shape[0]//batch_size

            # Generate minibatches and train on each one of them
            print('training')
            start_time = time.time()
            n_batches = num_seq_train // batch_size
            if n_batches*batch_size != num_seq_train:
                n_batches = num_seq_train // batch_size + 1
            pccs = np.zeros(X_train.shape[0])
            pccs_long = np.zeros(X_train.shape[0])
            confusion = list()
            for c,batch in enumerate(iterate_minibatches(X_train, y_train, batch_size, seed=counter)):
                # Inputs to the network
                print(c+1, 'out of', n_batches,round(time.time()-start_time,3),'s')
                start_time = time.time()
                inputs, targets, idx = batch
                # Calculate loss and prediction
                tr_err, predict = train_fn(inputs, targets.flatten())
                for i,p in zip(idx,predict):
                    length = lengths_train[i]
                    pccs[i] = round(pcc(y_train[i][:int(length)],p[:int(length)]),3)
                    pccs_long[i] = round(pcc(y_train[i],p),3)
                    confusion.append(confusion_matrix(y_train[i][:int(length)],p[:int(length)]))
                    # if np.isnan(pccs[i]):
                    #     print(pccs[i])

                zero_test = predict == np.zeros([*predict.shape])
                if zero_test.all():
                    print('all were predicted 0!!!!!!!!!!!!!!!!!!!!!!')
                train_err += tr_err
                train_batches += 1

            # print('pcc',np.array(list(zip(pccs,list(map(lambda x: ' '.join([str(round(i,0)) for i in x]),confusion))))))
            # print('pcc difference', pccs-pccs_target_train)
            print('pcc training')
            print('mean pcc', '|', 'mean target pcc', '|', 'mean pcc for 1000 long sequence')
            print(round(np.mean(pccs[pccs!=0]),3),round(np.mean(pccs_target_train[pccs!=0]),3),round(np.mean(pccs_long),3))

            # param = lasagne.layers.get_all_param_values(l_out)
            # for c,i in enumerate(param):
            #     norms = compute_norms(i)
            #     print('norm',i.shape,norms.shape,'\t',np.around(np.mean(norms),6),params_show[c])

            # Average loss and accuracy
            train_loss = train_err / train_batches

            loss_training[epoch] = train_loss

            val_err = 0
            val_batches = 0
            pccs = np.zeros(X_valid.shape[0])
            pccs_long = np.zeros(X_valid.shape[0])
            # Generate minibatches and validate on each one of them, same procedure as before
            print('\ntesting')
            for c,batch in enumerate(iterate_minibatches(X_valid, y_valid, batch_size, seed=counter)):
                
                inputs, targets, idx = batch
                err, predict_val = val_fn(inputs, targets.flatten())
                for i,p in zip(idx,predict_val):
                    length = lengths_valid[i]
                    pccs[i] = round(pcc(y_valid[i][:int(length)],p[:int(length)]),3)
                    pccs_long[i] = round(pcc(y_valid[i],p),3)
                    # if np.isnan(pccs[index]):
                    #     print(pccs[index])
                    #     print(np.where(y_valid[index]!=0),length)
                    #     print(np.array(list(zip(y_valid[index][:int(length)],i[:int(length)]))))
                val_err += err
                val_batches += 1

            val_loss = val_err / val_batches

            loss_validation[epoch] = val_loss
            print('pcc validation')
            print(np.array(list(zip(pccs[pccs!=0],pccs[pccs!=0]-pccs_target_valid[pccs!=0]))))
            # print('pcc',pccs[pccs!=0])
            # print('pcc difference', pccs[pccs!=0]-pccs_target_valid[pccs!=0])
            print('mean pcc', '|', 'mean target pcc', '|', 'mean pcc for 1000 long sequence')
            print(round(np.mean(pccs[pccs!=0]),3),round(np.mean(pccs_target_valid[pccs!=0]),3),round(np.mean(pccs_long),3))

            # Save the model parameters at the epoch with the lowest validation loss
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                best_path = deep_output_folder+f'{counter}_'+best_params_filename
                np.save(best_path,lasagne.layers.get_all_param_values(l_out))
                best_epoch_path = deep_output_folder+f'{counter}_'+best_epoch_filename
                with open(best_epoch_path, 'w') as epoch_file:
                    epoch_file.write(str(epoch))
        #         print(*lasagne.layers.get_all_param_values(l_out))

            now = time.time()
            time_since_start = now - start_time_est
            time_since_prev = now - prev_time
            prev_time = now
            est_time_left = time_since_prev * (num_epochs-epoch-1) + (4-counter)*(num_epochs)*time_since_prev
            eta = datetime.now() + timedelta(seconds=est_time_left)
            eta_str = eta.strftime("%c")
            print("  %.2f s since start (%.2f s for this epoch)" % (time_since_start, time_since_prev))
            print("  Estimated %.2f s to go (ETA: %s)" % (est_time_left, eta_str))
            print("  training loss:\t\t{:.6f}".format(train_loss))
            print("  validation loss:\t\t{:.6f}".format(val_loss))

        print(loss_validation.shape)
        print(loss_training.shape)

        filename = deep_output_folder+f'{counter}_'+train_loss_filename
        np.save(filename,loss_training)
        filename = deep_output_folder+f'{counter}_'+valid_loss_filename
        np.save(filename,loss_validation)

        
        # We use ftensor3 because the protein data is a 3D-matrix in float32 
        input_var = T.ftensor3('inputs')
        # ivector because the labels is a single dimensional vector of integers
        target_var = T.fvector('targets')
        # fmatrix because the masks to ignore the padded positions is a 2D-matrix in float32
        mask_var = T.fmatrix('masks')
        
        l_in, l_out = build_model_CNN(input_var=input_var,batch_size=batch_size,n_feat=n_feat,n_lstm=n_lstm)

        sym_x = T.ftensor3()
        inference = lasagne.layers.get_output(l_out, sym_x, deterministic=True)
        params = np.load(best_path,allow_pickle=True)
        lasagne.layers.set_all_param_values(l_out, params)
        predict = theano.function([sym_x], inference, allow_input_downcast=True)
        predictions = []
        n_batches = math.ceil(X_valid.shape[0]/batch_size)
        for i in range(n_batches):
            #idx = range(i*batch_size, (i+1)*batch_size)
            x_batch = X_valid[i*batch_size:(i+1)*batch_size]
            size = x_batch.shape[0]
            print('batch shape', x_batch.shape)
            if x_batch.shape[0] != batch_size:
                x_batch = X_valid[-batch_size:]
            p = predict(x_batch)
            zero_test = p == np.zeros([*p.shape])
            if zero_test.all():
                print('all were predicted 0')
                return None
            if size != 0:
                predictions.append(p[-size:])
        predictions = np.concatenate(predictions, axis = 0)
        print('valid prediction shape',predictions.shape)
        predictions_path = deep_output_folder+f'{counter}_'+prediction_valid_filename
        print("Storing predictions in %s" % predictions_path)
        np.save(predictions_path, predictions)

        predictions = []
        n_batches = math.ceil(X_train.shape[0]/batch_size)
        for i in range(n_batches):
            #idx = range(i*batch_size, (i+1)*batch_size)
            x_batch = X_train[i*batch_size:(i+1)*batch_size]
            size = x_batch.shape[0]
            print('batch shape', x_batch.shape)
            if x_batch.shape[0] != batch_size:
                x_batch = X_train[-batch_size:]
            p = predict(x_batch)
            zero_test = p == np.zeros([*p.shape])
            if zero_test.all():
                print('all were predicted 0!!!!!!!!!!!!!!!!!!!!!!')
                continue
            if size != 0:
                predictions.append(p[-size:])
        predictions = np.concatenate(predictions, axis = 0)
        print('training prediction shape',predictions.shape)
        predictions_path = deep_output_folder+f'{counter}_'+prediction_train_filename
        print("Storing predictions in %s" % predictions_path)
        np.save(predictions_path, predictions)

if __name__ == '__main__':
    main()
