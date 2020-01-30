#!/bin/sh
for LR in "0.001" "0.0001" "0.00001"
do
	for NLSTM in "200"
	do
		echo $LR $NLSTM
		qsub -F "$NLSTM $LR" 100_jobscript.sh -e /home/projects/vaccine/people/s143849/alternative_data/log_err/LSTM-$NLSTM-$LR-100.err -o /home/projects/vaccine/people/s143849/alternative_data/log_err/LSTM-$NLSTM-$LR-100.log -N LSTM-$NLSTM-$LR-100
		sleep 0.5
	done
done