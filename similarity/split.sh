#!/bin/sh
qsub -F "0 200" similarity.sh
sleep 0.5
qsub -F "200 400" similarity.sh
sleep 0.5
qsub -F "400 600" similarity.sh
sleep 0.5
qsub -F "600 800" similarity.sh
sleep 0.5
qsub -F "800 1000" similarity.sh
sleep 0.5
qsub -F "1000 1200" similarity.sh