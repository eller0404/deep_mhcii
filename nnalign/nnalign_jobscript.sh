#!/bin/csh

set NODE="thinnode"
set TIME="01:00:00:00"
set MEMORY="30gb"
#for i in "$@"
#do
#case $i in
#    -t=*|--time=*)
#    TIME="${i#*=}"
#    ;;
#    -m=*|--mem=*)
#    MEMORY="${i#*=}"
#   ;;
#   --fat)
#    NODE="fatnode"
#    ;;
#    -*)
#    echo "argument not found: ${i}"
#    exit 1
#esac
#done

echo NODE = $NODE
echo TIME = $TIME
echo MEMORY = $MEMORY

set FOLDER = "/home/projects/vaccine/people/s143849/alternative_data"

set NNLIN = "/home/projects/vaccine/people/morni/bin/nnalign_gaps_pan_play_MA_wgt_MN_v2_two_outputs_context_allelelist"
set WDIR = "$FOLDER/../alternative_script/nnalign"
set ALLELELIST = "$FOLDER/birkir/allelelist.txt"
set PSEUDO = "/home/projects/vaccine/people/birey/dat/nnalign/pseudosequence.2016.all.X.dat"
set MPAT = "/home/people/cadunl/nnalign-2.0/data/BLOSUM50"
set BLF = "/home/projects/vaccine/people/birey/dat/nnalign/blosum62.freq_rownorm"
set PEP = "$FOLDER/jupyter/s_filter/small_13_21.txt"

foreach counter ( 1 2 3 4 5 )
    foreach context ("" " -context")
        echo $counter
        echo $context
        if ("$context" == "") then
            set SYN_NR = "syn3"
            set EXT = "uc"
            set EVAL = "eval21"
        else if ("$context" == " -context") then
            set SYN_NR = "syn4"
            set EXT = "wc"
            set EVAL = "eval31"
        endif

        set SYN = "/home/projects/vaccine/people/birey/ELII/res/pipeline_2019_06_12/nnalign/$EVAL/$SYN_NR-2_10_20_40_60-$counter-10_1-9.txt"
        set OUTPUT = "$FOLDER/log_err/small_auc_$counter-$EXT.log"
        set ERROR = "$FOLDER/log_err/small_auc_$counter-$EXT.err"

        set RUN = "$counter-$EXT.csh"
        echo '#\!/bin/csh' > $RUN
        echo "$NNLIN$context -aX -afs -encmhc -elpfr 0 -eplen -1 -fl 3 -l 9 -nout 1 -allelelist $ALLELELIST -blf $BLF -classI 13,14,15,16,17,18,19,20,21 -mhc $PSEUDO $SYN $PEP" >> $RUN
        echo "# EOF" >> $RUN

        qsub -d ${WDIR} -W group_list=vaccine -A vaccine -l nodes=1:ppn=1:${NODE},mem=${MEMORY},walltime=${TIME} -m be -M asbjorn.skaarup@gmail.com -e ${ERROR} -o ${OUTPUT} -N $counter-$EXT $RUN
        sleep 0.5
    end
end
