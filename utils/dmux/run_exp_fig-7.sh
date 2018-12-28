#!/bin/bash

N_JOBS=25
N_DATA=2975

step=$(expr ${N_DATA} / ${N_JOBS} + 1)
gpu_i=0
i=0
while [ $i -lt $N_DATA ]
do
    start_i=$i
    i=$(expr $i + $step)
    if [ $i -ge $N_DATA ]
    then
        i=$N_DATA
    fi
    echo "START:$start_i END:$i"
    dmux run -- bash utils/dmux/ablation_study.sh 2 30 300 $start_i $i
    dmux run -- bash utils/dmux/ablation_study.sh 3 30 300 $start_i $i
    dmux run -- bash utils/dmux/ablation_study.sh 4 30 300 $start_i $i
    dmux run -- bash utils/dmux/ablation_study.sh 5 30 300 $start_i $i
    dmux run -- bash utils/dmux/ablation_study.sh 6 30 300 $start_i $i
    dmux run -- bash utils/dmux/ablation_study.sh 7 30 300 $start_i $i
    dmux run -- bash utils/dmux/ablation_study.sh 8 30 300 $start_i $i
done
