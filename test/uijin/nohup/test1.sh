#!/bin/bash

export PATH=~/anaconda3/bin:~/anaconda3/condabin:$PATH

. ~/anaconda3/etc/profile.d/conda.sh

conda activate

conda activate DF

today=`date +%y%m%d_%H%M%S`

echo $today

nohup python3 -u ../python/AASIST_test.py -val -es --config /home/aicontest/DF/config/AASIST.conf > /home/aicontest/DF/logs/uijin/log.$today 2>&1 &