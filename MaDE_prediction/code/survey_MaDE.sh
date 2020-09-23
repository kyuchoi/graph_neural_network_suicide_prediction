#!/bin/sh

## NEVER give a space between file names when you enter the input files !!

python data_read.py --new_subsample # NEED it to obtain edge matrix using qgraph, when you process just a single case
Rscript EBICglasso_qgraph_MaDE.R
#python main_made.py --num_epochs 5 # uncomment this ONLY, when you predict just a single case
python make_pseudolabels.py --single_prediction # for a single test case prediction


