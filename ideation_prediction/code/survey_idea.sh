#!/bin/sh

# to make df_train_valid.csv first to run qgraph
python data_read.py

# for model except PHQ9, set argument except_PHQ9 as FALSE
Rscript EBICglasso_qgraph_smote.R --except_PHQ9 FALSE

echo $1 'is' $2 # $1: --mode, $2: 'single'

if [ "$2" == 'single' ]; then

	python test_idea2wks_ensemble_single_case.py
	python test_idea2wks_ensemble_single_case_saliency_test.py
else

	### for test group
	python test_idea2wks_smote_best_ensemble.py
	python test_idea2wks_smote_best_ensemble_saliency_test.py

fi


