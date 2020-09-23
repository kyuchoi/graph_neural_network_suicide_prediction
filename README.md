# YAD_survey
Official repository for project YAD survey

## MaDE prediction

#### Make MaDE pseudolabels for acute suicidal ideation (<2wks) prediction
Run this code when you want to make MaDE pseudolabels that is used as feature of data for acute suicidal ideation (<2wks) prediction
```
$  cd YAD_surve/MaDE_prediction/code
$  bash survey_MaDE.sh
```
Put single case data as csv file in the following path: be sure to have the same file name (single_case_original.csv)
```
YAD_survey/MaDE_prediction/raw_data/raws_for_MaDE_pseudo/single_case_original.csv
```

- **Items for data for MaDE prediction** are 36 items as follows: 9 items in PHQ9, 7 items in GAD7, 20 items in STAI-X1. 
Please refer to ```YAD_survey/MaDE_prediction/raw_data/raws_for_MaDE_pseudo/single_case_original.csv```.

- **Items for data** are as follows: gender, site (screening=0, hospital=1, university counselling=2), 9 items in PHQ9, 7 items in GAD7, total scores of RAS, total scores of RSES, MaDE pseudolabel, suicidal attempt, true suicidal_idea_within_2wk label, and subject ID (any integer number). 
Please refer to ```YAD_survey/ideation_prediction/raw_data/data_for_smote/single_case_MaDE_pseudo.csv```.

## Directories Structure
```
├─ideation_prediction
│  ├─checkpoint_thr6
│  │  └─loss
│  │      ├─SMOTE
│  │      ├─u1
│  │      └─u2
│  ├─code
│  │  └─__pycache__
│  ├─data_thr6
│  │  ├─single
│  │  │  └─single_test
│  │  ├─SMOTE
│  │  │  ├─test
│  │  │  ├─train
│  │  │  ├─train_valid_for_attention
│  │  │  └─valid
│  │  ├─u1
│  │  │  ├─test
│  │  │  ├─train
│  │  │  └─valid
│  │  └─u2
│  │      ├─test
│  │      ├─train
│  │      └─valid
│  └─raw_data
│      └─data_for_smote
└─MaDE_prediction
    ├─checkpoint
    │  └─loss
    ├─code
    │  └─__pycache__
    └─raw_data
        └─raws_for_MaDE_pseudo
```
```logs``` and ```checkpoints``` which are directories for saving logs and checkpoints, respectively are automatically
created when you run train files. You can use the saved parameters from the specified checkpoints folder for inference.
