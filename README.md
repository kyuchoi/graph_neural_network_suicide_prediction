# Deep graph neural network-based prediction of acute suicidal ideation in young adults
_Official repository for "Deep graph neural network-based prediction of acute suicidal ideation in young adults" (Choi et al., Sci Rep, 2021)_
https://www.nature.com/articles/s41598-021-95102-7

The developed GIN-based network, _'MindWatchNet'_ predicts the presence of acute suicidal ideation (<2wks) using multi-dimensional questionnaires.
The _'MindWatchNet'_ is a two-staged model: first network is for MaDE prediction, making MaDE pseudolabel for acute suicidal ideation (<2wks) in the second network, using semi-supervised learning.

## MaDE prediction

#### Make MaDE pseudolabels for acute suicidal ideation (<2wks) prediction
Run this code when you want to make MaDE pseudolabels that is used as feature of data for acute suicidal ideation (<2wks) prediction
```
$  cd MaDE_prediction/code
$  bash survey_MaDE.sh
```
Put single case data as csv file in the following path: be sure to have the same file name (single_case_original.csv)
```
YAD_survey/MaDE_prediction/raw_data/raws_for_MaDE_pseudo/single_case_original.csv
```

- **Items for data for MaDE prediction** are 36 items as follows: 9 items in PHQ9, 7 items in GAD7, 20 items in STAI-X1. 
Please refer to ```YAD_survey/MaDE_prediction/raw_data/raws_for_MaDE_pseudo/single_case_original.csv```.

## Saliency result of representative case
![Average binary saliency for single case of suicidal_idea_within_2wk](https://user-images.githubusercontent.com/21270193/94001416-73be4e80-fdd3-11ea-80a6-3378177f696d.png)
![image](https://user-images.githubusercontent.com/21270193/94001349-58534380-fdd3-11ea-91d1-5f64a12a5cb5.png)
![Average saliency for single case per question of suicidal_idea_within_2wk](https://user-images.githubusercontent.com/21270193/94001478-85075b00-fdd3-11ea-9fdb-a7201bed469b.png)
![Average saliency for single case of suicidal_idea_within_2wk](https://user-images.githubusercontent.com/21270193/94001524-918bb380-fdd3-11ea-89b5-2ac2f7b177bc.png)

## Acute suicidal ideation (<2wks) prediction

#### Make acute suicidal ideation (<2wks) prediction using MaDE pseudolabels generated above
Run this code when you want to make acute suicidal ideation (<2wks) prediction that is the output of the ensemble of the three models with different subsamplings (i.e. u1, u2, and SMOTE) to overcome severe class imbalance.

Basically, this task is a binary classification with suicidal ideator as positive (minority class), and others as negative (majority class).

- **U1, U2, SMOTE models** indicates different models using undersampling of majority class with a ratio of 10, and 5 times to minority class, and oversampling of minority class, respectively.
```
$  cd ideation_prediction/code
$  bash survey_idea.sh --mode single # if you do a single case prediction
$  bash survey_idea.sh --mode group # if you do a group prediction for the test set
```
Put single case data as csv file in the following path: be sure to have the same file name (single_case_MaDE_pseudo.csv)
```
YAD_survey/ideation_prediction/raw_data/data_for_smote/single_case_MaDE_pseudo.csv
```

- **Items for data** are as follows: gender, site (screening=0, hospital=1, university counselling=2), 9 items in PHQ9, 7 items in GAD7, total scores of RAS, total scores of RSES, MaDE pseudolabel, suicidal attempt, true suicidal_idea_within_2wk label, and subject ID (any integer number). 
Please refer to ```YAD_survey/ideation_prediction/raw_data/data_for_smote/single_case_MaDE_pseudo.csv```.


## Requirements
Python3 (Anaconda) with following packages:
```
pytorch >= 1.5.0
cudatoolkit == 10.2
dgl-cu102
rstudio
```
MUST install dgl via: ```pip install dgl-cu102==0.4.3.post2```

Please use ```requirements.txt``` or ```environ_torch_dgl_r.yml``` for making Anaconda virtual environment.

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
