# YAD_survey
Official repository for project YAD survey

# EGD
endoscopy segmentation and depth prediction

## Train

#### Training segmentation
Run this code when you want to train the segmentation model that is used as feature extraction model when classifying
```
$  python train_seg.py
```

#### Training 3-way classifier
Run this code to train 3-way classifier which stratifies AGC, EGC(T1a/b into single category), and BGU
```
$  python train_classify_merge.py
```

#### Training EGC T1a / EGC T1b classifier
Run this code to train binary classifier which stratifies EGC into T1a phase and T1b phase
```
$  python train_classify_T1ab_binary.py
```

- **Hyperparameters** such as batch size can be modified directly in the python script directly with ```settings```
python dictionary.

## Tensorboard
```
$  tensorboard --logdir=logs
```
Running this command will activate tensorboard and you will be able to see all the logged information.

Upon running the command, go to [http://localhost:6006](http://localhost:6006)


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
