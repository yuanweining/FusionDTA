# FusionDTA: attention-based feature polymerizer and knowledge distillation for drug–target binding affinity prediction

## Data Preprocess
You should unzip the zip package in the Data folder. Then you will get the following files:
* davis_test.csv
* davis_train.csv
* kiba_test.csv
* kiba_train.csv
* protVec_100d_3grams.csv
The first four are the data from the baseline dataset and the last is the lookup table for the pre-trained protein.

## Training
First you should create a new folder for trained models, path = "FusionDTA/save_models".
then run command below to train FusionDTA.

    python training.py
  
## testing
run command below for validation.

    python validating.py 

## Knowledge distillation
First you should create a folder for pretrained models, path = "FusionDTA/pretrained_models".
Then download pretrianed models from https://diuyourmouse.cn/FusionDTA/DAT_KD_best.pkl or just used the trained model from "FusionDTA/save_models".
run command below to train FusionDTA-KD.

    python training-KD.py
    
You can test different scales of parameters of the student model, by setting different arguments in parser.
