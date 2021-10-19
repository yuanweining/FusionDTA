# FusionDTA: attention-based feature polymerizer and knowledge distillation for drug–target binding affinity prediction

## Data Preprocess
You first need to generate the standard data, cold-start data and 5-fold cross-validation data from raw data. 
Run command as follows:
    
    python datahelper.py

Then you will get the following files:
* data/davis_test.csv
* data/davis_train.csv
* data/kiba_test.csv
* data/kiba_train.csv
* data/davis_cold.csv
* data/kiba_cold.csv
* data/davis/davis_train_fold0-5.csv
* data/davis/davis_test_fold0-5.csv
* data/kiba/kiba_train_fold0-5.csv
* data/kiba/kiba_test_fold0-5.csv
* davis.npz
* kiba.npz


1-4: standard data
5-6: cold-start data
7-10: 5-fold cross-validation data
11-12: protein presentation from pretraining model(esm-1b). 

## Training
First you should create a new folder for trained models, path = "FusionDTA/save_models".

then run command below to train FusionDTA.

    python training.py
  
## testing
run command below for validation.

    python validating.py 

## Knowledge distillation
First you should create a folder for pretrained models, path = "FusionDTA/pretrained_models".

Then download pretrianed models from https://diuyourmouse.cn/FusionDTA/training_best.pkl, or just use the trained model from "FusionDTA/save_models".

run command below to train FusionDTA-KD.

    python training_KD.py
    
You can test different scales of parameters of the student model, by setting different arguments in parser.
