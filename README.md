# FusionDTA: attention-based feature polymerizer and knowledge distillation for drug–target binding affinity prediction

## Data Preprocess
You first need to generate the standard data, cold-start data and 5-fold cross-validation data from raw data. 
Run command as follows:
    
    python datahelper.py

Then you will get the following files:
1. data/davis_test.csv
2. data/davis_train.csv
3. data/kiba_test.csv
4. data/kiba_train.csv
5. data/davis_cold.csv
6. data/kiba_cold.csv
7. data/davis/davis_train_fold0-5.csv
8. data/davis/davis_test_fold0-5.csv
9. data/kiba/kiba_train_fold0-5.csv
10. data/kiba/kiba_test_fold0-5.csv
11. davis.npz
12. kiba.npz


1-4: standard data  
5-6: cold-start data  
7-10: 5-fold cross-validation data  
11-12: protein presentation from pretraining model (esm-1b https://github.com/facebookresearch/esm).  

## Training
First you should create a new folder for saved models, path = "FusionDTA/saved_models".  
then run command below to train FusionDTA.

    python training.py
  
## cold-start
Under the constraints of cold-start, FusionDTA can only predict binding affinity from unseen protein, unseen drug and both of them.  
run command below to train drug cold-start.

    python training_drug_cold.py
    
run command below to train protein cold-start.

    python training_protein_cold.py
    
run command below to train protein-drug cold-start.

    python training_drug_protein_cold.py

  
## testing
run command below for validation.

    python validating.py 

## Knowledge distillation
First you should create a folder for pretrained models, path = "FusionDTA/pretrained_models".  
Then download pretrianed models from https://diuyourmouse.cn/FusionDTA/training_best.pkl, or just use the trained model from "FusionDTA/save_models".  
run command below to train FusionDTA-KD.

    python training_KD.py
    
You can test different scales of parameters of the student model, by setting different arguments in parser.
