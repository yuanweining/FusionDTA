# FusionDTA: attention-based feature polymerizer and knowledge distillation for drug–target binding affinity prediction

## Data Preprocess
You first need to generate the standard data, cold-start data, 5-fold cross-validation data and protein representation from raw data.   
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
11-12: protein representation from pretraining model namely esm-1b (https://github.com/facebookresearch/esm).

Tips: Perhaps you don't have enough CUDA memory to generate the pretrained representation of the protein, hence we provide a cloud address for the protein representation file.  
[davis.npz](https://drive.google.com/file/d/1EF4MdCPJ_8bWgdABK_GTKbTqROYtObv6/view?usp=sharing)  
[kiba.npz](https://drive.google.com/file/d/1V0DRxVpfdle91-yiUdZ7LtVZ_YsAodPi/view?usp=sharing)  
Then you can generate the csv data locally by commenting out line 70： 

    generate_protein_pretraining_representation(dataset, prots)  
    
or download generated data from [data.zip](https://drive.google.com/file/d/1evKHwYmUkpJAR_BWK-WTMASKx3Fcivn0/view?usp=sharing).

## Training
First you should create a new folder for saved models, path = "FusionDTA/saved_models".  
Then run command below to train FusionDTA.

    python training.py
  
## Cold-start
Under the constraints of cold-start, FusionDTA can only predict binding affinity from unseen protein, unseen drug and both of them.  
Run command below to train drug cold-start.

    python training_drug_cold.py
    
Run command below to train protein cold-start.

    python training_protein_cold.py
    
Run command below to train protein-drug cold-start.

    python training_drug_protein_cold.py

  
## testing
Run command below for validation.

    python validating.py 

## Knowledge distillation
First you should create a folder for pretrained models, path = "FusionDTA/pretrained_models".  
Then download pretrianed models from [DAT_best_davis.pkl](https://drive.google.com/file/d/1FfFLPhM2-97qvgkzcTiU30PluRPCm6vU/view?usp=sharing), or just use the trained model from "FusionDTA/save_models".  
run command below to train FusionDTA-KD.

    python training_KD.py
    
You can test different scales of parameters of the student model, by setting different arguments in parser.
