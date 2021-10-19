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
11-12: protein presentation from pretraining model namely esm-1b (https://github.com/facebookresearch/esm).

tips: Perhaps you don't have enough CUDA memory to generate the pretrained representation of the protein, hence we provide a cloud address for the protein representation file.  
[davis](https://drive.google.com/file/d/1EF4MdCPJ_8bWgdABK_GTKbTqROYtObv6/view?usp=sharing)  
[kiba](https://drive.google.com/file/d/1V0DRxVpfdle91-yiUdZ7LtVZ_YsAodPi/view?usp=sharing).  
Then you can generate the csv data locally by commenting out line 70： 

    generate_protein_pretraining_presentation(dataset, prots)  
    
or download generated data from ().

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
Then download pretrianed models from https://diuyourmouse.cn/FusionDTA/training_best.pkl, or just use the trained model from "FusionDTA/save_models".  
run command below to train FusionDTA-KD.

    python training_KD.py
    
You can test different scales of parameters of the student model, by setting different arguments in parser.
