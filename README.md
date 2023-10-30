# hsp_embedding
# Project title: Using Natural Language Processing (NLP)-Based Deep Learning Approach to Predict Hansen Solubility Parameters  

# Project Description
We used the Morgan fingerprint approach and two NPL-based embedding models to predict Hansen Solubility Parameters (HSPs). Five combinations of molecular representations and machine learning methods are provided here: Morgan fingerprints with XGBoost, Mol2Vec embeddings using XGBoost and FFNN, respectively and the two ChemBERTa finetuned models. Each of them has been applied to delta_d, delta_h, and delta_p and as a benchmark, the ESOL aqueous solubility parameters. 

# Requirements
NumPy

Matplotlib

Pandas

RDKit

Scikit-learn

Jupyter notebook

Pytorch

Huggingface/Transformers

Mol2vec

Bertviz

openbabel


# Installation
The code is written in jupyter notebook and runs in an Anaconda environment. The required packages and the guidance of how to set up the environment are listed below:

1. Users should install Anaconda first (documentation [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html))
2. Open Anaconda and install Rdkit (documentation [here](https://www.rdkit.org/docs/Install.html)) 

        conda create -c conda-forge -n my-rdkit-env rdkit 

3. After installation of RDkit, activate the Rdkit environment using the following command

        conda activate my-rdkit-env 

6. Within my-rdkit-env
•	Install Scikit-learn (documentation [here](https://scikit-learn.org/stable/install.html)) 

        conda install scikit-learn  

•	Install Jupyternotebook (documentation [here](https://jupyter.org/install)) 

    conda install -c conda-forge notebook  

•	Install pytorch (documentation [here](https://pytorch.org/get-started/locally/)) 

    conda install pytorch torchvision torchaudio -c pytorch

•	Install Transformers (documentation [here](https://huggingface.co/docs/transformers/v4.15.0/installation)) 

    pip install transformers

•	Install mol2vec (documentation [here](https://github.com/samoturk/mol2vec))

    pip install git+https://github.com/samoturk/mol2vec

•	Install BertViz (documentation [here](https://github.com/jessevig/bertviz/blob/master/README.md)) 

    pip install bertviz
    pip install ipywidgets

•	Install openbabel

    conda install -c conda-forge openbabel

A requirement.txt file is provided to be used to create a conda environment.    

# Disclaimer
The code was written by non-experts in order to perform a specific task with a specific dataset. Please exercise caution and initiative in deploying the code. 

# Usage
Users should first set up the Anaconda environment with the packages listed in Installation, then use jupyter notebook to run the code. 

In dataset/
The Hansen dataset  Hansen_1k_smiles_shorter.csv and ESOL datasets esol.csv are provided.

The two datasets were split for 6 folds for cross-validation and an example split is provided. 

esol_bert_ds*.csv

hansen_d_bert_ds*.csv

hansen_h_bert_ds*.csv

hansen_p_bert_ds*.csv


Code to train the models and make prediction is available in the following 5 folders. 

fps_xgboost/

mol2vec_xgboost/

mol2vec_ffnn/

ChemBERTa-zinc-base-v1/

ChemBERTa_77M_MTR/


Running time for the Morgan fingerprint (fps) models is under 1 min. 
Mol2vec models complete under ~5 mins. 
Finetuning BERT models requires GPUs and takes approximately 5-10 mins depending on the number of epochs. 

The predicted HSPs two standardised residual deviations (SRDs) away from the experimental values are selected as the outliers in each model. Examples of outliers analysis are provided in the following folds. The csv files in each folder contain the SMILES of molecules and the predicted and experimental HSPs from the 6 fold cross-validation of that model. 

fps_xgboost_outliers_analysis/

mol2vec_xgboost_outliers_analysis/

mol2vec_ffnn_outliers_analysis/

ChemBERTa_zinc-base_v1_analysis/

ChemBERTa_77M_MTR_outliers_analysis/


Code to analyse the functional groups of the outlier is provided in outliers_functional_groups_analysis/


# Contact
Jiayun Pang, twitter handle @JiayunPang, email j.pang@gre.ac.uk
