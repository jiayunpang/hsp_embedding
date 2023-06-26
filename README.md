# hsp_embedding
# Project title
Using Natural Language Processing (NLP)-Based Deep Learning Approach to Predict Hansen Solubility Parameters  

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

1. Users should install Anaconda first (documentation [[here]([url](https://conda.io/projects/conda/en/latest/user-guide/install/index.html))] )
2. Open Anaconda and install Rdkit (documentation here) 
conda create -c conda-forge -n my-rdkit-env rdkit 

3. After installation of RDkit, activate the Rdkit environment using the following command
conda activate my-rdkit-env 

4. Within my-rdkit-env
•	Install Scikit-learn (documentation here) 

conda install scikit-learn  

•	Install Jupyternotebook (documentation here) 

conda install -c conda-forge notebook  

•	Install pytorch (documentation here) 

conda install pytorch torchvision torchaudio -c pytorch

•	Install Transformers (documentation here) 

pip install transformers

•	Install mol2vec https://github.com/samoturk/mol2vec

pip install git+https://github.com/samoturk/mol2vec

•	Install BertViz (documentation here) 

pip install bertviz
pip install ipywidgets

•	Install openbabel

conda install -c conda-forge openbabel
