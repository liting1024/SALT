<div align="center">
<h1> 🧂Give Me Some SALT </h1>
<h3> Structure-Aware Link Modeling for Temporal Weighted Link Prediction </h3>
</div>

This repository provides the official implementation of **Give Me Some SALT**: Structure-Aware Link Modeling for Temporal Weighted Link Prediction, a novel framework designed to tackle the challenges of temporal weighted link prediction, including long-tail distribution and short-term randomness.

Our codebase includes:
- The full implementation of our proposed **SALT** model.
- Includes all eight TWLP benchmark datasets used in our study.
- Setup for **baseline models** to ensure fair and reproducible comparison.

For more details, please refer to our paper[📄](https://cikm2025.org/)

## Environment
We recommend using Python 3.8+. Higher versions should also be compatible.
To install dependencies, run:
```bash
pip install -r requirements.txt
```

## Datasets

Download all datasets from [this link](https://figshare.com/s/641d0611f6298ce32ded), and extract them to the `./Data/` directory as follows:
```
Data/
├── DC/
├── HMob/
├── INVIS13/
├── INVIS15/
├── IoT/
├── LyonSchool/
├── T-Drive/
└── Thiers13/
```

## Reproduction
To directly reproduce our results, we provide pretrained models. For example, to reproduce results on the INVIS15 dataset, simply run:
```bash
python train.py --config ./SALT/config/INVIS15/SALT.yaml --reproduction true    # for reproduction
python train.py --config ./SALT/config/INVIS15/SALT.yaml --reproduction false   # for train
```

## Baselines

To ensure fair and reproducible comparisons, we use the official training code provided by each baseline whenever possible. if you want to run, please run the following command, such as for INVIS15:

```bash
# D2V
python ./Baseline/d2v/dyngraph2vec_INVIS15.py

# DDNE
python ./Baseline/ddne/DDNE_INVIS15.py

# ELSTMD
python ./Baseline/elstmd/E_LSTM_D_INVIS15.py

# EGCN-O
python train.py --config ./Baseline/evolve_gcn/config/INVIS15.yaml --reproduction false

# GC-LSTM
python train.py --config ./Baseline/gclstm/config/INVIS15.yaml --reproduction false

# WinGNN
## Step 1: Convert the data format
python Baseline/wingcn/data2wingnn.py
## Step 2: Train the model
python Baseline/wingcn/train.py

# GraphSSM
python train.py --config ./Baseline/graphssm/config/INVIS15.yaml --reproduction false

# STGSN
python ./Baseline/stgsn/STGSN_INVIS15.py

# GCN-GAN
python ./Baseline/gcn_gan/GCN_GAN_INVIS15.py

# IDEA
python ./Baseline/idea/IDEA_INVIS15.py
```

## Acknowledgements
We thank the authors of the following open-source projects for their valuable contributions to our implementation:

- [NCNC](https://github.com/GraphPKU/NeuralCommonNeighbor)  
- [Mamba](https://github.com/state-spaces/mamba)  
- [D2V](https://github.com/palash1992/DynamicGEM)  
- [DDNE](https://github.com/KuroginQin/OpenTLP)  
- [ELSTMD](https://github.com/jianz94/E-lstm-d/)  
- [EGCN-O](https://github.com/IBM/EvolveGCN)  
- [GC-LSTM](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)  
- [WinGNN](https://github.com/pursuecong/WinGNN)  
- [GraphSSM](https://github.com/EdisonLeeeee/GraphSSM)  
- [STGSN](https://github.com/KuroginQin/OpenTLP)  
- [GCN-GAN](https://github.com/KuroginQin/OpenTLP)  
- [IDEA](https://github.com/KuroginQin/IDEA)  




