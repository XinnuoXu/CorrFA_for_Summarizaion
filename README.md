# CorrFA_for_Summarizaion
Corr F/A evaluation metrics in paper "*Xinnuo Xu, Ondrej Dusek, Jingyi Li, Yannis Konstas, and Verena Rieser*. [Fact-based Content Weighting for Evaluating Abstractive Summarisation](https://www.aclweb.org/anthology/2020.acl-main.455.pdf)" *Proceedings of ACL2020* :tada: :tada: :tada:

## Environment setup

### Step1: Install pytorch env

```
(Install conda: wget https://repo.anaconda.com/archive/Anaconda2-2019.10-Linux-x86_64.sh; bash ~/Downloads/Anaconda2-2019.10-Linux-x86_64.sh; Reload)
conda create -n Highlight python=3.6
conda activate Highlight
conda install pytorch=1.1.0 torchvision cudatoolkit=10.0 -c pytorch
(or conda install pytorch torchvision cudatoolkit=10.0 -c pytorch; conda install pytorch=1.1.0 -c soumith)

pip install multiprocess
pip install pytorch_transformers
pip install pyrouge
pip install tensorboardX
```

### Step2: Install allennlp

```
pip install allennlp
wget https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz
mv srl-model-2018.05.25.tar.gz Evaluation/
```

## System attacking



