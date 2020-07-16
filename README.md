# CorrFA_for_Summarizaion
Corr F/A evaluation metrics in paper "*Xinnuo Xu, Ondrej Dusek, Jingyi Li, Yannis Konstas, and Verena Rieser*. [Fact-based Content Weighting for Evaluating Abstractive Summarisation](https://www.aclweb.org/anthology/2020.acl-main.455.pdf)" *Proceedings of ACL2020* :tada: :tada: :tada:

<span style="background-color: #FFCC00">[VIDEO](https://virtual.acl2020.org/paper_main.455.html)</span>  <span style="background-color: #FFAA00">[SLIDES](https://drive.google.com/file/d/1lZYRWRwiEZ0hlIY0nMetv77EgScCVb9N/view?usp=sharing)</span>


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

## Evaluate your output with CorrF/A
### Scenario1: With plain text inputs
We need three files for the evaluation, documents(`SRC_PATH`), gold summaries(`GOLD_PATH`), and model generated summaries(`CAND_PATH`). The format for document-file is one document per line and sentences are jointed by '\t'. The format for both gold-summary-file and generated-summary-file is one summary per line. The i-th row of document-file is paired with i-th row in gold-summary-file and generated-summary-file. The number of lines in each file should be the same. Examples are shown in `./Data/50_files`, `./Data/50_files.gold`, `./Data/50_files.cand`. To calculate CorrF/A, run: 

```bash
#!/bin/bash

SRC_PATH='./Data/50_files.src'
GOLD_PATH='./Data/50_files.gold'
CAND_PATH='./Data/50_files.cand'

python evaluate.py \
        -src_path ${SRC_PATH} \
        -gold_path ${GOLD_PATH} \
        -cand_path ${CAND_PATH}  
```
The Corr-F and Corr-A will be printed out. Also, the content weights referring to gold summaries and generated summaries are saved in file `./Data/cw_gold` and `./Data/cw_cand` respectively. 

### Scenario1: With Tree structured inputs
If the trees are built and saved in files, the evaluation can be run as:
```bash
#!/bin/bash

TREE_PATH='./Data/bert.tree'

python evaluate.py \
        -src_path ${SRC_PATH} \
        -gold_path ${GOLD_PATH} \
        -cand_path ${CAND_PATH} \
        -tree_path ${TREE_PATH} \
        -run_srl False \
        -run_tree False
```
The example file `./Data/bert.tree` is generate in `./Data/` by running `python full_cases_tree.py bert`. The script reads processed tree MRs of documents, old summaries and generated summaries from 
```
./Data/full_cases/bert_src.tree
./Data/full_cases/bert_gold.tree
./Data/full_cases/bert_cand.tree
```
respectively. The format for these three files is similar with plain text inputs. The only difference is that sentences are represented in tree MRs.

## Examples Evaluation results
