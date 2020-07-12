# AMT for Doc-Summ alignment
Manual Annotation Collection (AMT) for paper paper "*Xinnuo Xu, Ondrej Dusek, Jingyi Li, Yannis Konstas, and Verena Rieser*. Fact-based Content Weighting for Evaluating Abstractive Summarisation" *Proceedings of ACL2020* :tada: :tada: :tada:

## Quick Start

### Step1: Install Anaconda
```
wget https://repo.anaconda.com/archive/Anaconda2-2019.10-Linux-x86_64.sh
sh Anaconda2-2019.10-Linux-x86_64.sh
source .bashrc 
```
### Step2: Create python3.6 env

```
conda create -n HRes python=3.6
conda activate HRes
```
### Step3: Install pipenv, yarn, flask, and git

For `pipenv`
```
conda install -c conda-forge pipenv
```

For `yarn`
```
conda install -c conda-forge yarn 
```

For `git`
```
conda install -c anaconda git
```

For `flask`
```
conda install -c anaconda flask
```

In case some packages used by flask is missing, run

```
conda install -c carta jwt
conda install -c conda-forge flask-sqlalchemy
conda install -c conda-forge python-dotenv
```

### Step4: Clone this repo and
At the repo's root, run 
```
yarn install
pipenv install
```
(In Linux delete the `"fsevents": "^2.1.2",` from `package.json`. `fsevents` is only for Mac OS)

### Step5: Build the repository
Open terminal at this repo's root directory. Run
```
yarn build
yarn serve
```

### Step6: Start flask
Open another terminal window at the same location and run 
```
export FLASK_APP='backend/app'
flask run
```

Now you should be able to go to `localhost:8080/admin` to start playing with the interface.

### Step7: Setup an env on AWS

First, you should setup the correct [Security group rules](https://aws.amazon.com/premiumsupport/knowledge-center/connect-http-https-ec2/)

Then, install gunicorn by running
```
conda install -c anaconda gunicorn
```

Last, instead of running `flask run` in Step6, run `sh run_gunicorn.sh`

### Step8: For some functions
```
yarn add vue-slider-component
(npm install --save vue-slider-component)
```
