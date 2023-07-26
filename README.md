# Broadening the Scope: Evaluating the Potential of Recommender Systems beyond prioritizing Accuracy
This repository contains the source codes and datasets of the paper _Broadening the Scope: Evaluating the Potential of Recommender Systems beyond prioritizing Accuracy_ accepted to The 17th ACM Recommender Systems Conference (RecSys 2023), LBR track.

To reproduce our work, you should create two different projects:

 1. The first project refers to the folder _my_sir_elliot_ that you can download from this [link](https://drive.google.com/file/d/1ipIoRxhy3HdMnoBnwvT72dwkCMLWW-JX/view?usp=share_link). We load this project on drive due to the GitHub size limits;
 2. The second project refers to the folders contained in this GitHub repository.

### FIRST PROJECT

The first project contains the source codes and datasets to reproduce the experiments regarding the training of the baselines. It relies on an ad-hoc version of [Elliot](https://elliot.readthedocs.io/en/latest/), which is an open-source recommendation framework. Please, refer to the official documentation of Elliot for more details about the framework.

In the folder _config_files_, you can find the configuration files used to train the models. Specifically, on these files, you can notice the hyper-parameter explorations we have performed.

In the folder _data_, you can find the datasets exploited in the paper. These datasets are supplied in their split version.

To run these codes, make sure to have a Python `3.8.0` or later version installed on your device. Firstly, download the zip file from the link provided before. Then, unzip the file. In the project's folder,  you may create the virtual environment with the requirements files we included as follows:

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

To train the models, run the following commands:

```
$ python -u start_experiments.py
```

You will find the results in the folder _results_ for each dataset.

### SECOND PROJECT

The second project contains the source codes and files needed to compute the Pareto-optimal configurations and the quality indicators of the Pareto frontiers. To run these codes, make sure to have a Python `3.8.0` or later version installed on your device. Then,  you may create the virtual environment with the requirements files we included as follows:

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

To obtain the results, run the following command:

```
$ python main.py
```

In  the ```main.py```, you should comment/uncomment some rows at your convenience to gather the results of quality indicators according to the multi-objective scenario you are dealing with. Specifically, you may follow the indications from row 9 to row 23 to set the objectives of the corresponding scenario. According to the chosen scenario, you should follow the indications for the reference point (rows 32-37), the ObjectiveSpace object (rows 43-48), and the storage of the results (rows 53-58).

You will find the results in the folder _results_.

### Explored Hyper-parameters

By exploiting the Elliot framework, we train 32 hyper-parameters combinations for each model (`EASER`, `MultiVAE`, `LightGCN`, `RP3beta`, `UserKNN`). You may find the set of the explored hyper-parameters in each file belonging to the _config_files_ folder in the _my_sir_elliot_ project. For your convenience, we report such parameters in the following:

1. For the `EASER` model, we explore:
   
   - `l2_norm`: _[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 550]_
     
2. For the `LightGCN` model, we explore:
   
   - `factors`: _[64, 128, 256, 512]_
   - `l_w`: _[1e-4, 1e-3]_
   - `n_layers`: _[1, 2, 3, 4]_
     
3. For the `MultiVAE` model, we explore:
   
   - `intermediate_dim`: _[50, 100, 200, 300]_ 
   - `latent_dim`: _[50, 100, 200, 300]_ 
   - `dropout_pkeep`: _[ 0.8, 1]_
     
4. For the `RP3beta` model, we explore:
   
   - `neighborhood`: _[ 10, 100, 200, 300]_
   - `alpha`: _[ 0.6, 1,  1.5, 2]_
   - `beta`: _[ 0.8, 1.5 ]_
     
5. For the `UserKNN` model, we explore:
   
   - `neighbors`: _[ 10, 20, 30, 50, 100, 150, 200, 250 ]_ 
   - `similarity`: _[ cosine, dot, euclidean, manhattan ]_


