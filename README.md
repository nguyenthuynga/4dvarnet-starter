# 4DVarNet

## Prerequisite
- git
- conda

## Install
### Install project dependencies
```
Install conda, python, etc, you can follow the following steps:
Install python environment in local laptop (first install conda  and then install environment)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -p $HOME/miniconda3 (after this a directory will be created to contain all library of conda)
Done with installing conda here!
Now let’s check whether we have conda or not:
bash (this is to change to Bash mode)
conda --version in case we want to check the version
eval "$(/home/t23nguye/miniconda3/condabin/conda shell.bash hook)" (please change the directory to yours)
conda init
Now let install the new environment:
cd /home/t23nguye/ (in order to install the new environment here)(please change the directory to yours)
conda create -n 4dvarnet-starter mamba python=3.9 -c conda-forge
cd $HOME
conda activate 4dvarnet-starter
mamba env update -f /home/t23nguye/Documents/4dvarnet-starter/From-GPU-IMT/4dvarnet-starter/environment.yaml (please change the directory to yours)
You may need to install ipython and omegaconfig if needed

```

### Download example data
https://drive.google.com/drive/folders/1HVLi1ka7omexkil-uBhvnBjk78gpCSEB


## Run
The model uses hydra see [#useful-links]
```
python main.py xp=base 
```
## Saved weights:



## Useful links:
- [Hydra documentation](https://hydra.cc/docs/intro/)
- [Pytorch lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/index.html#get-started)
- 4DVarNet papers:
	- Fablet, R.; Amar, M. M.; Febvre, Q.; Beauchamp, M.; Chapron, B. END-TO-END PHYSICS-INFORMED REPRESENTATION LEARNING FOR SA℡LITE OCEAN REMOTE SENSING DATA: APPLICATIONS TO SA℡LITE ALTIMETRY AND SEA SURFACE CURRENTS. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences 2021, V-3–2021, 295–302. https://doi.org/10.5194/isprs-annals-v-3-2021-295-2021.
	- Fablet, R.; Chapron, B.; Drumetz, L.; Mmin, E.; Pannekoucke, O.; Rousseau, F. Learning Variational Data Assimilation Models and Solvers. Journal of Advances in Modeling Earth Systems n/a (n/a), e2021MS002572. https://doi.org/10.1029/2021MS002572.
	- Fablet, R.; Beauchamp, M.; Drumetz, L.; Rousseau, F. Joint Interpolation and Representation Learning for Irregularly Sampled Satellite-Derived Geophysical Fields. Frontiers in Applied Mathematics and Statistics 2021, 7. https://doi.org/10.3389/fams.2021.655224.

