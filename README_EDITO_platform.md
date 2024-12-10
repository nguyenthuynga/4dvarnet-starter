The purpose of this module is to reconstruct and map gappy, noisy satellite observations to obtain gap-free, less noisy reconstructions using a pretrained 4DVarNet model, as shown in the image below:
![4dVarNet Workflow](https://github.com/nguyenthuynga/4dvarnet-starter/blob/main/4dVarNet_workflow.png?raw=true)

The Jupyter notebook, turbidity_output_analysis.ipynb, explains the purpose of each cell at the beginning. When running, it returns a comparison of different mapping schemes: 4DVarNet, DInEOF, and eDInEOF:
![4dVarNet reconstruction](https://github.com/nguyenthuynga/4dvarnet-starter/blob/main/GWS_reconstruction.png?raw=true)

For more details about other technical aspects, such as CUDA version, Pytorch, and other library versions, you can find the information in this link: 
[https://docs.google.com/document/d/1rbCtkzSKczz4_xxPH5LHSynruZKv9Wfax0lCyAI5-U0/edit?usp=sharing](https://docs.google.com/document/d/1mR-0Sj28zl1R2r8qy-I2IjttZ5pO-4v8pbFvSPAhU9k/edit?usp=sharing)



## Useful links:
- [Hydra documentation](https://hydra.cc/docs/intro/)
- [Pytorch lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/index.html#get-started)
- 4DVarNet papers:
	- Fablet, R.; Amar, M. M.; Febvre, Q.; Beauchamp, M.; Chapron, B. END-TO-END PHYSICS-INFORMED REPRESENTATION LEARNING FOR SA℡LITE OCEAN REMOTE SENSING DATA: APPLICATIONS TO SA℡LITE ALTIMETRY AND SEA SURFACE CURRENTS. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences 2021, V-3–2021, 295–302. https://doi.org/10.5194/isprs-annals-v-3-2021-295-2021.
	- Fablet, R.; Chapron, B.; Drumetz, L.; Mmin, E.; Pannekoucke, O.; Rousseau, F. Learning Variational Data Assimilation Models and Solvers. Journal of Advances in Modeling Earth Systems n/a (n/a), e2021MS002572. https://doi.org/10.1029/2021MS002572.
	- Fablet, R.; Beauchamp, M.; Drumetz, L.; Rousseau, F. Joint Interpolation and Representation Learning for Irregularly Sampled Satellite-Derived Geophysical Fields. Frontiers in Applied Mathematics and Statistics 2021, 7. https://doi.org/10.3389/fams.2021.655224.

