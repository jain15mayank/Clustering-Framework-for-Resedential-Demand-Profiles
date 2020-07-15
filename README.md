# Clustering-Framework-for-Resedential-Demand-Profiles

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript:

> Jain, M., AlSkaif, T. and Dev, S.(2020). A Clustering Framework for Residential ElectricDemand Profiles: A Case Study in Amsterdam. In: International Conference on Smart Energy Systems and Technologies (SEST).

## Scripts

+ `read_data.py`: reads the raw data from the CSV file that contains the electric load consumption data of multiple households for the year 2018-19.
+ `main.py`: main program. Currently, it does the following tasks:
  + loads the data                                : `read_data.py`
  + pre-processesing                              : `utils.utils.preProcessing_clustering`
  + dimensionality reduction                      : `utils.dimReduction.dim_reduction_PCA`, and `dim_reduction_FA`
  + plot elbow heuristics                         : `utils.dimReduction.elbowHeuristic_PCA`, and `elbowHeuristic_FA`
  + optimal no. of clusters (Spectral Clustering) : `utils.spectral.optimalKspectral`
  + perform Spectral Clustering                   : `utils.spectral.spectralClustering_KM_KNN_Euc`
  + optimal no. of clusters (K-Means Clustering)  : `utils.kmeans.optimalK`
  + perform K-Means Clustering                    : `utils.kmeans.kmeans`
  + perform objective validation                  : `utils.spectral.validate_spectral_clusters`, and `objectiveValidation.validate_clusters`
  + plot results for subjective validation        : `utils.utils.plotClusters`
+ `utils` directory contains the helpful functions which are used in the `main.py`

# Note:
The dataset used in this project can not be disclosed due to external reasons. However, one may feel free to use/modify the code as per the requirement.
