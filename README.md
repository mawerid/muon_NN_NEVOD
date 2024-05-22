# Application of deep learning methods for cosmic ray muon characterisation reconstruction from CWD NEVOD data

## Introduction

This is research about muon's tracks reconstruction by Deep Learning methods.
Work in Cherenkov water detector NEVOD, placed in NRNU MEPhI.

# Main goals

- Create model based on neural net for reconstruction track of single muons from Cherenkov Water Detector
- Create model for detection cascades of muons and their parameters
- Get enough speed for working with all current data flow (at least 100 Hz)
- Get excellent precision and accuracy of reconstructions (about 1 degree for each angle)

# Current progress

- 

# TO-DO

LOG structure:
MODEL_DATA_TYPE_ITER_TIME:
- checkpoints
- hyperparams
- weights
- other (?)

- [DONE] Restructure all experiment
- [IN PROGRESS] Create class of raw data generator for simulation (SCT, DECOR, full sphere)
- Data loader class (from raw data (exp/sim) to proper datasets for pytorch)
- Write and check pipeline with standard NN class model (learning, save and checkpoints and show stats, breakpoints)
- Write class for working NN model with structure
- Test proper cuda !working!
- Write a pipeline for test models

- Try 3D CNN (ResNet)
- Try GraphNN
- Different loss functions (Best is MSE)
- Different activation functions (Best ReLu)
- Different optimization functions (Best is adam)

