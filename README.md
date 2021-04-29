# MAHOMES
Metal Activity Heuristic of Metalloprotein and Enzymatic Sites (MAHOMES) - Predicts if a protein bound metal ion is enzymatic or non-enzymatic

## Overview
The ability to distinguish enzyme from non-enzyme sites can be use for the study of enzymes and the design of enzymes. We've developed MAHOMES, a machine learning based tool which classifies metals bound to proteins as enzymatic or non-enzymatic. This repository contains the source code and necessary data for replicating the results in our manuscript. The following three functions are included: ML algorithms and feature set optimization, MAHOMES training and performance evaluation, and using MAHOMES to make predictions.

## System requirements
The source code in this repository has been tested using MacOS 10.15.7. Python scripts have also been tested using CentOS 7 using the KU Community Cluster. The source code is written in python with version 3.6.10. It also requires the following packages:
- numpy: https://numpy.org/install/
- pandas: https://pandas.pydata.org/docs/getting_started/index.html
- scikit-learn: https://scikit-learn.org/stable/install.html
- JupyterNotebook: https://jupyter.org

## Installation guide:
The source code in this repository does not require any instalation.

## Instructions for use:
This repository contains source code for three different functions.

1. ML algorithms and feature sets optimization:
    CV/MLwGrid.py uses nested cross validation to test the performance of an ML algorithm, feature set, and optmiization metric. Additional details and demo instructions are located at the top of the file.

2. MAHOMES training and performance evaluation:
    This can be done by going through each step of MAHOMES_eval_T-metal-site.ipynb. This notebook includes:
    1. Reading in the data-set and holdout T-metal-sites 
    2. Scaling the features
    3. Under-sampling the training data and removing features that are not a part of the all-category, mean sphere feature set
    4. Train an extra trees classifier to create a model that makes predictions for T-metal-sites and saves that model
    5. Repeat steps 3 and 4 with new random seeds nine times
    6. Average the predictions to get a final prediction which is rounded to 0 (non-enzyme) or 1 (enzyme)
    7. Calculate the performance metrics for the final T-metal-sites predictions
    This notebook should take less than five minutes.

3. Using MAHOMES to make predictions
    Prior to using MAHOMES to make predictions, features need to be calculated using third-party software. 
    1. Save a file containing calculated features as data/<job_name>/sites_calculated_features.txt (requires third-party tools, see methods in publication for details)
    2. From the MachineLearning directory, run "python MAHOMESNewPredictions.py <job_name>"
    3. The resulting data/<job_name>/sites_predictions.txt will contain the predictions for each site
        - final_prediction is MAHOMES enzyme or non-enzyme prediction
        - Prediction_<int> is the prediction made by one of the MAHOMES model using random seed <int>
    
    This can be tested using T-metal-sites for <job_name>. Because feature calculations require other software (Rosetta, Bluues, and FindGeo), we have provided the calculated features and expected predictions. Note that saving new models may result in small differences due to the stochastic nature of the random under-sampling the extra trees algorithm. However, the final_prediction should remain the same. The demo should take less than one minute.

