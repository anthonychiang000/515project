# 515project
https://www.sfu.ca/~ssurjano/goldpr.html
summarized breakdown of tasks:

    Data Visualization
    1a - Make dense heatmap of the function over certain domain
    1b - determine whether stationary, find transformation to make more stationary
    2a - Make kernel density estimate of distribution of values for benchmarks, interpret them
    2b - Find transformation that makes performance "better behaved"? (stationary i guess)

    Model Fitting
    1 - Choose 32 points for a dataset using a Sobol sequence and then find the value/label
    2 - Fit a GP model using constant mean and squared exp covariance, maximize marginal likelihood of data as function of hyperparameters
    2b - what values did you learn for hyperparams and do they agree with expectations
    3 - make heatmap of GP posterior mean, compare predicted values with true values, any systematic errors?
    4 - make heatmap of GP posterior SD, do values and scale make sense, does SD drop to near 0 at data points
    5 - make kernel density estimate of Z-scores of residuals between posterior mean and true values, should be std normal
    6 - Repeat using log transformation to output of function, does marginal likelihood improve, does it appear better calibrated
    7 - compute BIC score and model from last part
    8 - attempt search over models as fxn of the choice of mean and covariance functions to find best possible explanation of data
    8b - what is the best model and BIC score? 
    9 - perform similar search for SVM and LDA datasets, with 32 randomly sampled observations for each dataset

    Bayesian Optimization
    1 - implement expected improvement acquisition function for minimization
    2 - make new heatmaps for posterior mean and SD from datapoints we used before
    2b - make heatmap for EI value and mark where maximized, does it look like a good observation location?

    For each of G-P, SVM, LDA functions, implement this:
    1 - select 5 random observations as D
    repeat 30 times:
        2 - find point x that maximizes EI of current data
        3 - add observation to dataset
    4 - return dataset of 35 points
    5 - identify best points found using gap measure
    
    Perform the above experiment 20 times using diff random initializations and store all observations

    6 - use random search as a baseline for 150 observations
    7 - Make plot of learning curves for each methods on each of the datasets
    8 - what is the gap for EI and random search (see paper)

    Bonus - Take data in new direction