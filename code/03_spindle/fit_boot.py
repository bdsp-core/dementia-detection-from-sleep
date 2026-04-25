import os
import csv
import sys
import random
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from pathlib import WindowsPath
from LUNAspindles import main
from LUNAspindles import tools
from matplotlib import pyplot
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold
from sklearn import datasets, linear_model
from scipy.stats import pearsonr, spearmanr, mode
from sklearn.model_selection import train_test_split
import statistics
#define the regression model


def myfit(X, y, sids, test_sid_folds):
    coefs = []
    
    # standardize X
    X = (X-X.mean(axis=0)) / X.std(axis=0)
    
    #do the actual kfold model and calculate the mean alpha. 
   # k_fold = KFold(10, shuffle=True) 
    mses = []
    alphas = []
    lambdas = []
    ytes = []
    yptes = []
    sids_tes = []


    #split data in train set and test set
    for k, test_sid_fold in enumerate(test_sid_folds):
        test_ids = np.in1d(sids, test_sid_fold)
        train_ids = ~test_ids
        X_train = X[train_ids]
        X_test = X[test_ids] 
        y_train = y[train_ids]
        yte = y[test_ids]
        sids_te = sids[test_ids]
        #print('X_train shape', X_train.shape)
        
        # do lasso crossvalidation
        bridge = linear_model.BayesianRidge(n_iter=1000)

        #print(X_train, y_train)
        bridge.fit(X_train, y_train)
        ypte = bridge.predict(X_test)
        score = np.mean((yte - ypte)**2)
        ytes.extend(yte)
        yptes.extend(ypte)
        sids_tes.extend(sids_te)
        mses.append(score)
        alphas.append(bridge.alpha_)
        lambdas.append(bridge.lambda_)
        
    mean_mse = np.median(mses)
    mean_alpha = np.median(alphas)
    mean_lambda = np.median(lambdas)

    # now do the regression, using median alpha and lambda as alpha_init/lambda_init 

    bridge = linear_model.BayesianRidge(alpha_init=mean_alpha, lambda_init= mean_lambda, n_iter=1000)
    bridge.fit(X, y)
    #print(mses)
    #print(alphas)
    #print(lasso.coef_)
    return bridge.coef_, mean_mse, mean_alpha, mean_lambda, yptes, ytes, sids_tes

# do bootstrap to get the confidence interval around the coefs / OR NOT: 0 iterations = no confidence interval. 
def bootstrap(X, y, sids, Brain_age, n_iter=0, random_state=2019):
    np.random.seed(random_state)
    n_iterations = n_iter
    n_size = int(len(Brain_age) * 0.50)
    # run bootstrap
    stats = []
    N = len(X) ######
    coefs = []
    mses  = []
    alphas = []
    lambdas = []
    pcs = []
    pps = []
    scs = []
    sps = []

    X2, y2, sids2 = shuffle(X, y, sids, random_state=random_state)
    sids2=np.asarray(sids2)
    
    K = 10
    k_fold = GroupKFold(K)  # use GroupKFold to make sure no common sids (MRN) in training and testing folds (since sometimes there are multiple data with the same MRN, maybe not true in your case)

    test_sid_folds = []     # test_sid_folds is a list of the sids in each fold

    for train_ids, test_ids in k_fold.split(X2, y2, sids2):
        test_sid_folds.append(sids2[test_ids])

    
    for i in range(n_iterations+1):
        # generate bootstrapping ids
        ids = np.random.choice(N, N, replace=True)

        if i==0:
            Xbt = X
            ybt = y
            sidsbt = sids
        else:
            # use the bootstrapping ids to index X and y
            Xbt = X[ids]
            ybt = y[ids]
            sidsbt = sids[ids]

        # fit the model using Xbt and ybt
        coef, mean_mse, mean_alpha, mean_lambda, yptes, ytes, sids_tes_ = myfit(Xbt, ybt, sidsbt, test_sid_folds)

        # calculate the Pearson correlation between cognitive function and predicted cognitive function
        pc, pp =  pearsonr(yptes, ytes)
        sc, sp =  spearmanr(yptes, ytes)
        
##        if i==0:
##            yptes = yptes_
##            ytes = ytes_
##            sids_tes = sids_tes_

        # append to coefs
        coefs.append(coef)
        mses.append(mean_mse)
        alphas.append(mean_alpha)
        lambdas.append(mean_lambda)
        pcs.append(pc)
        pps.append(pp)
        scs.append(sc)
        sps.append(sp)

    if n_iter>0:
        # compute the upper and lower bound for the confidence interval of each coef
        alpha = 0.95
        p_lower = ((1.0-alpha)/2.0) * 100
        p_upper = (alpha+((1.0-alpha)/2.0)) * 100
        lower = np.percentile(coefs[1:], p_lower, axis=0)
        upper = np.percentile(coefs[1:], p_upper, axis=0)
        lower_pcs = np.percentile(pcs[1:], p_lower)
        upper_pcs = np.percentile(pcs[1:], p_upper)
        lower_scs = np.percentile(scs[1:], p_lower)
        upper_scs = np.percentile(scs[1:], p_upper)

        return (lower, upper, lower_pcs, upper_pcs,
            lower_scs, upper_scs, pcs, scs, coefs)
        
    #print('coefs[0].shape', coefs[0].shape)
    #big_ids = np.where(np.abs(coefs[0])>1e-6)[0]
    #output = (np.c_[np.array(Xnames2)[big_ids], coefs[0][big_ids]])

    # convert from np.array to pd.dataframe
    #dataset = pd.DataFrame({'Variable': output[:, 0], 'coefs': output[:, 1]})
    #print(dataset)


    """
    # optional: plot the distribution of the coefs
    coefs = np.array(coefs)  # coefs.shape = (1000, #coef)
    print(coefs.shape)
    for i in range(coefs.shape[1]):
        plt.hist(coefs[:, i])
    """
    return (yptes, ytes, coefs, mses, alphas, lambdas,
            pcs, pps, scs, sps)
