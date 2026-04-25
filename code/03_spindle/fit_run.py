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
from fit_boot import *

####################################################################################################
cogFileCorrected = '/home/exx/Desktop/spindleOpt/age_cog_data_corrected.csv'
temp = pd.read_csv(cogFileCorrected).dropna()
temp = temp.rename(columns={"Subject_ID": "ID"})

cogFile = '/home/exx/Desktop/spindleOpt/age_cog_data.csv'
Brain_age = pd.read_csv(cogFile).dropna()

id_pairs = pd.read_csv('/home/exx/Desktop/spindleOpt/spindles_id_pairs.csv')
np.array(id_pairs)
id_pairs_df = id_pairs.rename(columns={'ID':'ID_new', 'ID_old':'ID'})


output = '/home/exx/Desktop/spindleOpt/results/BridgeResults08152020_prelim_subtests.csv'


with open(output, 'w+') as f:
    col_names = ['cycles', 'fc', 'q', 'th', 'minT', 'maxT', 'merge', 'CH',
                 'cog_test', 'pearson_corr', 'pearson_pval',  'spearman_corr',
                 'spearman_pval', 'FFT', 'DUR', 'DENS', 'AMP', 'mses',
                 'alphas', 'lambdas', 'lower_CI', 'upper_CI', 'pearson_lowerCI',
                 'pearson_upperCI','spearman_lowerCI','spearman_upperCI',
                 'FFT_1000', 'DUR_1000', 'DENS_1000', 'AMP_1000',
                 'pearson_corr_1000', 'spearman_corr_1000', 'sample_size']
    
    for col in col_names:
        f.write(col + ',')
    f.write('\n')

input_path = '/home/exx/Desktop/spindleOpt/results'
files = [f.path for f in os.scandir(input_path) if f.is_dir()]

run = 1
for f in files:
    try:
        print('Starting run #: ' + str(run))

        run+=1
        n2_avg_path =  f + '/N2_SpindleAvg.csv'
        n2_data = pd.read_csv(n2_avg_path)
        Brain_age = pd.read_csv(cogFile).dropna()
        
        comb = f.split('/')[-1].split('_')

        if len(comb[1]) > 6:
            fc_2 =True
        else:
            fc_2 =False

        #need to match IDs
        n2_data = pd.merge(n2_data, id_pairs_df, how='inner', on='ID')
        n2_data = n2_data.rename(columns={"ID": "ID_old", "ID_new": "ID"})

        #split into 3 data groups by ch *average c3-m2 and c4-m1, run lasso on each group
        cols = n2_data.columns.drop(['ID', 'CH', 'ID_old'])
        n2_data[cols] = n2_data[cols].apply(pd.to_numeric, errors='coerce')
        cog_tests = Brain_age.columns.drop(['ID', 'Age'])

        if fc_2:
            fast = n2_data[n2_data.F >13]
            fast_mean = fast.groupby('ID').mean()
            fast_mean = fast_mean.add_suffix('_fast').reset_index()
            slow = n2_data[n2_data.F <13]
            slow_mean = slow.groupby('ID').mean()
            slow_mean = slow_mean.add_suffix('_slow').reset_index()
            
            Brain_age = pd.merge(fast_mean, Brain_age, how='inner', on='ID')
            Brain_age = pd.merge(slow_mean, Brain_age, how='inner', on='ID')

        else:
            data = n2_data.groupby('ID').mean().reset_index()
            Brain_age = pd.merge(data, Brain_age, how='inner', on='ID')

            
        #remove excluded subjects 
        Brain_age = pd.merge(Brain_age, temp[['ID']], how='inner', on='ID')

        #define subjects ID as a variable
        sids = Brain_age.ID.values


        if fc_2: 
            for cog in cog_tests:

                with open(output, 'a+') as f:
                    for item in comb:
                        f.write(str(item) + ',')
                        
                y2 = Brain_age[cog] #value to predict
                X2 = Brain_age[['FFT_slow', 'DUR_slow', 'DENS_slow', 'AMP_slow',
                         'FFT_fast', 'DUR_fast', 'DENS_fast', 'AMP_fast']] #  X is all the other variables you want to include in the lasso (with which you want to predict cognitive function). Drop all other variables you don't want to use.
                                                       #you dont want age or any cognitive measures in X2
                Xnames = X2.columns

                # To use X and y, we need to transform them from a df to ndarray. 
                y = y2.values
                X = X2.values

                #try:
                #do regression
                (yptes, ytes, coefs, mses, alphas, lambdas,
                pc, pp, sc, sp) = bootstrap(X, y, sids, Brain_age)

               #do bootstrapping
                (lower, upper, lower_pcs, upper_pcs,
                lower_scs, upper_scs, pcs, scs) = bootstrap(X, y, sids, Brain_age, n_iter=1000)

                coefs = coefs[0]

                #save to csv
                with open(output, 'a+') as f:
                    writer = csv.writer(f)
                    writer.writerow(['avg_c3+c4'] + [cog] + [pc] + [pp] + [sc] + [sp] +
                                    [coefs[0]] + [coefs[1]] + [coefs[2]] +
                                    [coefs[3]] +  [coefs[4]] + [coefs[5]] + [coefs[6]] +
                                    [coefs[7]] + [mses] + [alphas] + [lambdas] + [lower] +
                                    [upper] + [lower_pcs] + [upper_pcs] + [lower_scs] +
                                    [upper_scs] + [pcs] + [scs] + [len(y)] )
              #  except:
                 #   print(f)
        else:
            for cog in cog_tests[:7]:

                with open(output, 'a+') as f:
                    for item in comb:
                        f.write(str(item) + ',')
                        
                y2 = Brain_age[cog] #value to predict
                X2 = Brain_age[['FFT', 'DUR', 'DENS', 'AMP']] #  X is all the other variables you want to include in the lasso (with which you want to predict cognitive function). Drop all other variables you don't want to use.
                                                       #you dont want age or any cognitive measures in X2
                Xnames = X2.columns

                # To use X and y, we need to transform them from a df to ndarray. 
                y = y2.values
                X = X2.values

                #try:
                #do regression
                (yptes, ytes, coefs, mses, alphas, lambdas,
                pc, pp, sc, sp) = bootstrap(X, y, sids, Brain_age)

               #do bootstrapping
                (lower, upper, lower_pcs, upper_pcs,
                lower_scs, upper_scs, pcs, scs, coefs1000) = bootstrap(X, y, sids, Brain_age, n_iter=1000)

                coefs = coefs[0]
                coefs1000 = np.asarray(coefs1000)
                coefs1000 = np.transpose(coefs1000)

                #save to csv
                with open(output, 'a+') as out:
                    writer = csv.writer(out)
                    writer.writerow(['avg_c3+c4'] + [cog] + [pc[0]] + [pp[0]] + [sc[0]] + [sp[0]] +
                                    [coefs[0]] + [coefs[1]] + [coefs[2]] +
                                    [coefs[3]] + [mses[0]] + [alphas[0]] + [lambdas[0]] + [lower] +
                                    [upper] + [lower_pcs] + [upper_pcs] + [lower_scs] +
                                    [upper_scs] + [list(coefs1000[0])] + [list(coefs1000[1])] +
                                    [list(coefs1000[2])] + [list(coefs1000[3])] +[pcs] + [scs] + [len(y)] )
    except:
        print(f)
        






           
            

