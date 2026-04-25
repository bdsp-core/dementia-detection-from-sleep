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

output = '/home/exx/Desktop/spindleOpt/results/N2summary_07032020_rest.csv'
with open(output, 'a+') as f:
    col_names = ['cycles', 'fc', 'q', 'th', 'minT', 'maxT', 'merge', 'CH',
                  'N','N01', 'N02','DUR','DENS', 'AMP', 'FFT', 'FRQ', 'CHIRP','Q']
    for col in col_names:
        f.write(col + ',')
    f.write('\n')

input_path = '/home/exx/Desktop/spindleOpt/results'
files = [f.path for f in os.scandir(input_path) if f.is_dir()]
print(len(files))
run = 1
for f in files:
    try:
        print('Starting run #: ' + str(run))

        run+=1
        n2_avg_path =  f + '/N2_SpindleAvg.csv'
        n2_data = pd.read_csv(n2_avg_path)
        Brain_age = pd.read_csv(cogFile).dropna()
        
        comb = f.split('/')[-1].split('_')

        #need to match IDs
        n2_data = pd.merge(n2_data, id_pairs_df, how='inner', on='ID')
        n2_data = n2_data.rename(columns={"ID": "ID_old", "ID_new": "ID"})

        #split into 3 data groups by ch *average c3-m2 and c4-m1, run lasso on each group
        cols = n2_data.columns.drop(['ID', 'CH', 'ID_old'])
        n2_data[cols] = n2_data[cols].apply(pd.to_numeric, errors='coerce')
        cog_tests = ['NIH_Cog_Cognition_FluidComposite_Uncorrected_Standard_Score']


        data = n2_data.groupby('ID').mean().reset_index()
        Brain_age = pd.merge(data, Brain_age, how='inner', on='ID')

            
        #remove excluded subjects 
        Brain_age = pd.merge(Brain_age, temp[['ID']], how='inner', on='ID')

        #define subjects ID as a variable
        sids = Brain_age.ID.values

        x = list(Brain_age[['N','N01', 'N02','DUR','DENS', 'AMP', 'FFT', 'FRQ', 'CHIRP', 'Q']].mean())


        with open(output, 'a+') as f:
            for item in comb:
                f.write(str(item) + ',')
                
            writer = csv.writer(f)
            writer.writerow(['avg_c3+c4'] + [x[0]] + [x[1]] + [x[2]] + [x[3]] + [x[4]] +
                            [x[5]] + [x[6]] + [x[7]] + [x[8]] + [x[9]])
    except:
        pass







               
                

