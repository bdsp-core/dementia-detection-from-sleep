import os
import csv
import sys
import random
import itertools
import numpy as np
import pandas as pd
from LUNAspindles import main
from LUNAspindles import tools
from pathlib import WindowsPath
from pathlib import Path
from pandas import read_csv
from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, LinearRegression, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr, spearmanr, mode
import statsmodels.api as sm
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold
from LUNAspindles import main
from LUNAspindles import tools
import statistics
####################################################################################################

#define parameter ranges
cycles_lst = [8] #7
fc_lst = [14.5, 15, 15.5] #13.5
q_lst = [0]#np.linspace(0,1,11) #0

theta_lst = [4.5,5,5.5]  #th=4.5
min_t_lst = [0.4,0.2] #min=0.3
max_t_lst = [2.6,3.2,3.9]#np.linspace(2.2,4,19) #max=3
gap_lst = [0.6,0.7] #collate merge=0.5

all_param_lst = [cycles_lst,fc_lst,q_lst,theta_lst,min_t_lst,max_t_lst,gap_lst]

#find all possible combinations
combs = list(itertools.product(*all_param_lst))

cycles_lst = [12] #7
fc_lst = [12.5] #13.5
q_lst = [0]#np.linspace(0,1,11) #0

theta_lst = [5.5,4.5,5]  #th=4.5
min_t_lst = [0.4, 0.2] #min=0.3
max_t_lst = [2.6,3.9]#np.linspace(2.2,4,19) #max=3
gap_lst = [0.3,0.5,0.7] #

all_param_lst2 = [cycles_lst,fc_lst,q_lst,theta_lst,min_t_lst,max_t_lst,gap_lst]
#find all possible combinations
combs2 = list(itertools.product(*all_param_lst2))


cycles_lst = [6] #7
fc_lst = [12.5] #13.5
q_lst = [0]#np.linspace(0,1,11) #0

theta_lst = [5.5]  #th=4.5
min_t_lst = [0.4, 0.2] #min=0.3
max_t_lst = [2.6, 3.9]#np.linspace(2.2,4,19) #max=3
gap_lst = [0.3,0.5,0.7] #

all_param_lst2 = [cycles_lst,fc_lst,q_lst,theta_lst,min_t_lst,max_t_lst,gap_lst]
#find all possible combinations
combs3 = list(itertools.product(*all_param_lst2))


cycles_lst = [6] #7
fc_lst = [12.5] #13.5
q_lst = [0]#np.linspace(0,1,11) #0

theta_lst = [4.5,5]  #th=4.5
min_t_lst = [0.2] #min=0.3
max_t_lst = [3.9]#np.linspace(2.2,4,19) #max=3
gap_lst = [0.3,0.5,0.7] #

all_param_lst2 = [cycles_lst,fc_lst,q_lst,theta_lst,min_t_lst,max_t_lst,gap_lst]
#find all possible combinations
combs4 = list(itertools.product(*all_param_lst2))

combs = combs + combs2 + combs3 + combs4


#remove combs already analyzed
input_path = '/home/exx/Desktop/spindleOpt/results/07022020'
files = [f.path for f in os.scandir(input_path) if f.is_dir()]

done = []
for f in files:
    comb = tuple([float(c) for c in f.split('/')[-1].split('_')])
    done.append(comb)

l3 = [x for x in combs if x not in done]

#prep id pairs for matching IDs later
id_pairs = pd.read_csv('/home/exx/Desktop/spindleOpt/spindles_id_pairs.csv')

#loop through each comb in samp
run = 0
for comb in l3:
    print('Starting run #: ' + str(run))
    #parse
    cycles = comb[0]
    fc = comb[1]
    q = comb[2]
    th = comb[3]
    minT = comb[4]
    maxT = comb[5]
    merge = comb[6]

    comb_string = (str(cycles) + '_' + str(fc) + '_' + str(q) + '_' + str(th) +
                   '_' + str(minT) + '_' + str(maxT) + '_' + str(merge) )

    #delete old spindleCommand.txt file
    os.remove("spindleCommand.txt")
    print("spindleCommand.txt File Removed!")

    #first run luna
    lunaBaseDir = '/home/exx/luna-base/'
    LUNAsublist = '/home/exx/Desktop/spindleOpt/luna_sublist.lst'
    ch_to_analyze = ['C3-M2','C4-M1'] #['C3-C4','C3-M2','C4-M1','C3-avg(C3+C4)']
    output_results_path = '/home/exx/Desktop/spindleOpt/results/07022020/' + comb_string
    edf_dir ='/media/mad3/Projects/BrainAge_Cognition_EEG_data'
    sub_sublist = tools.sublist_filenames(edf_dir)
    
    main.run_luna(lunaBaseDir, LUNAsublist, ch_to_analyze, output_results_path,
             q, cycles, fc, th, minT, maxT, merge, sub_sublist=sub_sublist, N2=True, N3=False, so=False)


    #convert txt to csv file in output path
    tools.txt_to_csv(output_results_path, N2=True, N3=False)
    n2_avg_path = output_results_path + '/N2_SpindleAvg.csv'
