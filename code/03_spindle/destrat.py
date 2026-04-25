import sys
import csv
import os
import os.path
import subprocess
import pyedflib
from statistics import mode
from joblib import Parallel, delayed
from pathlib import Path
from pathlib import WindowsPath
import pandas as pd

def destrat(lunaBaseDir, swAvg=False, swData=False, swPhase=False,
            spindleAvg=False, spindleData=False):

    destrat = lunaBaseDir + 'destrat'
    s = '2' #N2

    from pathlib import Path
    input_path = '/home/exx/Desktop/spindleOpt/results'
    p=Path(input_path)

    f1 = [x for x in p.iterdir() if x.is_dir() and len(list(x.glob('*.db')))>1]

    dirs = [x for x in p.iterdir() if x.is_dir() and len(list(x.glob('*.db')))<1]
    f2= []
    for p in dirs:
        f2 +=[x for x in p.iterdir() if x.is_dir() and len(list(x.glob('*.db')))>1]
    files = f1+f2

    ouput_param = ['CH','CH N','CH PHASE','CH F','CH F SPINDLE']
    name = ['swAvg', 'swData', 'swPhase', 'SpindleAvg', 'SpindleData']
    criteria = [swAvg, swData, swPhase, spindleAvg, spindleData]


    subs = pd.read_csv('/home/exx/Desktop/spindleOpt/spindles_id_pairs.csv')['ID_old'].to_list()        

    for sub in subs:
        print(sub)
        for ch in ['N2_C4-M1_', 'N2_C3-M2_']:
            print(ch)
            data =[f/(ch + sub + '_out.db') for f in files]

            chunks_lst = chunks(data, 15)
    ##    for f in files:
    ##        data = list(f.glob('*.db'))
           # output_path = f
            for lst in chunks_lst: 
                for c, n, o in zip (criteria,name,ouput_param):
                    if c:
                        cmd_list =  [('"' + destrat + '"' + ' "{}" +SPINDLES -r {} -p 5 >> "' + str(f.parent) + os.sep + 'N{}_{}.txt"').format(str(f),o,s,n) for f in lst]
                        procs_list = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) for cmd in cmd_list]
                        for proc in procs_list:
                            print(proc.communicate())

def chunks(l, n):
    n = max(1, n)
    return ([l[i:i+n] for i in range(0, len(l), n)])
##
##
##destrat(lunaBaseDir='/home/exx/luna-base/', swAvg=False, swData=False, swPhase=False,
##            spindleAvg=False, spindleData=True)
##                


from pathlib import Path
input_path = '/home/exx/Desktop/spindleOpt/results'
p=Path(input_path)

f1 = [x for x in p.iterdir() if x.is_dir() and len(list(x.glob('*.db')))>1]

dirs = [x for x in p.iterdir() if x.is_dir() and len(list(x.glob('*.db')))<1]
f2= []
for p in dirs:
    f2 +=[x for x in p.iterdir() if x.is_dir() and len(list(x.glob('*.db')))>1]
files = f1+f2

i=0
for f in files:
   # try:
    data = pd.read_csv(str(f) + '/' + 'N2_SpindleData.txt', sep='\t')
    data.to_csv(str(f) + '/' + 'N2_SpindleData.csv')
   # except:
    #    print(str(f) + '/' + 'N2_SpindleData.txt' + ': ERROR WITH PANDAS READ FILE FUNCTION')
    i+=1
    print(i)
    
