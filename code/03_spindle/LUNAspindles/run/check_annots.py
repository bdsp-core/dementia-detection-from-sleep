# -*- coding: utf-8 -*-
"""
LUNAspindles.run.check_annots
-----------------------------
This module evaluates annotations and fixes issues with stage names.
(c) 2019 by Noor Adra (CDAC Westover Lab, MGH)
license?
"""

import pandas as pd
import re

def fix_stage_names(df,
                    N1 = 'Sleep_stage_N1',
                    N2 = 'Sleep_stage_N2',
                    N3 = 'Sleep_stage_N3',
                    N1wrong = ['Sleep_stage_1'],
                    N2wrong = ['Sleep_stage_2'],
                    N3wrong = ['Sleep_stage_3']):
    """
    Replaces incorrect sleep stage terminology with correct stage names.
    Directly alters annot.txt files.
    Assumes sleep stage names should follow this pattern: Sleep_stage_N#

    Parameters
    ----------
    df : pandas DataFrame
        Column names must include 'fixed_staging' or 'check_stage_names';
        generated using check_annots; must include annot.txt file locations
    N1 : optional, string
        desired N1 stage name
    N2 : optional, string
        desired N2 stage name
    N3 : optional, string
        desired N3 stage name
    N1wrong : optional, list
        incorrect N1 stage names seen across various annot files; min len == 1
    N2wrong : optional, list
        incorrect N2 stage names seen across various annot files; min len == 1 
    N3wrong : optional, list
        incorrect N3 stage names seen across various annot files; min len == 1
        
    Returns
    -------
    none 
    
    Notes
    -----
    Alters annot.txt files directly
    
    Assumes that incorrect staging terminology is 'Sleep_stage_#'
    Automatically tries to fix mistakes to 'Sleep_stage_N#'
    """
    #If default parameters, automatic staging correction
    if len(N1wrong) == 1 and N1wrong[0] == 'Sleep_stage_1':
        fix = df.loc[df['fixed_staging'] == True]

    #If non-default parameters are present, cases must have been manaully checked
    else:
        fix = df.loc[df['check_stage_names'] == True]

    for file in fix.annot_file:

        with open(file, 'r') as f:
            data = f.read()

        for i in range(len(N1wrong)):
            if N1wrong[i] in data:
                data = data.replace( N1wrong[i], N1 )
                data = data.replace( N2wrong[i], N2 )
                data = data.replace( N3wrong[i], N3 )
                break
            
        with open(file, 'w') as f:
          f.write(data)

          
def check_annots(LUNAsublist,
                 N2='Sleep_stage_N2',
                 N3 = 'Sleep_stage_N3',
                 UNK='Sleep_stage_?'):
    """
    Evaluate each annot.txt file found in LUNAsublist for presence of N2 and N3,
    incorrect staging terminology, and lack of staging.
    Assumes sleep stage names should follow this pattern: Sleep_stage_#

    Parameters
    ----------
    LUNAsublist : str
        path to LUNAsublist file
    N2 : optional, string
        desired N2 stage name
    N3 : optional, string
        desired N3 stage name
    UNK : optional, string
        label used for epochs with unknown stage labels; if NA, set to None
        
    Returns
    -------
    df : pandas DataFrame with the following columns:
        subID,
        N2_persent,
        N3_present,
        fixed_staging,
        needs_staging,
        check_stage_names,
        annot_file
    
    Notes
    -----
    Outputs 'stagingInfo.csv' to current dir
    
    Assumes that incorrect staging terminology is 'Sleep_stage_#'
    Automatically tries to fix it to 'Sleep_stage_N#'
    """
    #find subIDs and annot paths data from LUNAsublist
    with open(LUNAsublist, 'r') as sl:
        data = sl.read() 
        data_list = re.split('\n|\t',data)
        data_list.remove('')
        subID = data_list[0::3]
        annots = data_list[2::3]

    #define incorrect/missing staging 
    stage_unk = UNK

    #if default parameters, incorrect stage names will be missing an 'N'
    if N2 == 'Sleep_stage_N2':
        wrong_N1 = 'Sleep_stage_1'
        wrong_N2 = 'Sleep_stage_2'
        wrong_N3 = 'Sleep_stage_3'

    #if non-default parameters, label default parameters as wrong
    else:
        wrong_N1 = 'Sleep_stage_N1'
        wrong_N2 = 'Sleep_stage_N2'
        wrong_N3 = 'Sleep_stage_N3'
        
    #pre-set variables 
    N2_present = [True]*len(subID)
    N3_present = [True]*len(subID)
    wrong_staging = [False]*len(subID)
    needs_staging = [False]*len(subID)
    check_stage_names = [False]*len(subID)
        
    for i in range(len(subID)):

        #read annot data
        with open(annots[i], 'r') as f:
            stages = f.read().split()[12::3]
            
            if N2 not in stages:
                N2_present[i] = False

                if wrong_N2 in stages or wrong_N1 in stages:
                    wrong_staging[i] = True
                    
            if N3 not in stages:
                N3_present[i] = False
                
            if stages.count(stage_unk) > len(stages)/4: #what if UNK = None?
                needs_staging[i] = True
                
            elif not any(x in stages for x in [wrong_N1, wrong_N2, wrong_N3,
                                               N2, N3]):
                check_stage_names[i] = True
                
    df = pd.DataFrame( {'subID': subID,
                       'N2_present' : N2_present,
                       'N3_present' : N3_present,
                       'fixed_staging' : wrong_staging,
                       'needs_staging' : needs_staging,
                        'check_stage_names' : check_stage_names,
                       'annot_file' : annots},
                       index = list(range(1,len(subID)+1)))

    df.drop_duplicates()
    fix_stage_names(df)
    df.to_csv('stagingInfo.csv')
    
    return df
