# -*- coding: utf-8 -*-
"""
LUNAspindles.tools
------------------
This module defines helpful tools for reading and writing from/to dirs.
(c) 2019 by Noor Adra (CDAC Westover Lab, MGH)
license?
"""

from pathlib import Path
from pathlib import WindowsPath
import pandas as pd
import pyedflib
import sys
import os

def twin_sublist(sub_csv, colname_id):
    """
    For TwinData only. Reads colname_id from csv file and reformats names
    to match filenames. 

    Parameters
    ----------
    sub_csv : str
        csv file with subject IDs 
    colname_id : str
        column name for column with subject IDs in csv
    
    Returns
    -------
    sl : list
        properly formatted TwinData subIDs 

    Notes
    ----- 
    For TwinData only. Filenames include commas, which may not be included in
    subID. This functions makes sure subIDs match filenames. 
    """
    #read subID column
    df = pd.read_csv(sub_csv)
    subID = df[colname_id]

    #keep name if num < 1000. Else add comma so subIDs match filenames
    sl = [s.split('_')[0]+'_'+ s.split('_')[-1][0] + ',' + s.split('_')[-1][1:]
                if len(s.split('_')[-1]) > 3 else s for s in subID ]
    
    return sl



def sublist_folder_names(input_path):
    """
    If a folder has EDF/MAT signal data, the folder name will be used as subID.

    Parameters
    ----------
    input_path : str
        dir with subject folders, each with its own EDF/MAT file
    
    Returns
    -------
    sublist : list
        properly formatted subIDs 

    Notes
    ----- 
    Assumes each EDF/MAT file located in an individual subject folder
    """
    sublist = []
    for path,dirs,files in os.walk(input_path):
        sub_id = path.split(os.sep)[-1] #find subject id
        sublist.append(sub_id)
        
    return sublist


def sublist_filenames(input_path):
    """
    Locates EDF/MAT files in input_path. Extracts subID from filename.
    If EDF --> subID = filename without .edf extension
    If MAT --> subID = filename without .mat extension and after 'Signal\_'
    
    Parameters
    ----------
    input_path : str
        dir with EDF/MAT files
    
    Returns
    -------
    sublist : list
        properly formatted subIDs 

    Notes
    ----- 

    """
    sublist = []
    
    #find all signal files (edf or mat)
    p = str_to_Path(input_path)
    edf_files = list(p.glob('**/*.edf'))
    mat_files = list(p.glob('**/*.mat'))
    mat_sig_files = [file for file in mat_files if file.name.startswith('Signal')]

    if not edf_files and not mat_sig_files:
        print('No edf or Signals mat file found')
        return

    #find sub id
    for file in edf_files+mat_sig_files:

        if file.suffix == '.edf':
            sub_id = file.stem
        else:
            sub_id = file.stem[7:]
        sublist.append(sub_id)

    #remove duplicates   
    sublist = list(set(sublist))
    
    return sublist

def output_folders(*folders):
    """
    Creates output folders if they dont exist

    Parameters
    ----------
    *folders : str
        any number of folders

    Returns
    -------
    none 

    Notes
    -----
    """
    for folder in folders:
        try:
            f = str(folder)
            os.makedirs(f)
            print(f + ': path created')
        except:
            print(f + ': path already exists')
            pass


def create_edfp(output_path,sigs, channel_info, Fs):
    """
    Exports signal data to EDF+ file

    Parameters
    ----------
    output_path : str
        dir where EDF+ file will be written 
    sigs : array
        signals written in EDF+ file
    channel_info : list
        channel_info dictionaries for each signal in sigs
    Fs : int
        sampling rate (Hz)
    
    Returns
    -------
    none 

    Notes
    -----
    writes EDF+ file to output_path
    """
    with pyedflib.EdfWriter(output_path, len(sigs), file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setSignalHeaders(channel_info)
        f.writeSamples(sigs)


def txt_to_csv(output_results_path, N2=True, N3=True):
    """
    Converst spindle output data from txt to csv format

    Parameters
    ----------
    N2 : optional, bool
        if True, assumes spindle analysis was performed for Sleep Stage N2
    N3 : optional, bool
        if True, assumes spindle analysis was performed for Sleep Stage N3
    
    Returns
    -------
    none  

    Notes
    -----
    Creates csv files in output results folder
    """
    stages=[]
    
    if N3:
        stages.append('3')
    if N2:
        stages.append('2')
        
    for s in stages:
        for f in ['N{}_SpindleData',
                  'N{}_SpindleAvg',
                  'N{}_swAvg',
                  'N{}_swData',
                  'N{}_swPhase']:
            try:
                data = pd.read_csv(output_results_path + '/' + f.format(s)+'.txt', sep='\t')
                data.to_csv(output_results_path + '/' + f.format(s)+'.csv')
            except:
                print(f.format(s) +'.txt' + ': ERROR WITH PANDAS READ FILE FUNCTION')


def str_to_Path(string):
    """
    Converts a str with a path to a Windows or non-Windows Path object

    Parameters
    ----------
    string : str
        Windows, LINUX, or MAC OS path 

    Returns
    -------
    Path(string) : Path
        Windows, LINUX, or MAC OS path as a Path object

    Notes
    -----
    """
    if "C:\\" in string or "Z:\\" in string:
        return WindowsPath(string)
    else:
        return Path(string)
    

def save_error(error_info, sub_id):
    er = '{}: ** Error during signal extraction** {}. {}, line: {}'.format(sub_id,
                                                                           error_info[0],
                                                                           error_info[1],
                                                                           error_info[2].tb_lineno)
    return er
    
def protect_path(input_path):
    """
    Adds quotes around each object in input_path to avoid errors in the case of white space

    Parameters
    ----------
    input_path : str or Path object
        Windows, LINUX, or MAC OS path

    Returns
    -------
    new_path : str
        old path with "" around each object in path

    Notes
    -----
    """
    path = str(input_path)
    
    if "/" in path:
        sections = ['\"' + item + '\"' if len(item)>0 and ':' not in item else item for item in path.split('/')]
        new_path = '/'.join(sections)
    elif "\\" in path:
        sections = ['\"' + item + '\"' if len(item)>0 and ':' not in item else item for item in path.split('\\')]
        new_path = '\\'.join(sections)
    
    return new_path

def grass_to_natus(input_path, output_path):
    """
    reformats annotation file in grass format to natus format;

    Parameters
    ---------
    input_path : file,
            old annots note transitions in sleep stages

    Returns
    -------
    None

    Notes
    -----
    saves output_path csv file; each epoch labeled with a sleep stage
    """
    df = pd.read_csv(input_path)
    sleep_df = df[df['event'].str.contains("Stage")==True]
    sleep_df = sleep_df.replace('Stage - W', 'Sleep_stage_W', regex=True)
    sleep_df = sleep_df.replace('Stage - N1', 'Sleep_stage_N1', regex=True)
    sleep_df = sleep_df.replace('Stage - N2', 'Sleep_stage_N2', regex=True)
    sleep_df = sleep_df.replace('Stage - N3', 'Sleep_stage_N3', regex=True)
    sleep_df = sleep_df.replace('Stage - No Stage', 'Sleep_stage_?', regex=True)
    sleep_df = sleep_df.replace('Stage - R', 'Sleep_stage_R', regex=True)
    epoch_list = sleep_df['epoch']
    event_list = sleep_df['event']

    sleep_stages = pd.DataFrame( {'epoch': [], 'event' : []} )

    for i,event in enumerate(event_list):
        if 'stage' in event:
            e = int(epoch_list.iloc[i])
            if e != int(epoch_list.iloc[-1]):
                while e < epoch_list.iloc[i+1]:
                    sleep_stages = sleep_stages.append({'epoch': e, 'event':event}, ignore_index=True) 
                    e += 1
            else:
                sleep_stages = sleep_stages.append({'epoch': e, 'event':event}, ignore_index=True) 

    sleep_stages.to_csv(output_path, index=False, index_label=False)

