import os
import csv
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from pathlib import WindowsPath

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
    id_pairs = []

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
            filename = file.stem
        else:
            filename = file.stem[7:]

        sub_id_path = file.parent
        sub_id = sub_id_path.stem
        id_pairs.append((sub_id, filename))

    #remove duplicates   
    id_pairs_final = list(set(id_pairs))
    id_pairs_final.sort()
    
    return id_pairs_final

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

#prep id pairs for matching IDs later
edf_dir ='/media/mad3/Projects/BrainAge_Cognition_EEG_data'
id_pairs = sublist_filenames(edf_dir)
df = pd.DataFrame(id_pairs)
df.to_csv('spindles_id_pairs.csv', index=False, header=['ID', 'ID_old'])











