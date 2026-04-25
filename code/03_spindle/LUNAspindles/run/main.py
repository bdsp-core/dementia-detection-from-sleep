# -*- coding: utf-8 -*-
"""
LUNAspindles.run.main
---------------------
This module defines all luna specific functions to prepare files
and generate spindle features.
(c) 2019 by Noor Adra (CDAC Westover Lab, MGH)
license?
"""
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

from .. import tools as st
from LUNAspindles.signals import extract_signals as es


def run_luna(lunaBaseDir, LUNAsublist, ch_to_analyze, output_results_path,
             q=0, cycles=7, fc=13.5, th=4.5, minT=0.3, maxT=3, merge=0.5,
             N2=True, N3=True, sub_sublist=None, so=False):
    """
    Runs LUNA with given parameters and outputs sleep spindle featurs. 

    Parameters
    ---------- 
    lunaBaseDir : str
        dir pointing to luna-base and/or luna.exe
    LUNAsublist : str
        path to LUNA sublist file generated (LST file) 
    ch_to_analyze : list
        channels of interest for spindle analysis
    output_results_path : str
        results folder
    q : optional, '.1f' 
        quality metric; default=0; min=-9
    cycles : optional, int
        the higher the value, the higher the frequency-specificity; default=7
    fc : optional, int (1 fc) or str (multiple fc)
        target frequency (or frequencies) for the wavelet(s); default=13.5
        if >1 fc, separate str by commas spaces: '11,15'
    th : optional, '.1f'
        multiplicative threshold for core spindle detection; default=4.5
    minT : optional, '.1f'
        minimum duration for an entire spindle; default 0.5s
    maxT : optional, '.1f'
        maximum duration for an entire spindle; default 3s
    merge: optional, '.1f'
        interval within which two putative spindles are merged; default 0.5s
    N2 : Optional, bool
        if True, spindle analysis will be completed for Sleep Stage N2 
    N3 : Optional, bool
        if True, spindle analysis will be completed for Sleep Stage N3
    sub_sublist : optional, list.
        subset of subjects to be analyzed

    Returns
    -------
    none
    
    Notes
    ----- 
    outputs 5 txt files for each stage analyzed
        N{}_SpindleAvg.txt
        N{}_SpindleData.txt
        N{}_swData.txt
        N{}_swPhase.txt
        N{}_swAvg.txt
    """
  #define paths as str and make sure they are in correct format for os
    lunaBaseDir = st.str_to_Path(lunaBaseDir)
    LUNAsublist = st.str_to_Path(LUNAsublist)
    output_results_path = st.str_to_Path(output_results_path)

    #make sure output path exists
    st.output_folders(output_results_path)
         
    luna = str(lunaBaseDir / 'luna')
    destrat = str(lunaBaseDir / 'destrat')
    spindle_command = str(LUNAsublist.parent / 'spindleCommand.txt')
    LUNAsublist = str(LUNAsublist)
    output_results_path = str(output_results_path)
    
    #check for luna command file and create if necessary
    check_luna_cmd_file(q,cycles,fc,th, minT, maxT, merge, so)
##    to skip 1 or > EDFS you can use exclude. e.g. luna sublist.lst exclude=skip.txt -s ....
        
    stages=[]
    if N3:
        stages.append('3')
    if N2:
        stages.append('2')

    comb = []      
    for c in ch_to_analyze:
        for s in stages:
            comb.append((c,s))
    
    #gather data for each ch and stage and save in separate databases
    command = ('"' + luna + '"' + " " + '"' + LUNAsublist + '"' + " " + ' eeg="{}" stage=Sleep_stage_N{} -a "{}{}N{}_{}_out.db" < "{}"')

    if not sub_sublist:
        cmds_list= [command.format(c, s, output_results_path,os.sep, s,c[0:6],spindle_command) for c,s in comb]
        summary = [(s,'{}{}N{}_{}_out.db'.format(output_results_path, os.sep,s,c[0:6])) for c in ch_to_analyze for s in stages]
        print(cmds_list)
        
    else:
        cmds_temp = [('"' + luna + '"' + " " + '"' + LUNAsublist + '"  ' + sub +  " " +  'eeg="{}" stage=Sleep_stage_N{} -a "{}{}N{}_{}_' + sub + '_out.db" < "{}"')
                         for sub in sub_sublist]
        cmds_list= [cmd.format(c, s, output_results_path, os.sep, s,c[0:6], spindle_command) for cmd in cmds_temp for c,s in comb]

        summary = [(s,'{}{}N{}_{}_{}_out.db'.format(output_results_path, os.sep,s,c[0:6], sub)) for c in ch_to_analyze for s in stages for sub in sub_sublist]
        print(cmds_list)
        
    procs_list = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) for cmd in cmds_list]
    for proc in procs_list:
        print(proc.communicate())

    #concatanate database data into output text files by type
        
    ouput_param = ['CH F'] #['CH','CH N','CH PHASE','CH F SPINDLE','CH F']
    output = ['SpindleAvg'] #['swAvg', 'swData', 'swPhase', 'SpindleData', 'SpindleAvg']
    for param, out in zip(ouput_param, output):
        cmd_list =  [('"' + destrat + '"' + ' "{}" +SPINDLES -r {} -p 5 >> "' + output_results_path + os.sep + 'N{}_{}.txt"').format(f,param,s,out) for s,f in summary]
        print(cmd_list)
        procs_list = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) for cmd in cmd_list]
        for proc in procs_list:
            print(proc.communicate())
    
    
def luna_prep(edf_dir, output_path_edf, output_path_annot, input_sublist, output_LUNAsublist,
              channels_to_extract, channels_ref, ecg_corrected = False, ecg_label_lst = None,
              l_freq_eeg=0.3,h_freq_eeg=35, l_freq_ecg=0.3, h_freq_ecg=40,
              smooth=True, size=3, parallel=True, n_jobs=8):
    """
    Converts EDF+C/D & MAT files to EDF files and generates a LUNA-compatible annots.txt file for
    subjects in input_sublist, unless the files already exist. Saves/updates a LUNA-formatted sublist.

    Parameters
    ----------
    edf_dir : str
        dir with all EDF/MAT files to be analyzed
    output_path_edf : str
        dir where all output EDFs will be saved
    output_path_annot : str
        dir where all sub annot.txt files will be saved 
    input_sublist : list
        subject IDs (subIDs must be part of edf/signal filename)
    output_LUNAsublist : str
        path to LUNA sublist file generated (LST file) 
    channels_to_extract : list
        primary channels of interest from original signal data; depending on the original files,
        data might be pre-referenced (e.g., 'C3', 'C4') or post-reference (e.g, 'C3-M2', 'C4-M1')
    channels_ref : optional, list
        reference channels for channels of interest; order corresponds to items in channels_to_extract.
        if channels_ref < channels_to_extract, unmatched ch in channels_to_extract will be used  
    ecg_corrected : optional, bool
        if True, performs ecg correction on signals from selected channels
    ecg_label_lst : list
        ECG channel names; labels correspond to signals in ecg_sig_lst; ignored if `ecg_corrected` is False
    l_freq_eeg : optional, .2f
        lower frequency cut-off for filtering EEG signals
    h_freq_eeg : optional, int
        upper frequency cut-off for filtering EEG signals
    l_freq_ecg : optional, .2f
        lower frequency cut-off for filtering ECG signals
    h_freq_ecg : optional, int
        upper frequency cut-off for filtering ECG signals
    smooth : optional, bool
        if True, performs smoothing on the resulting heart rate
    size : optional, int
        size of smoothing window; ignored if `smooth` is False
    parallel : optional, bool
        if True, writes parallel for loops using multiprocessing
    n_jobs : optional, int
        number of CPU used for multiprocessing; ignored if `parallel` is False
    
    Returns
    -------
    none 
    
    Notes
    ----- 
    writes out an adjusted edf in output_path_edf, new annot.txt file for subject in out_annot dir, &
    LUNAsublist 
    """
    #make sure all paths are in correct OS format 
    edf_dir = st.str_to_Path(edf_dir)
    output_path_edf = st.str_to_Path(output_path_edf)
    output_path_annot = st.str_to_Path(output_path_annot)
    output_LUNAsublist = st.str_to_Path(output_LUNAsublist)

    #check if output files exist, if not create them
    st.output_folders(output_path_edf, output_path_annot)
    
    max_subID = len(max(input_sublist, key=len))

    #find all signal files (edf or mat)
    edf_files = list(edf_dir.glob('**/*.edf'))
    mat_files = list(edf_dir.glob('**/*.mat'))
    mat_sig_files = [file for file in mat_files if file.name.startswith('Signal')]
    csv_files = list(edf_dir.glob('**/*.csv'))
    annot_file = [file for file in csv_files if 'annotations' in file.name]
    
    #loop through each item and generate EDFs and ANNOTs if needed
    if not edf_files and not mat_sig_files:
        print('No edf or Signals mat file found')
        return

    for file in edf_files+mat_sig_files:
        #find edf filename and path
        edf =  file.name
        edf_path = file

        #find sub id and annot file
        if file.suffix == '.edf':
            sub_id = file.stem
        else:
            sub_id = file.stem[7:]

        #find annot path
        annot1 = 'annotations.csv'
        annot2 = sub_id +'_annotations.csv'
        annot3 = 'Labels_' + sub_id +'.mat'
        
        if (file.parent / annot1).exists():
            annot_path = file.parent / annot1
        elif (file.parent / annot2).exists():
            annot_path = file.parent / annot2
        elif (file.parent / annot3).exists():
            annot_path = file.parent / annot3

        #reformat annot file if necessary
        if annot_path.suffix == '.csv':
            df = pd.read_csv(annot_path)
            if not df[df['event'].str.contains("Stage - ")==True].empty:
                new_annot = file.parent / 'annotations_reformat.csv'
                st.grass_to_natus(annot_path, new_annot)
                annot_path = new_annot
    
        if sub_id in input_sublist:
            print(sub_id)

            #edf output filename
            sub_edf = '{}_adj.edf'.format(sub_id)
            out_edf = output_path_edf / sub_edf #output file

            #create edf if it doesn't exist
            if sub_edf not in os.listdir(output_path_edf):
                print(' EDF conversion in progress...')

                try:
                    tupleData = es.extract_signals(str(edf_path), channels_to_extract, channels_ref, ecg_corrected,
                                                   ecg_label_lst, l_freq_eeg=l_freq_eeg, h_freq_eeg=h_freq_eeg,
                                                   l_freq_ecg=l_freq_ecg, h_freq_ecg=h_freq_ecg, smooth=smooth,
                                                   size=size, parallel=parallel, n_jobs=n_jobs)
                    sig = tupleData[0]
                    Fs = tupleData[1]
                    channel_info = tupleData[3]
                    print(' tupleData generated')
                    
                    #write out edf
                    st.create_edfp(str(out_edf), sig, channel_info, Fs)
                    print(' EDF converted for ', sub_id)

                except:
                    er = '** Error during signal extraction** {}. {}, line: {}'.format(sys.exc_info()[0],
                                                      sys.exc_info()[1],
                                                      sys.exc_info()[2].tb_lineno)

                    with open ('errorLog.txt', 'a+') as txt:
                      space = (max_subID - len(sub_id))*' '
                      txt.write('\n'+ str(sub_id) + ':  ' + space + er)

                    if "'ECG-LA' is not in list" in sys.exc_info()[1].args[0]:
                        try:
                            tupleData = es.extract_signals(str(edf_path), channels_to_extract, channels_ref,
                                                        ecg_corrected, ecg_label_lst=['EKG'],
                                                        l_freq_eeg=l_freq_eeg, h_freq_eeg=h_freq_eeg,
                                                        l_freq_ecg=l_freq_ecg, h_freq_ecg=h_freq_ecg,
                                                        smooth=smooth, size=size,
                                                        parallel=parallel, n_jobs=n_jobs)
                            sig = tupleData[0]
                            Fs = tupleData[1]
                            channel_info = tupleData[3]
                            print(' tupleData generated')
                    
                            #write out edf
                            st.create_edfp(str(out_edf), sig, channel_info, Fs)
                            print(' EDF converted for ', sub_id)

                        except:
                            er = '** Error during signal extraction** {}. {}, line: {}'.format(sys.exc_info()[0],
                                                          sys.exc_info()[1],
                                                          sys.exc_info()[2].tb_lineno)

                            with open ('errorLog.txt', 'a+') as txt:

                              space = (max_subID - len(sub_id))*' '
                              txt.write('\n'+ str(sub_id) + ':  ' + space + er)

                    if "'EKG' is not in list" in sys.exc_info()[1].args[0]:
                        try:
                            tupleData = es.extract_signals(str(edf_path), channels_to_extract, channels_ref,
                                                        ecg_corrected, ecg_label_lst=['ECG-LA', 'ECG-V1'],
                                                        l_freq_eeg=l_freq_eeg, h_freq_eeg=h_freq_eeg,
                                                        l_freq_ecg=l_freq_ecg, h_freq_ecg=h_freq_ecg,
                                                        smooth=smooth, size=size,
                                                        parallel=parallel, n_jobs=n_jobs)
                            sig = tupleData[0]
                            Fs = tupleData[1]
                            channel_info = tupleData[3]
                            print(' tupleData generated')
                    
                            #write out edf
                            st.create_edfp(str(out_edf), sig, channel_info, Fs)
                            print(' EDF converted for ', sub_id)

                        except:
                            er = '** Error during signal extraction** {}. {}, line: {}'.format(
                                                                                          sys.exc_info()[0],
                                                                                          sys.exc_info()[1],
                                                                                          sys.exc_info()[2].tb_lineno)

                            with open ('errorLog.txt', 'a+') as txt:

                              space = (max_subID - len(sub_id))*' '
                              txt.write('\n'+ str(sub_id) + ':  ' + space + er)
                    
            else:
                print(' EDF already exists in path')

            #annot output filename
            sub_annot = '{}_annot.txt'.format(sub_id)
            out_annot = output_path_annot / sub_annot #output file

            #create annot if it doesn't exist
            if sub_annot not in os.listdir(output_path_annot):
              print('   Annotation file currently being generated...')
              
              try:
                luna_annots(str(annot_path), str(out_edf), str(out_annot), sub_id)
                print(' Annotations generated for ', sub_id, '\n')

              except:
                er = '**Error during annot generation** {}. {}, line: {}'.format(sys.exc_info()[0],
                                                      sys.exc_info()[1],
                                                      sys.exc_info()[2].tb_lineno)
                with open ('errorLog.txt', 'a+') as txt:
                  space = (max_subID - len(sub_id))*' '
                  txt.write('\n'+ str(sub_id) + ':  ' + space  + er)
                  
            else:
              print('   ANNOT already exists in path\n')

            #create subList.lst file 
            with open (str(output_LUNAsublist), 'a+') as f:
              f.seek(0)
              content = f.read()
              if sub_id not in content:
                f.write(sub_id + '\t' + str(out_edf) + '\t' + str(out_annot) + '\n')
    return


def luna_annots(input_path, edf_path, out_annot, sub_id):
    """
    Truncates original annotation files to only include sleep stage
    annotations & adjusts for missing epoch annotations. 

    Number of epochs in the output must align with duration of sleep study.
    Missing epochs will be appended to the beginning of the output file
    by subtracting 30s per epoch from the first timestamp and labeling it
    as Sleep_stage_W

    Parameters
    ----------
    input_path : str
        path to annotations.csv or Labels.mat file
    edf_path : str
        path to edf (original or adjusted)
    out_annot : str
        dir where annot.txt file will be saved
    sub_id : str
        subject ID corresponding to annotations file
    
    Returns
    -------
    none 
    
    Notes
    ----- 
    Writes out an annot.txt file for subject in out_annot dir
    """
    #read from original annot file and collect sleep stage annots
    sleep_stages = []
    data = pyedflib.EdfReader(edf_path)
    Fs = data.samplefrequency(0)
    sigs_len = data.getNSamples()[0]
    unit = int(Fs*30) #30 is epoch length
    n_epochs = int(sigs_len/unit)
  
    if '.csv' in input_path:

        with open(input_path) as csv_file:   
            csv_reader = csv.reader(csv_file, delimiter= ',')

            for row in csv_reader:
                if "Sleep_stage" in row[-1]:
                    sleep_stages.append(row)

    elif '.mat' in input_path:
        ss_key = {5: 'Sleep_stage_W', 4:'Sleep_stage_R',
                  3: 'Sleep_stage_N1', 2: 'Sleep_stage_N2', 1:'Sleep_stage_N3'}

        try:
          ffl = sio.loadmat(input_path)
        except:
          ffl = h5py.File(input_path, 'r')

        sleep_stages_raw = ffl['stage'][()].flatten()
        
        for i in range(n_epochs):
            step = int(unit*i)
            section = sleep_stages_raw[0+step:unit+step]
            try:
                label = mode(section)
                sleep_stages.append([str(i+1), ss_key[label]])
            except:
                sleep_stages.append([str(i+1), 'Sleep_stage_?'])

    #create new annot file (txt file tab delimited)
    with open(out_annot, 'w+') as txt:

        #to follow .annot (LUNA) format, add header for each annot class (sleep stage)
        txt.write('#' + '\t' + 'Sleep_stage_W\n') 
        txt.write('#' + '\t' + 'Sleep_stage_R\n')
        txt.write('#' + '\t' + 'Sleep_stage_N1\n')
        txt.write('#' + '\t' + 'Sleep_stage_N2\n')
        txt.write('#' + '\t' + 'Sleep_stage_N3\n')
        txt.write('#' + '\t' + 'Sleep_stage_?\n')
        

        if '.csv' in input_path:

          #find number of missing rows
          missing = int(n_epochs - len(sleep_stages)) - 1
          
          epoch_old = int(float(sleep_stages[0][0])) #first epoch num in sleep_stages list
          event = 'Sleep_stage_W'             #event names for new epochs

          missing_ch = missing                #missing_ch will change as we fill in epochs

          if epoch_old != 1:
              #add missing epochs to txt file first
              for i in range(missing):

                epoch_adj = epoch_old - missing_ch # new epoch num

                if epoch_adj>0: 
                    txt.write('Sleep_stage_W\t' + str(epoch_adj) + '\te:'  + str(epoch_adj) + '\n')
                    missing_ch = missing_ch - 1 #new num of missing epochs
                else:
                    txt.write('Sleep_stage_W\t' + '1' + '\te:'  + str(epoch_adj) + '\n')
                    missing_ch = missing_ch - 1 #new num of missing epochs

        #add all epochs from sleep_stages list to txt file 
        for epoch in sleep_stages:
          txt.write(epoch[-1] + '\t' + epoch[0] + '\te:' + epoch[0] + '\n')

    return 



def make_luna_cmd_file(q=0,cycles=7,fc=13.5, th=4.5, minT=0.3, maxT=3, merge=0.5, so=False):
    """
    Creates spindleCommand.txt using given paramters. 

    Parameters
    ----------
    q : optional, '.1f'
        quality metric; default=0; min=-9
    cycles : optional, int
        the higher the value, the higher the frequency-specificity; default=7
    fc : optional, int (1 fc) or str (multiple fc)
        target frequency (or frequencies) for the wavelet(s); default=13.5
        if >1 fc, separate str by commas spaces: '11,15'
    th : optional, '.1f'
        multiplicative threshold for core spindle detection; default=4.5
    minT : optional, '.1f'
        minimum duration for an entire spindle; default 0.5s
    maxT : optional, '.1f'
        maximum duration for an entire spindle; default 3s
    merge: optional, '.1f'
        interval within which two putative spindles are merged; default 0.5s
    so: optional, bool
        include or exclude spindle coupling analysis
        
    Returns
    -------
    none 
    
    Notes
    -----
    Writes out spindles command file to directory ('spindleCommand.txt')  

    Command includes:
    filtering bandpass=0.1,20 ripple=0.02,
    artifact masking
    spindle parameters defined by user (q,cycles,fc)
    Spindle paramters other: so; f-lwr=0.5; f-upr=4;  mag=1;
    smoothing window = 0.1s
    detection thresholds (core,flank,max) = 4.5, 2x; 
    core duration thresholds (core, min, max) = 0.3, 0.5, 3s
    filtering at 11.5 to 15.5.                   

    for detailed methods visit <http://zzz.bwh.harvard.edu/luna/ref/spindles-so/>
    """

    #split script into phrases
    if so:
        
        c = ('''EPOCH MASK all MASK unmask-if=${{stage}} RESTRUCTURE FILTER sig=${{eeg}}
                bandpass=0.1,20 ripple=0.02 tw=1 ARTIFACTS sig=${{eeg}} mask SIGSTATS epoch
                sig=${{eeg}} mask threshold=3,3,3 RESTRUCTURE SPINDLES q={} sig=${{eeg}}
                method=wavelet cycles={} fc={} ftr-dir=fcAnnotsOut/
                so list-all-spindles th={} min={} max={} collate merge={} f-lwr=0.5 f-upr=4
                mag=1 RESTRUCTURE''').format(q,cycles,fc,th,minT,maxT,merge).split()
    else:
        c = ('''EPOCH MASK all MASK unmask-if=${{stage}} RESTRUCTURE FILTER sig=${{eeg}}
                bandpass=0.1,20 ripple=0.02 tw=1 ARTIFACTS sig=${{eeg}} mask SIGSTATS epoch
                sig=${{eeg}} mask threshold=3,3,3 RESTRUCTURE SPINDLES q={} sig=${{eeg}}
                method=wavelet cycles={} fc={} ftr-dir=fcAnnotsOut/ th={} min={} max={}
                collate merge={} f-lwr=0.5 f-upr=4 mag=1
                RESTRUCTURE''').format(q,cycles,fc,th,minT,maxT,merge).split()
    
    with open('spindleCommand.txt', 'w+') as txt:
        for phrase in c:

            #if the word is in all caps, then start on a new line
            if phrase == phrase.upper() and phrase != c[0]:
                txt.write('\n' + phrase + '\t')
                
            #if the word is not in all caps, continue line
            else:
                txt.write(phrase + '\t')



def check_luna_cmd_file(q=0,cycles=7,fc=13.5, th=4.5, minT=0.3, maxT=3, merge=0.5, so=False):
    """
    Checks if spindleCommand.txt exists and, if so, whether old & input paramters match;
    If file does not exist, calls make_luna_cmd_file.

    Parameters
    ----------
    q : optional, '.1f' 
        quality metric; default=0; min=-9
    cycles : optional, int
        the higher the value, the higher the frequency-specificity; default=7
    fc : optional, int (1 fc) or str (multiple fc)
        target frequency (or frequencies) for the wavelet(s); default=13.5
        if >1 fc, separate str by commas spaces: '11,15'
    th : optional, '.1f'
        multiplicative threshold for core spindle detection; default=4.5
    minT : optional, '.1f'
        minimum duration for an entire spindle; default 0.5s
    maxT : optional, '.1f'
        maximum duration for an entire spindle; default 3s
    merge: optional, '.1f'
        interval within which two putative spindles are merged; default 0.5s
        
    Returns
    -------
    none 
    
    Notes
    ----- 
    may directly alter 'spindleCommand.txt' if parameters conflict
    """
    #make the command script if it doesn't exist in the dir
    if not os.path.isfile('spindleCommand.txt'):
        print('1')
        make_luna_cmd_file(q,cycles,fc,th, minT, maxT, merge, so)

    #open the command script if it does exist 
    else:
        print('2')
        with open('spindleCommand.txt', 'r+') as f:
            param = f.read()

            #check if parameters of old file == input parameters
            if ('q={}\t'.format(q) in param
                and 'cycles={}\t'.format(cycles) in param
                and 'fc={}\t'.format(fc)  in param
                and 'th={}\t'.format(th) in param
                and 'min={}\t'.format(minT) in param
                and 'max={}\t'.format(maxT)  in param
                and 'merge={}\t'.format(merge)  in param):

                print('''spindleCommand.txt already exits and has same parameters desired,
                        using old spindleCommand.txt''')

            #if not, change parameters in old txt file 
            else:
                i_q = param[param.find('q='):param.find('\t', param.find('q='))]
                i_c = param[param.find('cycles='):param.find('\t', param.find('cycles='))]
                i_f = param[param.find('fc='):param.find('\t', param.find('fc='))]
                i_th = param[param.find('th='):param.find('\t', param.find('th='))]
                i_minT = param[param.find('min='):param.find('\t', param.find('min='))]
                i_maxT = param[param.find('max='):param.find('\t', param.find('max='))]
                i_merge = param[param.find('merge='):param.find('\t', param.find('merge='))]


                param = param.replace(i_q,'q={}'.format(q))
                param = param.replace(i_c,'cycles={}'.format(cycles))
                param = param.replace(i_f,'fc={}'.format(fc))
                param = param.replace(i_th,'th={}'.format(th))
                param = param.replace(i_minT,'min={}'.format(minT))
                param = param.replace(i_maxT,'max={}'.format(maxT))
                param = param.replace(i_merge,'merge={}'.format(merge))

                f.truncate(0)
                f.write(param)
                print('''spindleCommand.txt already exits.
                         Parameters in file were changed to match desired parameters''')



