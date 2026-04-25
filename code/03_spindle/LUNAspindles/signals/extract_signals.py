# -*- coding: utf-8 -*-
"""
LUNAspindles.signals.extract_signals
------------------------------------
This module pre-processes and extracts selected biosignals
from EDF+(C/D) filesor MAT files.

(c) 2019 by Noor Adra (CDAC Westover Lab, MGH)
license?
"""

from scipy.stats import linregress, skew
from joblib import Parallel, delayed
from collections import namedtuple
import numpy as np
import pandas as pd
import mne

from . import read_signals as rs
from . import ecg_correction 

def filter(signal, Fs, l_freq=0.3, h_freq=35):
    """
    Filters signal using given low freq and high freq parameters.

    Parameters
    ----------
    signal : list or array
        signals from one channel
    Fs : int
        sampling rate (Hz)
    l_freq : optional, .2f
        lower frequency cut-off for filtering
    h_freq : optional, int
        upper frequency cut-off for filtering

    Returns
    -------
    filtered : array
        filtered signal
    
    Notes
    ----- 

    """
    #make sure signal is in the correct format
    signal = np.array(signal)
    signal = signal.astype(np.float64)
    #filter
    n = signal.ndim
    filtered = mne.filter.filter_data(signal, Fs, l_freq=l_freq,
                                      h_freq=h_freq, verbose=False, n_jobs=n)

    return filtered


def check_skew(signal):
    """
    Checks if the signal is upside down and corrects the signal when necessary.

    Parameters
    ----------
    signal : list or array
        signals from one channel
    
    Returns
    -------
    signal : list or array
        Adjusted signal if original was skewed. Else, returns original data
    
    Notes
    ----- 

    """
    if skew(signal)<0:
        try:
            signal = -signal
        except:
            signal = [np.negative(x) for x in signal]
        #print('skew')
    #else:
        #print('no skew')
            
    return signal


def preprocessing(signal, Fs, l_freq=0.3, h_freq=35):
    """
    Minimal signal pre-processing of a signal (check skewness & filter).

    Parameters
    ----------
    signal : list or array
        signals from one channel
    Fs : int
        sampling rate (Hz)
    l_freq : optional, .2f
        lower frequency cut-off for filtering
    h_freq : optional, int
        upper frequency cut-off for filtering

    Returns
    -------
    signal : array
        filtered signal, adjusted for any skewness 
    
    Notes
    ----- 

    """
    # be careful! sometimes the signals are upside down!
    signal = check_skew(signal)

    # filter EEG based on paper
    signal2 = filter(signal, Fs, l_freq, h_freq)

    return signal2



def preprocessing_ecg(eeg_sigs,ecg_sig_lst, ecg_label_lst, Fs, smooth=True,
                      size=3, l_freq_eeg=0.3, h_freq_eeg=35, l_freq_ecg=0.3,
                      h_freq_ecg=40, parallel=True, n_jobs=8):
    """
    Thorough EEG signal pre-processing (check skewness, filter, ecg correction).

    Parameters
    ----------
    eeg_sigs : list
        EEG signal arrays from 1 or more channels; shape = (#channel, T)
    ecg_sig_lst : list
        ECG signal arrays from 1 or more channels
    ecg_label_lst : list
        ECG channel names; labels corresponds to signals in ecg_sig_lst
    Fs : int
        sampling rate (Hz)
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
    eeg_corrected : array
        pre-processed & ecg-corrected EEG signals. shape = (#channel, T) 
    
    Notes
    ----- 

    """
    #check skewness and filter each EEG signal 
    eeg_sigs2 = []
    
    for eeg_sig in eeg_sigs:
        eeg = preprocessing(eeg_sig, Fs, l_freq_eeg, h_freq_eeg)
        eeg_sigs2.append(eeg)

    #check skewness and filter each ECG signal 
    ecg_sig_lst2 = []
    
    for ecg_sig in ecg_sig_lst:
        ecg = preprocessing(ecg_sig, Fs, l_freq_ecg, h_freq_ecg)
        ecg_sig_lst2.append(ecg)
        
    #ecg correction
    eeg_corrected = ecg_correction.ecg_correction(eeg_sigs2, ecg_sig_lst2, ecg_label_lst,
                                   Fs, return_rpeaks=False, smooth=smooth,
                                   size=size, parallel=parallel, n_jobs=n_jobs)

    return eeg_corrected


def extract_signals(input_path, channels_to_extract, channels_ref= None, ecg_corrected = False, ecg_label_lst = None,
                    l_freq_eeg=0.3, h_freq_eeg=35, l_freq_ecg=0.3, h_freq_ecg=40, smooth=True, size=3,
                    parallel=True, n_jobs=8):
    """
    Convert signal data to EDF+C format, extract signals from selected channels using the corresponding
    reference channel (if applicable), and preprocess new signals using given paramters. 

    Parameters
    ----------
    input_path : str
        path to signal data in EDF, EDF+C/D, or MAT format
    channels_to_extract : list
        primary channels of interest from original signal data; depending on the original files,
        data might be pre-referenced (e.g., 'C3', 'C4') or post-reference (e.g, 'C3-M2', 'C4-M1')
    channels_ref : optional, list
        reference channels for channels of interest; order corresponds to items in channels_to_extract.
        if channels_ref < channels_to_extract, unmatched ch in channels_to_extract will be used  
    ecg_corrected : optional, bool
        if True, performs ecg correction on signals from selected channels
    ecg_label_lst : optional, list
        ECG channel names; ignored if `ecg_corrected` is False
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
    Named tuple with the fields:
      X : NumPy array with shape p by n.
        Raw recording of n samples in p dimensions.
      sample_rate : float
        The sample rate of the recording. Note that mixed sample-rates are not
        supported.
      chan_lab : list of length p with strings
        The labels of the sensors used to record X.
      channel_info : list of dictionaries with info for each channel:
                    -label
                    -dimension
                    -sample_rate
                    -physical_max
                    -physical_min
                    -digital_max
                    -digital_min
                    -transducer
                    -prefilter
    
    Notes
    ----- 

    """
    #load data
    if input_path.lower().endswith('.mat'):
        data = rs.load_mat(input_path)
    else:
        data = rs.load_edf(input_path)
 
    sigs_all = data[0]
    Fs = data[1]
    chans = data[2]
    channel_info = data[3]
    
    #define channels to extract
    if channels_ref == None:

        # C3-M2 and C4-M1 signals already exist
        sigs = [sigs_all[chans.index(ch)] for ch in channels_to_extract]
        ch_names = channels_to_extract
        channel_info_short = ([channel_info[chans.index(ch)]
                              for ch in channels_to_extract])
        
    else:                   
        # generate C3-M2 and C4-M1 signals
        sigs = []
        ch_names = []
        channel_info_short = []
        
        for i,ch in enumerate(channels_to_extract):
          try:
              #find new channel name
              ch_name = ch+'-'+channels_ref[i]

              #find signal values for new channel
              sig_adj = (sigs_all[chans.index(ch)] -
                         sigs_all[chans.index(channels_ref[i])])

              #add channel_info for new channel
              temp_info = dict(channel_info[chans.index(ch)])
              temp_info['label'] = ch_name

              #append all channel info to lists
              ch_names.append(ch_name)
              sigs.append(sig_adj)
              channel_info_short.append(temp_info)

          except:
              if '+' in channels_ref[i]:

                  #find new channel name
                  ch_name = ch+'-avg('+channels_ref[i]+')'

                  #find signal values for new channel
                  sigs_ref = sum([sigs_all[chans.index(c)]
                                  for c in channels_ref[i].split('+')])/2
                  
                  sig_adj = sigs_all[chans.index(ch)] - sigs_ref

                  #add channel_info for new channel
                  temp_info = dict(channel_info[chans.index(ch)])
                  temp_info['label'] = ch_name

                  #append all channel info to lists
                  ch_names.append(ch_name)
                  sigs.append(sig_adj)
                  channel_info_short.append(temp_info)

              else: 
                  #no ref for ch
                  ch_names.append(ch)
                  sigs.append(sigs_all[chans.index(ch)])
                  channel_info_short.append(channel_info[chans.index(ch)])
    
    if ecg_corrected and ecg_label_lst is not None:

        ecg_sig_lst = [sigs_all[chans.index(ecg)] for ecg in ecg_label_lst]
        
        eeg_corrected = preprocessing_ecg(sigs, ecg_sig_lst, ecg_label_lst, Fs,
                                          l_freq_eeg=l_freq_eeg, l_freq_ecg=l_freq_ecg,
                                          h_freq_eeg=h_freq_eeg, h_freq_ecg=h_freq_ecg,
                                          smooth=smooth, size=size, parallel=parallel,
                                          n_jobs=n_jobs)

        #re-generate EDF with desired signals
        tup = namedtuple('EDF', 'sigs_selected sample_rate chan_lab channel_info')
        tupleData = tup(eeg_corrected, Fs, ch_names, channel_info_short)

        return tupleData
    
    else:
        #minimal pre-processing of EEG 
        eeg_sigs2 = []
    
        for eeg_sig in sigs:
            eeg = preprocessing(eeg_sig, Fs, l_freq_eeg, h_freq_eeg)
            eeg_sigs2.append(eeg)

        #re-generate EDF with desired signals
        tup = namedtuple('EDF', 'sigs_selected sample_rate chan_lab channel_info')
        tupleData = tup(eeg_sigs2, Fs, ch_names, channel_info_short)
            
    return tupleData

