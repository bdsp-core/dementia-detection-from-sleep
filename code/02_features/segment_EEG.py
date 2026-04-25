#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from scipy.signal import detrend
from scipy.stats import mode
from joblib import Parallel, delayed
import mne
mne.set_log_level(verbose='WARNING')
from mne.filter import filter_data, notch_filter
from scikits.samplerate import resample
from mne.time_frequency import psd_array_multitaper
from extract_features_parallel import *


seg_mask_explanation = [
    'normal',
    'around sleep stage change point',
    'NaN in sleep stage',
    'NaN in EEG',
    'overly high/low amplitude',
    'flat signal',
    'NaN in feature',
    'NaN in spectrum',
    'muscle artifact',
    'spurious spectrum']


def segment_EEG(EEG, labels, window_time, step_time, Fs, eeg_channels, newFs=200, notch_freq=None, bandpass_freq=None, start_end_remove_window_num=0, amplitude_thres=500, n_jobs=-1, to_remove_mean=False):#
    """Segment EEG signals.

    Arguments:
    EEG -- np.ndarray, size=(channel_num, sample_num)
    labels -- np.ndarray, size=(sample_num,)
    window_time -- in seconds
    Fz -- in Hz

    Keyword arguments:
    notch_freq
    bandpass_freq
    start_end_remove_window_num -- default 0, number of windows removed at the beginning and the end of the EEG signal
    amplitude_thres -- default 1000, mark all segments with np.any(EEG_seg>=amplitude_thres)=True
    to_remove_mean -- default False, whether to remove the mean of EEG signal from each channel

    Outputs:    
    EEG segments -- a list of np.ndarray, each has size=(window_size, channel_num)
    labels --  a list of labels of each window
    segment start ids -- a list of starting ids of each window in (sample_num,)
    segment masks --
    """
    std_thres = 0.2
    std_thres2 = 1.
    flat_seconds = 5
    padding = 0
    
    if to_remove_mean:
        EEG = EEG - np.mean(EEG,axis=1, keepdims=True)
    window_size = int(round(window_time*Fs))
    step_size = int(round(step_time*Fs))
    flat_length = int(round(flat_seconds*Fs))
    
    start_ids = np.arange(0, EEG.shape[1]-window_size+1, step_size)
    if start_end_remove_window_num>0:
        start_ids = start_ids[start_end_remove_window_num:-start_end_remove_window_num]
    labels_ = []
    for si in start_ids:
        labels2 = labels[si:si+window_size]
        labels2[np.isinf(labels2)] = np.nan
        labels2[np.isnan(labels2)] = -1
        label__ = mode(labels2).mode[0]
        if label__==-1:
            labels_.append(np.nan)
        else:
            labels_.append(label__)
    labels = np.array(labels_)
    
    seg_masks = [seg_mask_explanation[0]]*len(start_ids)
    
    label_diff = np.diff(labels)
    label_diff_pos = np.array([i for i in range(len(label_diff)) if not np.isnan(label_diff[i]) and label_diff[i]!=0])
    label_diff_pos = sorted(label_diff_pos.tolist() + (label_diff_pos+1).tolist())
    if len(label_diff_pos)>0:
        for i in label_diff_pos:
            seg_masks[i] = seg_mask_explanation[1]
    if np.any(np.isnan(labels)):
        ids = np.where(np.isnan(labels))[0]
        for i in ids:
            seg_masks[i] = seg_mask_explanation[2]
    
    if notch_freq is not None and bandpass_freq is not None and np.max(bandpass_freq)>=notch_freq:
        EEG = notch_filter(EEG, Fs, notch_freq, fir_design="firwin")  # (#window, #ch, window_size+2padding)
    if bandpass_freq is not None:
        EEG = filter_data(EEG, Fs, bandpass_freq[0], bandpass_freq[1], fir_design="firwin")#detrend(EEG, axis=1), n_jobs='cuda'
    
    # resample
    if Fs!=newFs:
        r = newFs*1./Fs
        EEG = Parallel(n_jobs=-1, verbose=False)(delayed(resample)(EEG[i], r, 'sinc_best') for i in range(len(EEG)))
        EEG = np.array(EEG).astype(float)
        Fs = newFs
        window_size = int(round(window_time*Fs))
        step_size = int(round(step_time*Fs))
        flat_length = int(round(flat_seconds*Fs))
        start_ids = np.arange(0, EEG.shape[1]-window_size+1, step_size)
        if start_end_remove_window_num>0:
            start_ids = start_ids[start_end_remove_window_num:-start_end_remove_window_num]
    
    # normalize
    q1,q2,q3 = np.nanpercentile(EEG, (25,50,75), axis=1)
    
    #else:
    EEG_segs = EEG[:,list(map(lambda x:np.arange(x-padding,x+window_size+padding), start_ids))].transpose(1,0,2)  # (#window, #ch, window_size+2padding)
    #    raise NotImplementedError('Notch freqency within band pass range.')
    #EEG_segs = detrend(EEG_segs[:,:,padding:-padding], axis=2)  # (#window, #ch, window_size)
    
    #TODO detrend(EEG_segs)
    #TODO remove_mean(EEG_segs) to remove frequency at 0Hz
    
    NW = 10.
    BW = NW*2./window_time
    specs, freq = psd_array_multitaper(EEG_segs, Fs, fmin=bandpass_freq[0], fmax=bandpass_freq[1], adaptive=False, low_bias=True, n_jobs=n_jobs, verbose='ERROR', bandwidth=BW, normalization='full')
    specs = specs.transpose(0,2,1)
    
    nan2d = np.any(np.isnan(EEG_segs), axis=2)
    nan1d = np.where(np.any(nan2d, axis=1))[0]
    for i in nan1d:
        seg_masks[i] = seg_mask_explanation[3]
    
    amplitude_large2d = np.any(np.abs(EEG_segs)>amplitude_thres, axis=2)
    amplitude_large1d = np.where(np.any(amplitude_large2d, axis=1))[0]
    for i in amplitude_large1d:
        seg_masks[i] = seg_mask_explanation[4]
            
    # if there is any flat signal with flat_length
    short_segs = EEG_segs.reshape(EEG_segs.shape[0], EEG_segs.shape[1], EEG_segs.shape[2]//flat_length, flat_length)
    flat2d = np.any(detrend(short_segs, axis=3).std(axis=3)<=std_thres, axis=2)
    flat2d = np.logical_or(flat2d, np.std(EEG_segs,axis=2)<=std_thres2)
    flat1d = np.where(np.any(flat2d, axis=1))[0]
    for i in flat1d:
        seg_masks[i] = seg_mask_explanation[5]
    """
    spec_median, spec_max = np.percentile(specs, (50,100), axis=1)
    spec_ratio = spec_max/spec_median
    spec_ratio[np.logical_or(np.isinf(spec_ratio),np.isnan(spec_ratio))] = np.inf
    ids = np.where(np.any(spec_ratio>=200, axis=1))[0]
    for i in ids:
        seg_masks[i] = seg_mask_explanation[9]
    """
    
    # normalize
    nch = EEG_segs.shape[1]
    EEG_segs = (EEG_segs - q2.reshape(1,nch,1)) / (q3.reshape(1,nch,1)-q1.reshape(1,nch,1))
    
    lens = [len(EEG_segs), len(labels), len(start_ids), len(seg_masks), len(specs)]
    if len(set(lens))>1:
        minlen = min(lens)
        EEG_segs = EEG_segs[:minlen]
        labels = labels[:minlen]
        start_ids = start_ids[:minlen]
        seg_masks = seg_masks[:minlen]
        specs = specs[:minlen]

    return EEG_segs, labels, start_ids, seg_masks, specs, freq


