# -*- coding: utf-8 -*-
"""
LUNAspindles.signals.ecg_correction
-----------------------------------
This module processes ecg signals and uses them to correct eeg signals.
(c) 2019 by Noor Adra (CDAC Westover Lab, MGH)
license?
"""

from scipy.stats import linregress, skew
from joblib import Parallel, delayed
from biosppy.signals.ecg import *
import biosppy.signals.tools as st
import numpy as np
#import matplotlib.pyplot as plt
#import pdb

def extract_r_peaks(ecg_sig, Fs):
       """
       Modified 'ecg' from biosppy. Processes an ECG signal and extracts rpeaks.

       Pan-Tompkins algorithm
       [ref] Pan,J. and Tompkins,W.J.,1985. A real-time QRS detection algorithm.
       IEEE Trans. Biomed. Eng, 32(3), pp.230-236. 

       Parameters
       ----------
       ecg_sig : list or array
              signals from one ECG channel, shape = (T,)
       Fs : int
              sampling rate (Hz)

       Returns
       -------
       rpeaks : array
              rpeak locations indices
    
       Notes
       ----- 

       """
       # check inputs
       if ecg_sig is None:
           raise TypeError("Please specify an input signal.")
       
       # ensure numpy
       ecg_sig = np.array(ecg_sig)
       Fs = float(Fs)
       
       if len(ecg_sig)>0:
              # filter signal
              order = int(0.3 * Fs)
              filtered, _, _ = st.filter_signal(signal=ecg_sig,
                                             ftype='FIR',
                                             band='bandpass',
                                             order=order,
                                             frequency=[3, 45],
                                             sampling_rate=Fs)
              # segment
              rpeaks, = hamilton_segmenter(signal=filtered, sampling_rate=Fs)
              # correct R-peak locations
              rpeaks, = correct_rpeaks(signal=filtered,
                                    rpeaks=rpeaks,
                                    sampling_rate=Fs,
                                    tol=0.05)
              # extract templates
              templates, rpeaks = extract_heartbeats(signal=filtered,
                                                  rpeaks=rpeaks,
                                                  sampling_rate=Fs,
                                                  before=0.2,
                                                  after=0.4) 
       else:
            rpeaks = []
            
       return rpeaks


def extract_r_peaks_parallel(ecg_sig, Fs, n_jobs, verbose=False):
       """
       R peak detection in parallel using joblib
       extract_r_peaks is very slow for long ECG, here we segment into shorter
       segments, detect R peaks in parallel, then combine.

       Pan-Tompkins algorithm
       [ref] Pan,J. and Tompkins,W.J.,1985. A real-time QRS detection algorithm.
       IEEE Trans. Biomed. Eng, 32(3), pp.230-236. 

       Parameters
       ----------
       ecg_sig : list or array
              signals from one ECG channel, shape = (T,)
       Fs : int
              sampling rate (Hz)
       n_jobs : optional, int
              number of CPU used for multiprocessing
       verbose: optional, bool
              if True, displays progress
       
       Returns
       -------
       rpeaks : array
              rpeak locations indices
    
       Notes
       ----- 

       """
       ids = np.array_split(np.arange(len(ecg_sig)), 100)
       r_peaks = (Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(
                            extract_r_peaks)(ecg_sig[id_], Fs) for id_ in ids))
    
       #Parallel returns a list of rpeaks from each call of extract_r_peaks
       #combine while shifting them based on the start of each shorter segment
       r_peaks = np.concatenate([r_peaks[i]+ids[i][0]
                                 for i in range(len(r_peaks))])
       
       return r_peaks


def get_hr(rpeaks,Fs,smooth=True, size=3):
       """
       Calculates instantaneous HR from array of rpeaks.

       Parameters
       ----------
       rpeaks : array
              rpeak locations indices
       Fs : int
              sampling rate (Hz)
       smooth : optional, bool
              if True, performs smoothing on the resulting heart rate
       size : optional, int
              size of smoothing window; ignored if `smooth` is False

       Returns
       -------
       hr : array
              instantaneous hr within physiological means 
       hrOutlierN : int
              number of hr measurements >120 or <40  

       Notes
       ----- 

       """
       #find HR
       hr = Fs * (60. / np.diff(rpeaks))  #('heart rate = %.1f/min'%(len(rpeaks)/(len(ecg)/Fs/60),))
    
       #only include physiologically possible HR readings
       indx = np.nonzero(np.logical_and(hr >= 40, hr <= 200))
       hr = hr[indx]

       #smooth with moving average
       if smooth and (len(hr) > 1):
              hr, _ = st.smoother(signal=hr, kernel='boxcar',
                                  size=size, mirror=True)

       #look at specific window (40-120bpm) to assess quality
       hrOutlier = np.nonzero(np.logical_or(hr < 40, hr > 120))
       hrOutlierN = len(hr[hrOutlier])

       return hr, hrOutlierN


def choose_ecg_ch_parallel(ecg_sig_lst, ecg_label_lst, Fs,
                           smooth=True, size=3, parallel=True, n_jobs=8):
       """
       Selects ECG channel from a list.
       -If 2+ channels, selects channel with best HR (measurements closest to
       physiological readings).
       -In the case of a tie, one is selected (for natus data:
       ECG-LA will be selected in ties)

       Parameters
       ----------
       ecg_sig_lst : list
              ECG signal arrays from 1 or more channels
       ecg_label_lst : list
              ECG channel names; labels corresponds to signals in ecg_sig_lst
       Fs : int
              sampling rate (Hz)
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
       rpeaks : array
              rpeak locations indices
       
       Notes
       ----- 

       """
       tupLst = []

       for ecg_sig,ecg_label in zip(ecg_sig_lst, ecg_label_lst):

              if parallel:
                  rp = extract_r_peaks_parallel(ecg_sig, Fs, n_jobs=n_jobs, verbose=False)
                  hr, hrOutlierN = get_hr(rp,Fs,smooth=smooth, size=size)
                  tupLst.append((hrOutlierN,ecg_label, rp))

              else:
                  rp = extract_r_peaks(ecg_sig, Fs)
                  hr, hrOutlierN = get_hr(rp,Fs,smooth=smooth, size=size)
                  tupLst.append((hrOutlierN,ecg_label, rp))
                  
       rpeaks = min(tupLst)[2] #in case of tie, ECG-LA will be chosen

       return rpeaks



def ecg_correction(eeg_sig, ecg_sig_lst, ecg_label_lst, Fs,
                   return_rpeaks=False, smooth=True, size=3, parallel =True, n_jobs=8):
       """
       Remove ECG artifact by substracting the ensemble average
       
       [ref] Purcell, S.M., et. al., 2017.
       Characterizing sleep spindles in 11,630 individuals from the
       National Sleep Research Resource. Nature communications, 8, p.15930.
       
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
       return_rpeaks= optional, bool
              if True, returns array of rpeak locations indices
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
       eeg2: array
              ECG-corrected EEG signals 
       rpeaks : array
              rpeak locations indices
       
       Notes
       ----- 

       """
       eeg_sig = np.array(eeg_sig)
       assert eeg_sig.ndim==2, 'eeg.shape == (#channel, T)'

       for ecg_sig in ecg_sig_lst:
              ecg_sig = np.array(ecg_sig)
              assert ecg_sig.ndim==1, 'ecg.shape != (T,)'
              assert eeg_sig.shape[1]==len(ecg_sig), 'length of eeg and ecg are not the same'
        
       # get R peaks from ecg
       # rpeaks is an array of R peak indices
       # use parallel version to be faster (depends on your machine)
       # *DO NOT USE* parallel version if this function itself is being used in parallel
       #rpeaks = extract_r_peaks_parallel(ecg_sig, Fs, n_jobs=8)

       rpeaks = choose_ecg_ch_parallel(ecg_sig_lst, ecg_label_lst, Fs, n_jobs=n_jobs,
                                       smooth=smooth, size=size, parallel=True)
    
       #get the averaged signature of eeg aligned to each R-peak
       left_offset = int(round(0.5*Fs))   # left 0.5s
       right_offset = int(round(0.5*Fs))  # right 0.5s
       segs = []
       for rloc in rpeaks:
              if rloc-left_offset<0 or rloc+right_offset>eeg_sig.shape[1]:  # ignore boundary case
                   continue
            
              segs.append(eeg_sig[:, rloc-left_offset:rloc+right_offset])
 
       #convert to array, shape=(#rpeaks, #channel, (end-start)*Fs)
       segs = np.array(segs)

       #take the average, shape=(#channel, (end-start)*Fs)
       eeg_signature = np.nanmean(segs, axis=0)

       #substract eeg signature in the signal aligned to each R peak
       eeg2 = np.array(eeg_sig)
       slopes = []

       for rloc in rpeaks:
              if rloc-left_offset<0 or rloc+right_offset>eeg_sig.shape[1]:  # ignore boundary case
                     continue
     
              old_val = eeg_sig[:, rloc-left_offset:rloc+right_offset]

              #for each channel,
              for ci in range(eeg_sig.shape[0]):
                     #do linear regression between the signature and the actual signal
                     #substract slope*signature
                     slope, _, _, pval, _ = linregress(eeg_signature[ci], old_val[ci])
                     if pval<0.05:
                            new_val = old_val[ci] - slope*eeg_signature[ci]
                     else:
                            new_val = old_val[ci]
                     eeg2[ci, rloc-left_offset:rloc+right_offset] = new_val
                     slopes.append(slope)
     
       if return_rpeaks:
              return eeg2, rpeaks
       else:
              return eeg2

##def plot_ecg(ecg_sig, Fs, rpeaks=False):
##    if not rpeaks:
##        ecg = preprocessing(ecg_sig, Fs, l_freq=0.3, h_freq=40)
##        plt.plot(ecg)
##        plt.xlim([4000000,4000000+Fs*10])
##        plt.show()
##    else:
##        pdb.set_trace()
