import datetime
import mne
import os.path
import re
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
from dateutil.parser import parse


def check_load_Twin_dataset(data_path, label_path, channels=None, report_and_actual_time_tol=300, typeoftest=None):
    if data_path.endswith('.mat'):
        try:
            ff = sio.loadmat(data_path)
            signals = ff['s']
            channel_names = [ff['hdr'][0,i]['signal_labels'][0].upper() for i in range(ff['hdr'].shape[1])]
        except Exception as ee:
            with h5py.File(data_path, 'r') as ff:
                signals = ff['s'][:]
                channel_names = [''.join(map(chr,ff[ff['hdr']['signal_labels'][i,0]][:].flatten())).upper() for i in range(len(ff['hdr']['signal_labels']))]
        Fs = 200.
    elif data_path.endswith('.edf'):
        ff = mne.io.read_raw_edf(data_path, preload=True, verbose=False, stim_channel=None, exclude=['Fp1', 'Fp2', 'P3', 'P4', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Chin1', 'Chin2', 'Chin3', 'E2', 'E1', 'AirFlow', 'Chest', 'LAT', 'Snore', 'Abdomen', 'RAT', '30', 'IC', 'Leak', 'EtCO2', 'CO2 Wave', 'DC6', 'DC7', 'DC8', 'DC9', 'DC10', 'DC11', 'PTAF', 'SpO2', 'PR', 'Pleth', 'EDF Annotations'])
        signals = ff.get_data()
        channel_names = ff.info['ch_names']
        signals = np.array([signals[channel_names.index('F3')] - signals[channel_names.index('M2')],
                        signals[channel_names.index('F4')] - signals[channel_names.index('M1')],
                        signals[channel_names.index('C3')] - signals[channel_names.index('M2')],
                        signals[channel_names.index('C4')] - signals[channel_names.index('M1')],
                        signals[channel_names.index('O1')] - signals[channel_names.index('M2')],
                        signals[channel_names.index('O2')] - signals[channel_names.index('M1')],])
        signals = signals*1e6  # mne.io.read_raw_edf automatically converts to V, we convert back to uV
        channel_names = ['F3M2','F4M1','C3M2','C4M1','O1M2','O2M1']
        starttime = datetime.datetime.utcfromtimestamp(ff.info['meas_date'][0])
        Fs = ff.info['sfreq']
    else:
        raise Exception('File is not mat or edf')
    channel_names = [x.replace('FP1-M2','F3-M2').replace('FP2-M1','F4-M2') for x in channel_names]
            
    # check channel number
    if signals.shape[0]!=len(channel_names) and signals.shape[1]==len(channel_names):
        signals = signals.T
    elif signals.shape[0]!=len(channel_names) and signals.shape[1]!=len(channel_names):
        raise Exception('Inconsistent channel number')

    # load labels
    if label_path.endswith('.mat'):
        with h5py.File(label_path, 'r') as ffl:
            sleep_stage = ffl['stage'][()].flatten()
    elif label_path.endswith('.csv'):
        stage_txt2num = {'w':5, 'awake':5,
                         'r':4, 'rem':4,
                         'n1':3, 'nrem1':3,
                         'n2':2, 'nrem2':2,
                         'n3':1, 'nrem3':1,}
        ss_df = pd.read_csv(label_path)
        ss_df.event = ss_df.event.str.lower().str.strip()
        ss_df = ss_df[ss_df.event.str.startswith('sleep_stage_')].reset_index(drop=True)
        ss_df['stage'] = ss_df.event.str.split('_', expand=True)[2]
        ss_df = ss_df[np.in1d(ss_df['stage'], list(stage_txt2num.keys()))].reset_index(drop=True)
                         
        oneday = datetime.timedelta(days=1)
        sleep_stage = np.zeros(signals.shape[1])+np.nan
        for i in range(len(ss_df)):
            thistime = parse(ss_df.time.iloc[i], default=starttime)
            if thistime.hour<12 and starttime.hour>12:
                thistime += oneday
            assert thistime>=starttime, 'thistime<starttime'
            startid = round((thistime - starttime).total_seconds()*Fs)
            if startid<len(sleep_stage):
                sleep_stage[startid:] = stage_txt2num.get(ss_df.stage.iloc[i], np.nan)
            if i==len(ss_df)-1 and startid+round(Fs*30)<len(sleep_stage):
                sleep_stage[startid+round(Fs*30):] = np.nan

    # check signal length = sleep stage length
    if sleep_stage.shape[0]!=signals.shape[1]:
        raise Exception('Inconsistent sleep stage length (%d) and signal length (%d)'%(sleep_stage.shape[0],signals.shape[1]))
    
    # if split night, decide the split point
    split_sample_id = np.nan
    if type(typeoftest)==str and 'split' in typeoftest.lower():
        with h5py.File(label_path, 'r') as ffl:
            start = ffl['features']['Treatment']['Start'][:].flatten()
        ids = np.where(start>0)[0]
        if len(ids)>0:
            split_sample_id = min(ids)  # take the earliest split sample id
        else:
            raise Exception('Split start = %s'%ids)
    
    # only take EEG channels to study
    if channels is None:
        EEG_channel_ids = list(range(len(channel_names)))
    else:
        EEG_channel_ids = []
        for i in range(len(channels)):
            channel_name_pattern = re.compile(channels[i][:2].upper()+'-*[AM][12]')
            found = False
            for j in range(len(channel_names)):
                if channel_name_pattern.match(channel_names[j].upper()):
                    EEG_channel_ids.append(j)
                    found = True
                    break
            if not found:
                raise Exception('Channel %s is not found'%channels[i])
        EEG = signals[EEG_channel_ids,:]#.T

    # check whether the EEG signal contains NaN
    if np.any(np.isnan(EEG)):
        raise Exception('Found Nan in EEG signal')

    # check whether sleep_stage contains all 5 stages
    stages = np.unique(sleep_stage[np.logical_not(np.isnan(sleep_stage))]).astype(int).tolist()
    if len(stages)<=2:
        raise Exception('#sleep stage <= 2: %s'%stages)

    params = {'Fs':Fs, 'EEG_channel_ids':EEG_channel_ids, 'split_sample_id':split_sample_id}
    return EEG, sleep_stage, params

