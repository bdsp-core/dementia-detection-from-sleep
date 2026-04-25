import os
import subprocess
import shutil
import sys
import numpy as np
import pandas as pd
import h5py
import scipy.io as sio
from tqdm import tqdm
import pyedflib
from docx import Document
import mne
sys.path.insert(0, 'myml_lib')
from load_mgh_sleep_dataset import check_load_Twin_dataset


def convert_to_edf(input_path, output_path, Fs, channels=None):

    import pdb;pdb.set_trace()
    if input_path.lower().endswith('.mat'):
        try:
            ff = sio.loadmat(input_path)
            channel_names = [ff['hdr'][0,i]['signal_labels'][0] for i in range(ff['hdr'].shape[1])]
            channel_names = [x.upper().replace('A','M').replace('-','').replace('C3M1','C3M2').replace('C4M2','C4M1') for x in channel_names]
            if channels is None:
                channel_ids = np.arange(ff['hdr'].shape[1])
            else:
                channel_ids = [channel_names.index(ch) for ch in channels]
                
            channel_info = [
                    {'label': channel_names[i],
                     'dimension': ff['hdr'][0,i]['physical_dimension'][0],
                     'sample_rate': Fs,
                     'physical_max': ff['hdr'][0,i]['physical_max'][0,0],
                     'physical_min': ff['hdr'][0,i]['physical_min'][0,0],
                     'digital_max': ff['hdr'][0,i]['digital_max'][0,0],
                     'digital_min': ff['hdr'][0,i]['digital_min'][0,0],
                     'transducer': ff['hdr'][0,i]['tranducer_type'][0],
                     'prefilter': ff['hdr'][0,i]['prefiltering'][0] if len(ff['hdr'][0,i]['prefiltering'])>0 else ''}
                for i in channel_ids]
            EEG = ff['s'][channel_ids]
            
        except Exception as ee:
            with h5py.File(input_path, 'r') as ff:
                channel_names = [''.join(map(chr,ff[ff['hdr']['signal_labels'][i,0]][:].flatten())) for i in range(ff['hdr']['signal_labels'].shape[0])]
                channel_names = [x.upper().replace('A','M').replace('-','').replace('C3M1','C3M2').replace('C4M2','C4M1') for x in channel_names]
                if channels is None:
                    channel_ids = np.arange(ff['hdr']['signal_labels'].shape[0])
                else:
                    channel_ids = [channel_names.index(ch) for ch in channels]
                    
                physical_dimension = [''.join(map(chr,ff[ff['hdr']['physical_dimension'][i,0]][:].flatten())) for i in range(ff['hdr']['physical_dimension'].shape[0])]
                tranducer_type = [''.join(map(chr,ff[ff['hdr']['tranducer_type'][0,0]][:].flatten())) for i in range(ff['hdr']['tranducer_type'].shape[0])]
                
                channel_info = [
                        {'label': channel_names[i],
                         'dimension': physical_dimension[i],
                         'sample_rate': Fs,
                         'physical_max': 32767,
                         'physical_min': -32768,
                         'digital_max': 32767,
                         'digital_min': -32768,
                         'transducer': tranducer_type[i],
                         'prefilter': ''}
                    for i in channel_ids]
                
                EEG = ff['s'][:, channel_ids].T
    
    elif input_path.lower().endswith('.edf'):
        ff = mne.io.read_raw_edf(input_path, verbose=False, stim_channel=None)
        channel_names = ff.info['ch_names']
        channel_names = [cn.upper().replace('A','M').replace('-','').replace('C3M1','C3M2').replace('C4M2','C4M1').replace('CZM1','C4M1') for cn in channel_names]
        
        if channels is None:
            channel_ids = np.arange(ff['hdr'].shape[1])
        else:
            channel_ids = [channel_names.index(ch) for ch in channels]
            
        Fs = ff.info['sfreq']
        
        channel_info = [
                {'label': channel_names[i],
                 'dimension': 'uV',
                 'sample_rate': Fs,
                 'physical_max': 32767,
                 'physical_min': -32768,
                 'digital_max': 32767,
                 'digital_min': -32768,
                 'transducer': 'E',
                 'prefilter': ''}
            for i in channel_ids]
        
        EEG = ff.get_data()[channel_ids]*1e6
        
    if EEG.shape[1]<=100:
        raise ValueError('Short EEG')

    with pyedflib.EdfWriter(output_path, len(EEG), file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setSignalHeaders(channel_info)
        f.writeSamples(EEG)


def convert_to_xml(input_path, output_path, Fs):
    sleep_stage_mapping = {-1:0, 0:0, 5:0, 4:5, 3:1, 2:2, 1:2} # map N3 to N2, so that N2 = N2 + N3
    
    with h5py.File(input_path, 'r') as ff:
        sleep_stages = ff['stage'][:].flatten()
    
    sleep_stages = sleep_stages[np.arange(0, len(sleep_stages), Fs*30)]
    sleep_stages[np.isnan(sleep_stages)] = -1
    with open(output_path, 'w') as ff:
        ff.write('<CMPStudyConfig>\n')
        ff.write('<EpochLength>30</EpochLength>\n')
        ff.write('<SleepStages>\n')
        for ss in sleep_stages:
            ff.write('<SleepStage>%d</SleepStage>\n'%sleep_stage_mapping[ss])
        ff.write('</SleepStages>\n')
        ff.write('</CMPStudyConfig>')


def detect_spindle_so(data_list, output_dir, channels, verbose=True):
    edf_paths = []
    xml_paths = []
    Fs = 200
    for ii in tqdm(range(len(data_list)), disable=not verbose):
        try:
            basename = os.path.basename(os.path.splitext(data_list.signal_path.iloc[ii])[0])
                
            edf_path = os.path.join(output_dir, basename+'.edf')
            xml_path = os.path.join(output_dir, basename+'.xml')
            if os.path.exists(edf_path) and os.path.exists(xml_path):
                edf_paths.append(edf_path)
                xml_paths.append(xml_path)
                continue
                
            # create edf
            convert_to_edf(data_list.signal_path.iloc[ii], edf_path, Fs, channels=channels)
            # create sleep stage xml
            convert_to_xml(data_list.label_path.iloc[ii], xml_path, Fs)
                
            edf_paths.append(edf_path)
            xml_paths.append(xml_path)
            
        except Exception as ee:
            print('%s: %s'%(basename, ee.message))
            if os.path.exists(edf_path):
                os.remove(edf_path)
            if os.path.exists(xml_path):
                os.remove(xml_path)
            continue
        
    # create the list file
    list_path = os.path.join(output_dir, 'luna.lst')
    df = pd.DataFrame(data={'sid': [os.path.basename(x).lower().replace('.edf','') for x in edf_paths], 'edf':edf_paths, 'xml':xml_paths})
    df = df[['sid', 'edf', 'xml']]
    df.to_csv(list_path, sep='\t', index=False, header=False)
    
    # spindle detection for N2 + N3
    extract_code_path = os.path.join(output_dir, 'convert_luna_output_db2mat.R')
    
    luna_db_path = os.path.join(output_dir, 'luna_output.db')
    output_path = os.path.join(output_dir, 'luna_output.xlsx')
    subprocess.check_call(['luna', list_path, '-o', luna_db_path, '-s', 'MASK ifnot=NREM2 & SPINDLES sig=C3M2,C4M1 q=0.3 sw mag=1'])
    with open(extract_code_path, 'w') as ff:
        ff.write("""library(luna)
library(R.matlab)
library(xlsx)

k<-ldb('%s')
d_spindle<-lx(k,"SPINDLES", "CH_F")
d_sw<-lx(k,"SPINDLES", "CH")
write.xlsx(d_spindle, '%s', sheetName='spindle', row.names=FALSE)
write.xlsx(d_sw, '%s', sheetName='sw', append=TRUE, row.names=FALSE)"""%(luna_db_path, output_path, output_path))
    subprocess.check_call(['Rscript', extract_code_path])
    if os.path.exists(extract_code_path):
        os.remove(extract_code_path)
        
    res_spindle = pd.read_excel(output_path, sheet_name='spindle')
    res_sw = pd.read_excel(output_path, sheet_name='sw')
    if os.path.exists(output_path):
        os.remove(output_path)
    res = res_spindle.merge(res_sw, on=['ID', 'CH'])
    
    return res


if __name__=='__main__':
    
    df_paths = pd.read_excel('subject_files.xlsx')
    df = pd.read_csv('cohort_subset.csv')
    
    df_paths['PID'] = [os.path.basename(df_paths.signal_path[i])[len('Signal_'):-4].replace(',','').replace('.', '') for i in range(len(df_paths))]
    df.PID = df.PID.str.replace(',', '').replace('.', '')
    df_paths = df_paths.join(df.set_index('PID'), on='PID', how='right')
    
    df_paths = df_paths[df_paths.PID=='TwinData3_263']
    
    res = detect_spindle_so(df_paths, 'spindles', ['C3M2','C4M1'])
    
    res.to_excel('spindle_so_complete.xlsx', index=False)
    
