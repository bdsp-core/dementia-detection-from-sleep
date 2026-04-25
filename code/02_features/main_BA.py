#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import Counter
import datetime
from dateutil.parser import parse
import os
import os.path
import shutil
import sys
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import io as sio
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from load_mgh_sleep_dataset import *
from segment_EEG import *
from extract_features_parallel import *
sys.path.insert(0, '/data/brain_age/mycode')
from dnn_regressor import my_nobias_loss
from step3_train_age import impute_missing_stage

import matplotlib
matplotlib.rc('pdf', fonttype=42)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn
seaborn.set_style('ticks')

epoch_length = 30 # [s]
start_end_remove_epoch_num = 1
amplitude_thres = 500 # [uV]
changepoint_epoch_num = 1
EEG_channels = ['F3M2','F4M1','C3M2','C4M1','O1M2','O2M1']
combined_EEG_channels =  ['F', 'C', 'O']
line_freq = 60.  # [Hz]
bandpass_freq = [0.5, 20.]  # [Hz]
tostudy_freq = [0.5, 20.]  # [Hz]
random_state = 2
n_jobs = 18


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


def myprint(seg_mask):
    sm = Counter(seg_mask)
    for ex in seg_mask_explanation:
        if ex in sm:
            print('%s: %d/%d, %g%%'%(ex,sm[ex],len(seg_mask),sm[ex]*100./len(seg_mask)))


if __name__=='__main__':
    normal_only = True
    np.random.seed(random_state)

    feature_dir = '/media/mad3/Projects/SLEEP/SLEEP_STAGING/all_brain_age_features2'

    subject_files_path = 'subject_files.xlsx'
    if os.path.exists(subject_files_path):
        print('Reading from %s'%subject_files_path)
        subject_files = pd.read_excel(subject_files_path)

    else:
        subject_files1 = pd.read_csv('/media/mad3/Datasets_ConvertedData/sleeplab/grass_studies_list.csv', sep=',')
        subject_files2 = pd.read_csv('/media/mad3/Datasets_ConvertedData/sleeplab/natus_studies_list.csv', sep=',')
        subject_files1 = subject_files1[['MRN', 'LastName', 'FirstName', 'Sex', 'DateOfBirth', 'DateOfVisit', 'TypeOfTest', 'FolderName', 'Path']]
        subject_files2 = subject_files2[['MRN', 'LastName', 'FirstName', 'Sex', 'DateOfBirth', 'DateOfVisit', 'TypeOfTest', 'FolderName', 'Path']]
        subject_files = pd.concat([subject_files1, subject_files2]).reset_index(drop=True)
        subject_files.Path = subject_files.Path.str.replace('M:', '/media/mad3').str.replace('\\',os.sep).astype(str)
        subject_files.Sex = subject_files.Sex.astype(str).str.strip().str.lower()\
                                        .str.replace('x','')\
                                        .str.replace('nan','')\
                                        .str.replace('female','f')\
                                        .str.replace('male','m')
        subject_files.DateOfVisit = subject_files.DateOfVisit.astype(str)
        subject_files.DateOfBirth = subject_files.DateOfBirth.astype(str)
        subject_files.TypeOfTest = subject_files.TypeOfTest.astype(str)
        subject_files.TypeOfTest = subject_files.TypeOfTest.str.strip().str.lower()\
                                        .str.replace('bipap titration','cpap all night')\
                                        .str.replace('cpap from start','cpap all night')\
                                        .str.replace('dedicated bpap','cpap all night')\
                                        .str.replace('full noc cpap','cpap all night')\
                                        .str.replace('psg all night','cpap all night')\
                                        .str.replace('cpap all night cpap','cpap all night')\
                                        .str.replace('treatm.*','cpap all night')\
                                        .str.replace('titration','cpap all night')\
                                        .str.replace('psg all night cpap','cpap all night')\
                                        .str.replace('psg with cpap all night','cpap all night')\
                                        .str.replace('dia.*','diagnostic')\
                                        .str.replace('psg diagnostic','diagnostic')\
                                        .str.replace('psg split night','split night')\
                                        .str.replace('splight night study','split night')\
                                        .str.replace('bedsid.*','bedside')\
                                        .str.replace('extend.*','extend')\
                                        .str.replace('mwt','mslt')\
                                        .str.replace('smslt','mslt')\
                                        .str.replace('^resear$','research')\
                                        .str.replace('foldername','')\
                                        .str.replace('visit','')\
                                        .str.replace('psg','')\
                                        .str.replace('nan', '')\
                                        .str.replace('lin_xianzhe_072211_2205.000','')

        ## generate subject_files: signal_path    sleep_stage_paths   feature_path  typeoftest    age
        signal_paths = []
        label_paths = []
        feature_paths = []
        typeoftests =[]
        ages = []
        mrns = []
        sexs = []
        for si, sd in enumerate(tqdm(subject_files.Path.values)):
            if not os.path.exists(sd):
                continue
            signal_path = []
            label_path = []
            sfs = os.listdir(sd)

            ss1 = [sf for sf in sfs if re.match('Signal_TwinData[0-9]+_[0-9,.]+\.mat', sf) is not None]
            ss2 = [sf for sf in sfs if re.match('Signal_TwinData[0-9]+_Exported_[0-9,.]+\.mat', sf) is not None]
            ss3 = [sf for sf in sfs if sf.startswith('Signal_') and sf.lower().endswith('.mat')]
            ss4 = [sf for sf in sfs if sf.lower().endswith('.edf')]
            if len(ss1)==1:
                ss_toadd = ss1[0]
                ff_toadd = ss_toadd.replace('Signal_', 'Feature_')
            elif len(ss2)==1:
                ss_toadd = ss2[0]
                ff_toadd = ss_toadd.replace('Signal_', 'Feature_').replace('Exported_','')
            elif len(ss3)==1:
                ss_toadd = ss3[0]
                ff_toadd = ss_toadd.replace('Signal_', 'Feature_')
            elif len(ss4)==1:
                ss_toadd = ss4[0]
                ff_toadd = 'Feature_'+ss_toadd.replace('.edf', '.mat')
            else:
                continue
            if ff_toadd.startswith('Feature_TwinData'):
                ff_toadd = ff_toadd[:-4].replace(',','').replace('.','')+'.mat'

            ll1 = [sf for sf in sfs if re.match('Labels_TwinData[0-9]+_[0-9,.]+\.mat', sf) is not None]
            ll2 = [sf for sf in sfs if re.match('Labels_TwinData[0-9]+_Exported_[0-9,.]+\.mat', sf) is not None]
            ll3 = [sf for sf in sfs if sf.startswith('Labels_') and sf.lower().endswith('.mat')]
            ll4 = [sf for sf in sfs if sf.lower()=='annotations.csv']

            if len(ll1)==1:
                ll_toadd = ll1[0]
            elif len(ll2)==1:
                ll_toadd = ll2[0]
            elif len(ll3)==1:
                ll_toadd = ll3[0]
            elif len(ll4)==1:
                ll_toadd = ll4[0]
            else:
                continue

            signal_paths.append(os.path.join(sd, ss_toadd))
            label_paths.append(os.path.join(sd, ll_toadd))
            feature_paths.append(os.path.join(feature_dir, ff_toadd))
            mrns.append(subject_files.MRN.iloc[si])
            typeoftests.append(subject_files.TypeOfTest.iloc[si])
            sexs.append(subject_files.Sex.iloc[si])

            try:
                dov = parse(subject_files.DateOfVisit.iloc[si])
                dob = parse(subject_files.DateOfBirth.iloc[si])
                if dov.year<=2005:
                    raise ValueError
                if dob.year<=1900 or dob.year>=2018:
                    raise ValueError
                age = (dov-dob).total_seconds()/365./24./3600.
            except Exception as ee:
                age = np.nan
            ages.append(age)

        subject_files = pd.DataFrame({'MRN':mrns, 'signal_path':signal_paths, 'label_path':label_paths, 'feature_path':feature_paths, 'typeoftest':typeoftests, 'sex':sexs, 'age':ages})
        subject_files = subject_files[['MRN', 'age', 'sex', 'typeoftest', 'signal_path', 'label_path', 'feature_path']]
        #subject_files = subject_files.sort_values('signal_path').reset_index(drop=True)
        subject_files.to_excel(subject_files_path, index=False)

    subject_err_path = 'err_subject_reason.txt'
    if os.path.isfile(subject_err_path):
        err_subject_reasons = []
        err_subjects = []
        with open(subject_err_path,'r') as f:
            for row in f:
                if row.strip()=='':
                    continue
                i = row.split(':::')
                err_subjects.append(i[0].strip())
                err_subject_reasons.append(i[1].strip())
    else:
        err_subject_reasons = []
        err_subjects = []

    """
    aa = pd.read_excel('/data/GlennBrainAge_Cognition/mycode/output_BA.xlsx')
    feature_paths = [os.path.basename(subject_files.feature_path[x]) for x in range(len(subject_files))]
    ids = [feature_paths.index(aa.FeaturePath[i]) for i in range(len(aa)) if not pd.isna(aa.FeaturePath[i])]
    subject_files = subject_files.iloc[ids].reset_index(drop=True)
    """

    features = []
    subject_num = len(subject_files)
    for si in range(subject_num):
        data_path = subject_files.signal_path.iloc[si]
        label_path = subject_files.label_path.iloc[si]
        feature_path = subject_files.feature_path.iloc[si]
        subject_file_name = os.path.basename(feature_path)
        if subject_file_name in err_subjects:
            continue
        if os.path.isfile(feature_path):
            print('====== [(%d)/%d] %s %s ======'%(si+1,subject_num,subject_file_name.replace('.mat',''),datetime.datetime.now()))

        else:
            print('\n====== [%d/%d] %s %s ======'%(si+1,subject_num,subject_file_name.replace('.mat',''),datetime.datetime.now()))
            try:
                # check and load dataset
                EEG, sleep_stages_, params = check_load_Twin_dataset(data_path, label_path, channels=EEG_channels, typeoftest=subject_files.typeoftest.iloc[si])
                Fs = params.get('Fs')
                split_sample_id = params.get('split_sample_id')
                newFs = 200.

                # segment EEG
                segs_, sleep_stages_, seg_times_, seg_mask, specs_, freq = segment_EEG(EEG, sleep_stages_, epoch_length, epoch_length, Fs, EEG_channels, newFs, notch_freq=line_freq, bandpass_freq=bandpass_freq, start_end_remove_window_num=start_end_remove_epoch_num, amplitude_thres=amplitude_thres, to_remove_mean=False, n_jobs=n_jobs)
                if segs_.shape[0] <= 0:
                    raise ValueError('Empty EEG segments')

                # for split night
                diag_indicator = np.zeros(len(segs_))
                if ~np.isnan(split_sample_id):
                    diag_indicator[:(split_sample_id//int(epoch_length*Fs)-start_end_remove_epoch_num)] = 1

                Fs = newFs
                if normal_only:
                    good_ids = np.where(np.in1d(seg_mask,seg_mask_explanation[:2]))[0]
                    if len(good_ids)<=300:
                        myprint(seg_mask)
                        raise ValueError('<=300 normal segments')
                    segs_ = segs_[good_ids]
                    specs_ = specs_[good_ids]
                    sleep_stages_ = sleep_stages_[good_ids]
                    seg_times_ = seg_times_[good_ids]
                    diag_indicator = diag_indicator[good_ids]
                else:
                    good_ids = np.arange(len(seg_mask))

                # extract features

                features_, feature_names = extract_features(segs_, EEG_channels, combined_EEG_channels, Fs, 2, tostudy_freq, 2, 1, seg_times_, return_feature_names=True, n_jobs=n_jobs, verbose=True)
                features_[np.isinf(features_)] = np.nan
                nan_ids = np.where(np.any(np.isnan(features_),axis=1))[0]
                for ii in nan_ids:
                    seg_mask[ii] = seg_mask_explanation[6]
                if normal_only:
                    good_ids2 = np.where(np.in1d(np.array(seg_mask)[good_ids],seg_mask_explanation[:2]))[0]
                    segs_ = segs_[good_ids2]
                    specs_ = specs_[good_ids2]
                    sleep_stages_ = sleep_stages_[good_ids2]
                    seg_times_ = seg_times_[good_ids2]
                    diag_indicator = diag_indicator[good_ids2]
                if np.sum(diag_indicator==1)>0:
                    diag_seg_end_id = np.where(diag_indicator==1)[0].max()
                else:
                    diag_seg_end_id = np.nan

                myprint(seg_mask)

            except Exception as e:
                err_info = str(e).split('\n')[0].strip()
                print('\n%s.\nSubject %s is IGNORED.\n'%(err_info,subject_file_name))
                err_subject_reasons.append(err_info)
                err_subjects.append(subject_file_name)

                with open(subject_err_path,'a') as f:
                    msg_ = '%s::: %s\n'%(subject_file_name,err_info)
                    f.write(msg_)
                continue

            sio.savemat(feature_path, {
                'EEG_feature_names':feature_names,
                'EEG_features':features_,
                'EEG_specs':specs_,
                'EEG_frequency':freq,
                'sleep_stages':sleep_stages_,
                'seg_times':seg_times_,
                'age':subject_files.age.iloc[si],
                'sex':subject_files.sex.iloc[si],
                'typeoftest':subject_files.typeoftest.iloc[si],
                'seg_mask':seg_mask,
                'Fs':Fs,
                'diag_end_id_splitnight':diag_seg_end_id})


    ## build input matrix X for testing

    stages = ['W','N1','N2','N3','R']
    stage2num = {'W':5,'R':4,'N1':3,'N2':2,'N3':1}
    num2stage = {stage2num[x]:x for x in stage2num}

    # load EEG features
    ###
    #feature_files = sorted(os.listdir(feature_dir))
    feature_files1 = sorted(set(os.listdir(feature_dir)) & set([os.path.basename(x) for x in subject_files.feature_path.values]))

    # build BA features
    minimum_epochs_per_stage = 5
    X = []
    ages = []
    typeoftests = []
    feature_files2 = []
    for pid in tqdm(range(len(feature_files1))):
        thisdata = sio.loadmat(os.path.join(feature_dir, feature_files1[pid]))
        sleep_stages = thisdata['sleep_stages'].flatten()
        features = thisdata['EEG_features']
        diag_end_id = thisdata['diag_end_id_splitnight'][0,0]
        age = thisdata['age'][0,0]
        typeoftest = thisdata['typeoftest'].flatten()[0]
        
        if pid==0:
            feature_names = np.array(list(map(lambda x:x.strip(), thisdata['EEG_feature_names'])))
            feature_num_each_stage = features.shape[1]

        """
        # criterion 1: slope of spectrum < 0
        thisdata['EEG_specs'] = thisdata['EEG_specs'].transpose(0,2,1)
        specs_db = 10*np.log10(thisdata['EEG_specs'])
        criteria1 = np.all([np.polyfit(np.arange(specs_db.shape[1]), specs_db[ii], 1)[0]<0 for ii in range(len(specs_db))], axis=1)
        # criterion 2: total power not too big, not too small
        totalpower = specs_db.mean(axis=1)
        criteria2 = np.all(totalpower<=20, axis=1)&np.all(totalpower>=-10, axis=1)

        good_id = np.where(criteria1&criteria2)[0]#
        features = features[good_id]
        sleep_stages = sleep_stages[good_id]
        """

        features = np.sign(features)*np.log1p(np.abs(features))
        features2 = []
        for stage in stages:
            ids = sleep_stages==stage2num[stage]
            if ids.sum()>=minimum_epochs_per_stage:
                features2.append(features[ids].mean(axis=0))
            else:
                features2.append(np.zeros(features.shape[1])+np.nan)
        X.append(np.concatenate(features2))
        feature_files2.append(feature_files1[pid])
        ages.append(age)
        typeoftests.append(typeoftest)

        if not np.isnan(diag_end_id):
            features = features[:diag_end_id+1]
            sleep_stages = sleep_stages[:diag_end_id+1]
            features2 = []
            for stage in stages:
                ids = sleep_stages==stage2num[stage]
                if ids.sum()>=minimum_epochs_per_stage:
                    features2.append(features[ids].mean(axis=0))
                else:
                    features2.append(np.zeros(features.shape[1])+np.nan)
            X.append(np.concatenate(features2))
            feature_files2.append('diagnostic_part_in_splitnight_'+feature_files1[pid])
            ages.append(age)
            typeoftests.append(typeoftest)
    X = np.array(X)

    sio.savemat('all_features.mat', {'X':X,
        'FeaturePath':feature_files2,
        'feature_names':np.array(sum([[fn+'_'+stage for fn in feature_names] for stage in stages], [])),
        'ages':ages, 'typeoftests':typeoftests,
        })

    #training_info_folder = '/data/brain_age/mycode'
    training_info_folder = '/data/brain_age/brain_age_SplitNight_Robert/brain_age_model'
    Xnonan = sio.loadmat(os.path.join(training_info_folder, 'mgh_eeg_data.mat'))['Xtr']
    missing_stage = list(np.sum(np.isnan(X), axis=1)//feature_num_each_stage)
    K = 10  # number of patients without any missing stage to average
    X = impute_missing_stage(X, K, Xnonan=Xnonan)

    with open(os.path.join(training_info_folder, 'feature_normalizer_eeg.pickle'), 'rb') as f:
        standardizer = pickle.load(f, encoding='latin1')
    X = (X-standardizer.mean_)/standardizer.scale_

    ## testing (applying the trained model)

    model = load_model(os.path.join(training_info_folder, 'dnn_eeg_nomissingstage.pickle'), custom_objects={'loss':my_nobias_loss(1.)})
    BA = model.predict({'X':X, 'sample_weight':np.ones((len(X),1))})[:,0].flatten()

    ## generate output

    # add subjects with error
    notes = []
    for i, ba in enumerate(BA):
        if ba<1 or ba>120:
            notes.append('Possible EEG artifact')
        elif missing_stage[i]>=3:
            notes.append('Too many missing stages')
        else:
            notes.append('')
    feature_files2.extend(err_subjects)
    BA = np.r_[BA, [np.nan]*len(err_subjects)]
    missing_stage.extend([np.nan]*len(err_subjects))
    notes.extend(err_subject_reasons)
    ages.extend([np.nan]*len(err_subjects))
    typeoftests.extend([np.nan]*len(err_subjects))
    df = pd.DataFrame(data={'FeaturePath':feature_files2, 'BA':BA, 'CA':ages,
                            'TypeOfTest':typeoftests, 'NumMissingStage':missing_stage, 'Note':notes})

    # add MRN
    feature_path2mrn = {}
    for i in range(len(subject_files)):
        mrn = subject_files.MRN.iloc[i].lower().replace('x','').replace('-','').replace('_','').replace('.','')
        if '/' in mrn:
            if [len(x) for x in mrn.split('/')]==[3,2,2]:
                mrn = mrn.replace('/','')
            else:
                mrn = '-1'
        if len(mrn)!=7:
            mrn = -1
        else:
            mrn = int(mrn)
        feature_path2mrn[subject_files.feature_path.iloc[i]] = mrn
    df['MRN'] = [feature_path2mrn.get(os.path.join(feature_dir, ff), -1) for ff in feature_files2]

    # adjust BA
    df_adj = pd.read_csv('../BA_adjustment_bias.csv')
    df_adj.loc[0, 'CA_min'] = -np.inf
    df_adj.loc[len(df_adj)-1, 'CA_max'] = np.inf
    BA_adj = np.array(df['BA'].values)
    for i in range(len(df_adj)):
        ids = (df['CA'].values >= df_adj.CA_min.iloc[i])&(df['CA'].values <= df_adj.CA_max.iloc[i])
        BA_adj[ids] = df['BA'].values[ids] + df_adj.bias.iloc[i]
    df['BA_adj'] = BA_adj

    df = df[['MRN', 'CA', 'BA', 'BA_adj', 'TypeOfTest', 'NumMissingStage', 'Note', 'FeaturePath']]
    #df = df.sort_values('FeaturePath')
    import pdb;pdb.set_trace()
    df.to_excel('output_BA.xlsx', index=False)

