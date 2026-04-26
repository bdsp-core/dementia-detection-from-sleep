from dateutil.parser import parse
import datetime
from itertools import product
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
    
    
if __name__=='__main__':
    outcome = str(sys.argv[1])
    assert outcome in ['MCI+Dementia', 'Dementia']#, 'AD', 'PD', 'VaD']

    # Override these via the corresponding environment variables when running
    # against the BDSP-deID release on S3.
    criteria_path = os.environ.get("DEMENTIA_CRITERIA_PATH",
                                   "Dementia_MCI_table.xlsx")
    last_time_path = os.environ.get("DEMENTIA_STATUS_FINAL_PATH",
                                    "../shared_data/MGH/df_status_final.csv")
    features_path = os.environ.get("DEMENTIA_FEATURES_PATH",
                                   "../shared_data/MGH/to_be_used_features.csv")
    
    # get
    criteria_df = pd.read_excel(criteria_path)
    if outcome=='MCI+Dementia':
        criteria_df = criteria_df[np.in1d(criteria_df.Predicted_Stage, ['MCI','Dementia'])].reset_index(drop=True)
    elif outcome=='Dementia':
        criteria_df = criteria_df[criteria_df.Predicted_Stage=='Dementia'].reset_index(drop=True)
    elif outcome=='AD':
        criteria_df = criteria_df[(criteria_df.Predicted_Stage=='Dementia')&(criteria_df.Predicted_Disease=='AlzD')].reset_index(drop=True)
    elif outcome=='PD':
        criteria_df = criteria_df[(criteria_df.Predicted_Stage=='Dementia')&(criteria_df.Predicted_Disease=='PD')].reset_index(drop=True)
    elif outcome=='VaD':
        criteria_df = criteria_df[(criteria_df.Predicted_Stage=='Dementia')&(criteria_df.Predicted_Disease=='VaD')].reset_index(drop=True)
    
    # add Date column
    ids = ~pd.isna(criteria_df.Predicted_dT)
    criteria_df['Date'] = np.nan
    criteria_df['DateOfVisit'] = pd.to_datetime(criteria_df.DateOfVisit)
    criteria_df.loc[ids, 'Date'] = criteria_df.DateOfVisit[ids] - pd.TimedeltaIndex(criteria_df.Predicted_dT[ids], unit="D")
    
    # read last time info
    df_last_time = pd.read_csv(last_time_path)
    assert np.all(~pd.isna(df_last_time.PatientID))
    df_last_time = df_last_time.drop_duplicates('PatientID').reset_index(drop=True)
    df_last_time['last_encounter_date'] = pd.to_datetime(df_last_time.last_encounter_date)
    df_last_time['DeathDTS'] = pd.to_datetime(df_last_time.DeathDTS)
    assert len(df_last_time.dropna(subset=['last_encounter_date', 'DeathDTS'], how='all'))==len(df_last_time)
    # make sure DeathDTS>=last_encounter_date
    ids = (~pd.isna(df_last_time.DeathDTS)) & (df_last_time.DeathDTS.dt.date<df_last_time.last_encounter_date.dt.date)
    df_last_time.loc[ids, 'DeathDTS'] = df_last_time.last_encounter_date[ids]
    
    # read all pts
    df_pt = pd.read_csv(features_path)
    dovs = pd.to_datetime(df_pt.DateOfVisit).dt.to_pydatetime()
    
    # generate time to event
    cens_outcome = []
    time_outcome = []
    cens_death = []
    time_death = []
    stop_date = parse('2020/9/1')
    for i in tqdm(range(len(df_pt))):
        mrn = df_pt.MRN.iloc[i]
        
        # get sleep time
        sleep_time = dovs[i]
        
        # get outcome time
        icd_times = sorted(set(criteria_df.Date[criteria_df.MRN_key==mrn].dt.to_pydatetime()))
        #report_times = sorted(set(df_report.Date[df_report.MRN==mrn].dt.to_pydatetime()))
        outcome_time = None
        if len(icd_times)>0:
            outcome_time = min(icd_times)
        
        # get death time
        death_time = None
        last_encounter_time = None
        idx = np.where(df_last_time.PatientID==df_pt.PatientID.iloc[i])[0][0]
        #if len(ids)==1:
        if df_last_time.PatientStatusDSC.iloc[idx]=='Deceased':
            if not pd.isna(df_last_time.DeathDTS.iloc[idx]):
                death_time = df_last_time.DeathDTS.iloc[idx]
            else:
                death_time = df_last_time.last_encounter_date.iloc[idx]
        else:
            # they are alive, so use stop_date
            last_encounter_time = df_last_time.last_encounter_date.iloc[idx]#stop_date
        
        # decide death cens and time
        if death_time is None:
            cens_death.append(1)
            time_death.append((last_encounter_time-sleep_time).total_seconds()/3600./24./365.)
        else:
            cens_death.append(0)
            time_death.append((death_time-sleep_time).total_seconds()/3600./24./365.)
            
        # decide outcome cens and time
        if outcome_time is None:
            cens_outcome.append(1)
            if death_time is None:
                time_outcome.append((last_encounter_time-sleep_time).total_seconds()/3600./24./365.)
            else:
                time_outcome.append((death_time-sleep_time).total_seconds()/3600./24./365.)
        else:
            tt = (outcome_time-sleep_time).total_seconds()/3600./24./365.
            time_outcome.append(tt)
            cens_outcome.append(0 if tt>=0 else np.nan)
    
    df_pt['cens_outcome'] = cens_outcome
    df_pt['time_outcome'] = time_outcome
    df_pt['cens_death'] = cens_death
    df_pt['time_death'] = time_death
    df_time2event = df_pt[['PatientID', 'MRN', 'DateOfVisit',
                           'cens_outcome', 'time_outcome', 'cens_death', 'time_death']]
                                  
    # fix time_outcome and time_death <=0
    # for time_outcome<0 & cens_outcome=0, do not exist
    assert np.sum((df_time2event.time_outcome<0)&(df_time2event.cens_outcome==0))==0
    # for time_outcome<0 & cens_outcome=1, their latest time < study time, adjust time_outcome=0
    df_time2event.loc[(df_time2event.time_outcome<0)&(df_time2event.cens_outcome==1), 'time_outcome'] = 0
    # for time_outcome=0 & cens_outcome=0, outcome already at sleep study
    df_time2event.loc[(df_time2event.time_outcome==0)&(df_time2event.cens_outcome==0), 'cens_outcome'] = np.nan
    
    # for time_death<=0 & cens_death=0, do not exist
    assert np.sum((df_time2event.time_death<=0)&(df_time2event.cens_death==0))==0
    # for time_death<=0 & cens_death=1, their latest time < study time, adjust time_outcome=0
    df_time2event.loc[(df_time2event.time_death<=0)&(df_time2event.cens_death==1), 'time_death'] = 0

    df_time2event.to_excel('time2event_%s.xlsx'%outcome, index=False)
    
