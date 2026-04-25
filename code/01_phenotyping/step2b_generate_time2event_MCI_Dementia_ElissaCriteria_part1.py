from datetime import datetime
from dateutil.parser import parse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


## load data

data_path = '${DEMENTIA_DATA_ROOT}'

criteria_df = pd.read_csv(os.path.join(data_path, 'medical_data/study_criteria_table.csv'))
criteria_df = criteria_df[['FolderName', 'PatientID', 'EMPI', 'MRN', 'MRN_key',
       'Sex', 'Age', 'DateOfBirth', 'DateOfVisit', 'TypeOfTest', 'Path']]
print('criteria_df.shape =', criteria_df.shape)

ID_df = criteria_df[['FolderName','PatientID','EMPI','DateOfVisit']].drop_duplicates(ignore_index=True)
print('ID_df.shape =', ID_df.shape)

EDW_Enc_df  = pd.read_csv(os.path.join(data_path, 'medical_data/EDW_EncounterDiagnosis.csv'), encoding='utf-8')
EDW_Prob_df = pd.read_csv(os.path.join(data_path, 'medical_data/EDW_ProblemList.csv'), encoding='utf-8')
EDW_Med_df  = pd.read_csv(os.path.join(data_path, 'medical_data/EDW_MedicationDiagnosis.csv'), encoding='utf-8')
EDW_Prob_df = EDW_Prob_df[~EDW_Prob_df['DiagnosisNM'].astype(str).str.lower().str.contains('family history')].reset_index(drop=True)
EDW_Med_df = EDW_Med_df.dropna(subset=['MedicationDSC']).reset_index(drop=True)
EDW_Enc_df = EDW_Enc_df[~EDW_Enc_df['DiagnosisNM'].astype(str).str.lower().str.contains('family history')].reset_index(drop=True)
print('EDW_Enc_df.shape =', EDW_Enc_df.shape)
print('EDW_Prob_df.shape =', EDW_Prob_df.shape)
print('EDW_Med_df.shape =', EDW_Med_df.shape)

RPDR_Enc_df = pd.read_csv(os.path.join(data_path, 'medical_data/RPDR_Enc_All.csv'), encoding='utf-8')
RPDR_Med_df = pd.read_csv(os.path.join(data_path, 'medical_data/RPDR_Med_All.csv'), encoding='utf-8')
RPDR_Dia_df = pd.read_csv(os.path.join(data_path, 'medical_data/RPDR_Dia_All.csv'), encoding='utf-8')
#RPDR_Med_df = RPDR_Med_df.dropna(subset=['Medication']).reset_index(drop=True)
RPDR_Dia_df = RPDR_Dia_df.dropna(subset=['Diagnosis_Name']).reset_index(drop=True)
RPDR_Med_df = RPDR_Med_df.merge(RPDR_Dia_df[['Diagnosis_Name','Encounter_number']],on=['Encounter_number']).reset_index(drop=True)
RPDR_Dia_df = RPDR_Dia_df[~RPDR_Dia_df['Diagnosis_Name'].astype(str).str.lower().str.contains('family history')].reset_index(drop=True)
print('RPDR_Enc_df.shape =', RPDR_Enc_df.shape)
print('RPDR_Med_df.shape =', RPDR_Med_df.shape)
print('RPDR_Dia_df.shape =', RPDR_Dia_df.shape)

MoCA_EDW_df = pd.read_excel(os.path.join(data_path, 'medical_data/MoCA_EDW_Final.xlsx'))
MoCA_RPDR_df = pd.read_excel(os.path.join(data_path, 'medical_data/MoCA_RPDR_Final.xlsx'))
ids = pd.isna(MoCA_EDW_df.MoCADate); MoCA_EDW_df.loc[ids, 'MoCADate'] = MoCA_EDW_df.loc[ids, 'ContactDTSForNote']
ids = pd.isna(MoCA_RPDR_df.MoCADate); MoCA_RPDR_df.loc[ids, 'MoCADate'] = MoCA_RPDR_df.loc[ids, 'LMRNote_Date']
print('MoCA_EDW_df.shape =', MoCA_EDW_df.shape)
print('MoCA_RPDR_df.shape =', MoCA_RPDR_df.shape)

MMSE_EDW_df = pd.read_excel(os.path.join(data_path, 'medical_data/MMSE_EDW_Final.xlsx'))
MMSE_RPDR_df = pd.read_excel(os.path.join(data_path, 'medical_data/MMSE_RPDR_Final.xlsx'))
ids = pd.isna(MMSE_EDW_df.MMSEDate); MMSE_EDW_df.loc[ids, 'MMSEDate'] = MMSE_EDW_df.loc[ids, 'ContactDTSForNote']
print('MMSE_EDW_df.shape =', MMSE_EDW_df.shape)
print('MMSE_RPDR_df.shape =', MMSE_RPDR_df.shape)

CDR_EDW_df = pd.read_excel(os.path.join(data_path, 'medical_data/CDR_EDW_Final.xlsx'))
CDR_RPDR_df = pd.read_excel(os.path.join(data_path, 'medical_data/CDR_RPDR_Final.xlsx'))

Neuropsych_RPDR_df = pd.read_excel(os.path.join(data_path, 'medical_data/Neuropsychiatric Scores/Neuropsych_RPDR_Review.xlsx'))
Neuropsych_EDW_df = pd.read_excel(os.path.join(data_path, 'medical_data/Neuropsychiatric Scores/Neuropsych_EDW_Review.xlsx'))


## Exclusion Criteria

with open(os.path.join(data_path, 'medical_data/exclusion_regex'), 'r') as file:
    lines = file.read().split('\n')
lines.remove('[Ss]troke')   # do not exclude stroke, otherwise form competing risk with stroke
exclusion_regex = '|'.join(lines)

RPDR_Exclude_df = RPDR_Dia_df[RPDR_Dia_df['Diagnosis_Name'].astype(str).str.lower().str.contains(exclusion_regex)].reset_index(drop=True)
RPDR_Exclude_df = RPDR_Exclude_df.merge(criteria_df[['FolderName','EMPI','DateOfVisit']],on=['EMPI']).reset_index(drop=True)

EDW_Exclude_df = EDW_Enc_df[EDW_Enc_df['DiagnosisNM'].astype(str).str.lower().str.contains(exclusion_regex)].reset_index(drop=True)
EDW_Exclude_df = EDW_Exclude_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID']).reset_index(drop=True)

RPDR_Exclude_df['dT'] = (pd.to_datetime(RPDR_Exclude_df.DateOfVisit) - pd.to_datetime(RPDR_Exclude_df.Date)).dt.days
EDW_Exclude_df['dT'] = (pd.to_datetime(EDW_Exclude_df.DateOfVisit) - pd.to_datetime(EDW_Exclude_df.ContactDTSForEncounter)).dt.days

criteria_df['Exclusion_Enc'] = 0
criteria_df['Exclusion_ICD'] = ""
for index, row in tqdm(criteria_df.iterrows(), total=criteria_df.shape[0], position=0):
    df1 = EDW_Exclude_df[EDW_Exclude_df['FolderName'] == row['FolderName']]
    df2 = RPDR_Exclude_df[RPDR_Exclude_df['FolderName'] == row['FolderName']]
    criteria_df.at[index,'Exclusion_Enc'] = len(df2.Encounter_number.unique()) + len(df1.PatientEncounterID.unique())
    ICD_list = list(set(list(df1['DiagnosisNM']) + list(df2['Diagnosis_Name'])))
    if len(ICD_list) != 0:
        criteria_df.at[index,'Exclusion_ICD'] = ';'.join(ICD_list)


## get Neuropsychiatric Scores

criteria_df['CDR_Score'] = np.nan
criteria_df['CDR_Note'] = ""
criteria_df['CDR_dT'] = np.nan

Temp_CDR_RPDR_df = CDR_RPDR_df[CDR_RPDR_df.Correct=='Y'][['EMPI','LMRNote_Date','Comments','CDRText','CDRScore']].merge(ID_df)
Temp_CDR_RPDR_df['LMRNote_Date'] = Temp_CDR_RPDR_df['LMRNote_Date'].astype(str)
Temp_CDR_RPDR_df['DateOfVisit'] = Temp_CDR_RPDR_df['DateOfVisit'].astype(str)
Temp_CDR_RPDR_df['dT'] = (pd.to_datetime(Temp_CDR_RPDR_df.DateOfVisit) - pd.to_datetime(Temp_CDR_RPDR_df.LMRNote_Date)).dt.days
#Temp_CDR_RPDR_df = Temp_CDR_RPDR_df[Temp_CDR_RPDR_df['dT'] >= -365]
Temp_CDR_RPDR_df['abs_dT'] = np.abs(Temp_CDR_RPDR_df.dT)
Temp_CDR_RPDR_df = Temp_CDR_RPDR_df.sort_values('abs_dT').drop_duplicates('FolderName', ignore_index=True)
Temp_CDR_RPDR_df = Temp_CDR_RPDR_df[['FolderName','LMRNote_Date', 'CDRScore', 'Comments', 'dT','abs_dT', 'DateOfVisit']]
Temp_CDR_RPDR_df.columns = ['FolderName','CDR_Date', 'CDR_Score', 'CDR_Note', 'CDR_dT','abs_dT', 'DateOfVisit']

Temp_CDR_EDW_df = CDR_EDW_df[CDR_EDW_df.Correct=='Y'][['PatientID', 'ContactDTSForNote', 'NoteTXT', 'CDRScore']].merge(ID_df)
Temp_CDR_EDW_df['ContactDTSForNote'] = Temp_CDR_EDW_df['ContactDTSForNote'].astype(str)
Temp_CDR_EDW_df['DateOfVisit'] = Temp_CDR_EDW_df['DateOfVisit'].astype(str)
Temp_CDR_EDW_df['dT'] = (pd.to_datetime(Temp_CDR_EDW_df.DateOfVisit) - pd.to_datetime(Temp_CDR_EDW_df.ContactDTSForNote)).dt.days
#Temp_CDR_EDW_df = Temp_CDR_EDW_df[Temp_CDR_EDW_df['dT'] >= -365]
Temp_CDR_EDW_df['abs_dT'] = np.abs(Temp_CDR_EDW_df.dT)
Temp_CDR_EDW_df = Temp_CDR_EDW_df.sort_values('abs_dT').drop_duplicates('FolderName', ignore_index=True)
Temp_CDR_EDW_df = Temp_CDR_EDW_df[['FolderName','ContactDTSForNote', 'CDRScore', 'NoteTXT', 'dT','abs_dT', 'DateOfVisit']]
Temp_CDR_EDW_df.columns = ['FolderName','CDR_Date', 'CDR_Score', 'CDR_Note', 'CDR_dT','abs_dT', 'DateOfVisit']

Temp_CDR_df = pd.concat([Temp_CDR_RPDR_df,Temp_CDR_EDW_df], ignore_index=True)
Temp_CDR_df = Temp_CDR_df.sort_values(by = ['abs_dT']).drop_duplicates('FolderName', ignore_index=True)

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = Temp_CDR_df[Temp_CDR_df['FolderName'] == row['FolderName']]
    if len(df1 ) != 0:
        criteria_df.at[index,'CDR_Note'] =df1.iloc[0]['CDR_Note']
        criteria_df.at[index,'CDR_dT'] = df1.iloc[0]['CDR_dT']
        criteria_df.at[index,'CDR_Score'] = df1.iloc[0]['CDR_Score']
        
criteria_df['MMSE_Score'] = np.nan
criteria_df['MMSE_Note'] = ""
criteria_df['MMSE_dT'] = np.nan

Temp_MMSE_RPDR_df = MMSE_RPDR_df[MMSE_RPDR_df['Correct'] == 'Y'][['EMPI','LMRNote_Date','Comments','MMSEText','MMSEScore']].merge(ID_df)
Temp_MMSE_RPDR_df['DateOfVisit'] = Temp_MMSE_RPDR_df['DateOfVisit'].astype(str)
Temp_MMSE_RPDR_df['LMRNote_Date'] = Temp_MMSE_RPDR_df['LMRNote_Date'].astype(str)
Temp_MMSE_RPDR_df['dT'] = (pd.to_datetime(Temp_MMSE_RPDR_df.DateOfVisit) - pd.to_datetime(Temp_MMSE_RPDR_df.LMRNote_Date)).dt.days
#Temp_MMSE_RPDR_df = Temp_MMSE_RPDR_df[Temp_MMSE_RPDR_df['dT'] >= -365]
Temp_MMSE_RPDR_df['abs_dT'] = np.abs(Temp_MMSE_RPDR_df.dT)
Temp_MMSE_RPDR_df = Temp_MMSE_RPDR_df.sort_values('abs_dT').drop_duplicates('FolderName', ignore_index=True)
Temp_MMSE_RPDR_df = Temp_MMSE_RPDR_df[['FolderName','LMRNote_Date', 'MMSEScore', 'Comments', 'dT','abs_dT', 'DateOfVisit']]
Temp_MMSE_RPDR_df.columns = ['FolderName','MMSE_Date', 'MMSE_Score', 'MMSE_Note', 'MMSE_dT','abs_dT', 'DateOfVisit']

Temp_MMSE_EDW_df = MMSE_EDW_df[MMSE_EDW_df['Correct'] == 'Y'][['PatientID','ContactDTSForNote','NoteTXT','MMSEScore']].merge(ID_df)
Temp_MMSE_EDW_df['ContactDTSForNote'] = Temp_MMSE_EDW_df['ContactDTSForNote'].astype(str)
Temp_MMSE_EDW_df['DateOfVisit'] = Temp_MMSE_EDW_df['DateOfVisit'].astype(str)
Temp_MMSE_EDW_df['dT'] = (pd.to_datetime(Temp_MMSE_EDW_df.DateOfVisit) - pd.to_datetime(Temp_MMSE_EDW_df.ContactDTSForNote)).dt.days
#Temp_MMSE_EDW_df = Temp_MMSE_EDW_df[Temp_MMSE_EDW_df['dT'] >= -365]
Temp_MMSE_EDW_df['abs_dT'] = np.abs(Temp_MMSE_EDW_df.dT)
Temp_MMSE_EDW_df = Temp_MMSE_EDW_df.sort_values('abs_dT').drop_duplicates('FolderName', ignore_index=True)
Temp_MMSE_EDW_df = Temp_MMSE_EDW_df[['FolderName','ContactDTSForNote', 'MMSEScore', 'NoteTXT', 'dT','abs_dT', 'DateOfVisit']]
Temp_MMSE_EDW_df.columns = ['FolderName','MMSE_Date', 'MMSE_Score', 'MMSE_Note', 'MMSE_dT','abs_dT', 'DateOfVisit']

Temp_MMSE_df = pd.concat([Temp_MMSE_RPDR_df,Temp_MMSE_EDW_df], ignore_index=True)
Temp_MMSE_df = Temp_MMSE_df.sort_values(by = ['abs_dT']).drop_duplicates('FolderName', ignore_index=True)

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = Temp_MMSE_df[Temp_MMSE_df['FolderName'] == row['FolderName']]
    if len(df1 ) != 0:
        criteria_df.at[index,'MMSE_Note'] =df1.iloc[0]['MMSE_Note']
        criteria_df.at[index,'MMSE_dT'] = df1.iloc[0]['MMSE_dT']
        criteria_df.at[index,'MMSE_Score'] = df1.iloc[0]['MMSE_Score']

criteria_df['MoCA_Score'] = np.nan
criteria_df['MoCA_Note'] = ""
criteria_df['MoCA_dT'] = np.nan

Temp_MoCA_RPDR_df = MoCA_RPDR_df[MoCA_RPDR_df['Correct'] == 'Y'][['EMPI','LMRNote_Date','Comments','MoCAText','MoCAScore']].merge(ID_df)
Temp_MoCA_RPDR_df['DateOfVisit'] = Temp_MoCA_RPDR_df['DateOfVisit'].astype(str)
Temp_MoCA_RPDR_df['LMRNote_Date'] = Temp_MoCA_RPDR_df['LMRNote_Date'].astype(str)
Temp_MoCA_RPDR_df['dT'] = (pd.to_datetime(Temp_MoCA_RPDR_df.DateOfVisit) - pd.to_datetime(Temp_MoCA_RPDR_df.LMRNote_Date)).dt.days
#Temp_MoCA_RPDR_df = Temp_MoCA_RPDR_df[Temp_MoCA_RPDR_df['dT'] >= -365]
Temp_MoCA_RPDR_df['abs_dT'] = np.abs(Temp_MoCA_RPDR_df.dT)
Temp_MoCA_RPDR_df = Temp_MoCA_RPDR_df.sort_values('abs_dT').drop_duplicates('FolderName', ignore_index=True)
Temp_MoCA_RPDR_df = Temp_MoCA_RPDR_df[['FolderName','LMRNote_Date', 'MoCAScore', 'Comments', 'dT','abs_dT', 'DateOfVisit']]
Temp_MoCA_RPDR_df.columns = ['FolderName','MoCA_Date', 'MoCA_Score', 'MoCA_Note', 'MoCA_dT','abs_dT', 'DateOfVisit']

Temp_MoCA_EDW_df = MoCA_EDW_df[MoCA_EDW_df['Correct'] == 'Y'][['PatientID','ContactDTSForNote','NoteTXT','MoCAScore']].merge(ID_df)
Temp_MoCA_EDW_df['ContactDTSForNote'] = Temp_MoCA_EDW_df['ContactDTSForNote'].astype(str)
Temp_MoCA_EDW_df['DateOfVisit'] = Temp_MoCA_EDW_df['DateOfVisit'].astype(str)
Temp_MoCA_EDW_df['dT'] = (pd.to_datetime(Temp_MoCA_EDW_df.DateOfVisit) - pd.to_datetime(Temp_MoCA_EDW_df.ContactDTSForNote)).dt.days
#Temp_MoCA_EDW_df = Temp_MoCA_EDW_df[Temp_MoCA_EDW_df['dT'] >= -365]
Temp_MoCA_EDW_df['abs_dT'] = [abs(row['dT']) for index,row in Temp_MoCA_EDW_df.iterrows()]
Temp_MoCA_EDW_df = Temp_MoCA_EDW_df.sort_values('abs_dT').drop_duplicates('FolderName', ignore_index=True)
Temp_MoCA_EDW_df = Temp_MoCA_EDW_df[['FolderName','ContactDTSForNote', 'MoCAScore', 'NoteTXT', 'dT','abs_dT', 'DateOfVisit']]
Temp_MoCA_EDW_df.columns = ['FolderName','MoCA_Date', 'MoCA_Score', 'MoCA_Note', 'MoCA_dT','abs_dT', 'DateOfVisit']

Temp_MoCA_df = pd.concat([Temp_MoCA_RPDR_df,Temp_MoCA_EDW_df], ignore_index=True)
Temp_MoCA_df = Temp_MoCA_df.sort_values(by = ['abs_dT']).drop_duplicates('FolderName', ignore_index=True)

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = Temp_MoCA_df[Temp_MoCA_df['FolderName'] == row['FolderName']]
    if len(df1 ) != 0:
        criteria_df.at[index,'MoCA_Note'] = df1.iloc[0]['MoCA_Note']
        criteria_df.at[index,'MoCA_dT'] = df1.iloc[0]['MoCA_dT']
        criteria_df.at[index,'MoCA_Score'] = df1.iloc[0]['MoCA_Score']
        
        
Neuropsych_RPDR_df = Neuropsych_RPDR_df.merge(criteria_df[['FolderName','EMPI','DateOfVisit']],on=['EMPI'])
Neuropsych_RPDR_df['dT'] = (pd.to_datetime(Neuropsych_RPDR_df.DateOfVisit) - pd.to_datetime(Neuropsych_RPDR_df.LMRNote_Date)).dt.days
Neuropsych_RPDR_df['abs_dT'] = np.abs(Neuropsych_RPDR_df.dT)
Neuropsych_RPDR_df = Neuropsych_RPDR_df.sort_values(by=['abs_dT'], ignore_index=True)
Temp_Neuropsych_RPDR_df = Neuropsych_RPDR_df[['FolderName','dT', 'abs_dT','Comments']]
Temp_Neuropsych_RPDR_df.columns = ['FolderName','dT', 'abs_dT','Neuropsych_Note']

Neuropsych_EDW_df  = Neuropsych_EDW_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])
Neuropsych_EDW_df['dT'] = (pd.to_datetime(Neuropsych_EDW_df.DateOfVisit) - pd.to_datetime(Neuropsych_EDW_df.ContactDTSForNote)).dt.days
Neuropsych_EDW_df['abs_dT'] = np.abs(Neuropsych_EDW_df.dT)
Neuropsych_EDW_df = Neuropsych_EDW_df.sort_values(by=['abs_dT'], ignore_index=True)
Temp_Neuropsych_EDW_df = Neuropsych_EDW_df[['FolderName','dT', 'abs_dT','NoteTXT']]
Temp_Neuropsych_EDW_df.columns = ['FolderName','dT', 'abs_dT','Neuropsych_Note']

criteria_df['Neuropsych_Note'] = ""
criteria_df['Neuropsych_dT'] = np.nan
for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = Temp_Neuropsych_RPDR_df[Temp_Neuropsych_RPDR_df['FolderName'] == row['FolderName']]
    df2 = Temp_Neuropsych_EDW_df[Temp_Neuropsych_EDW_df['FolderName'] == row['FolderName']]
    if (len(df1) >= 1)|(len(df2) >= 1):
        criteria_df.at[index,'Neuropsych_Note'] = pd.concat([df1,df2]).sort_values(by=['abs_dT']).iloc[0]['Neuropsych_Note']
        criteria_df.at[index,'Neuropsych_dT'] = pd.concat([df1,df2]).sort_values(by=['abs_dT']).iloc[0]['dT']


## dementia criteria

criteria_df['Dementia_Enc'] = 0
criteria_df['Dementia_dT'] = np.nan
criteria_df['Dementia_ICD'] = ""
criteria_df['Dementia_Med'] = 0
criteria_df['Dementia_Prob'] = 0

with open(os.path.join(data_path, 'medical_data/dementia_regex'), 'r') as file:
    lines = file.read().split('\n')
dementia_regex = '|'.join(lines)

RPDR_Dementia_df = RPDR_Dia_df[RPDR_Dia_df['Diagnosis_Name'].astype(str).str.lower().str.contains(dementia_regex)]
EDW_Dementia_df = EDW_Enc_df[EDW_Enc_df['DiagnosisNM'].astype(str).str.lower().str.contains(dementia_regex)]

RPDR_Dementia_df = RPDR_Dementia_df.merge(criteria_df[['FolderName','EMPI','DateOfVisit']],on=['EMPI'])
EDW_Dementia_df = EDW_Dementia_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])

RPDR_Dementia_df['dT'] = (pd.to_datetime(RPDR_Dementia_df.DateOfVisit) - pd.to_datetime(RPDR_Dementia_df.Date)).dt.days
RPDR_Dementia_df['abs_dT'] = np.abs(RPDR_Dementia_df.dT)
RPDR_Dementia_df = RPDR_Dementia_df.sort_values(by=['abs_dT'], ignore_index=True)
#RPDR_Dementia_df = RPDR_Dementia_df[RPDR_Dementia_df['dT'] >= -365]

EDW_Dementia_df['dT'] = (pd.to_datetime(EDW_Dementia_df.DateOfVisit) - pd.to_datetime(EDW_Dementia_df.ContactDTSForEncounter)).dt.days
EDW_Dementia_df['abs_dT'] = np.abs(EDW_Dementia_df.dT)
EDW_Dementia_df = EDW_Dementia_df.sort_values(by=['abs_dT'], ignore_index=True)
#EDW_Dementia_df = EDW_Dementia_df[EDW_Dementia_df['dT'] >= -365]

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_Dementia_df[EDW_Dementia_df['FolderName'] == row['FolderName']]
    df2 = RPDR_Dementia_df[RPDR_Dementia_df['FolderName'] == row['FolderName']]
    criteria_df.at[index,'Dementia_Enc'] = len(df2.Encounter_number.unique()) + len(df1.PatientEncounterID.unique())
    ICD_list = list(set(list(df1['DiagnosisNM']) + list(df2['Diagnosis_Name'])))
    if len(ICD_list) != 0:
        criteria_df.at[index,'Dementia_ICD'] = ';'.join(ICD_list)
        criteria_df.at[index,'Dementia_dT'] = pd.concat([df1[['dT','abs_dT']],df2[['dT','abs_dT']]]).sort_values(by=['abs_dT']).iloc[0]['dT']


with open(os.path.join(data_path, 'medical_data/medications_regex.txt'), 'r') as file:
    lines = file.read().split('\n')
med_regex = '|'.join(lines)

RPDR_Medication_df = RPDR_Med_df[RPDR_Med_df['Medication'].astype(str).str.lower().str.contains(med_regex)].reset_index(drop=True)
EDW_Medication_df = EDW_Med_df[EDW_Med_df['MedicationDSC'].astype(str).str.lower().str.contains(med_regex)].reset_index(drop=True)

RPDR_Dementia_Medication_df = RPDR_Medication_df[RPDR_Medication_df['Diagnosis_Name'].str.contains(dementia_regex)]
EDW_Dementia_Medication_df = EDW_Medication_df[EDW_Medication_df['DiagnosisNM'].str.contains(dementia_regex)]

RPDR_Dementia_Medication_df = RPDR_Dementia_Medication_df.merge(criteria_df[['FolderName','EMPI','DateOfVisit']],on=['EMPI'])
EDW_Dementia_Medication_df = EDW_Dementia_Medication_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])

RPDR_Dementia_Medication_df['dT'] = (pd.to_datetime(RPDR_Dementia_Medication_df.DateOfVisit) - pd.to_datetime(RPDR_Dementia_Medication_df.Medication_Date)).dt.days
EDW_Dementia_Medication_df['dT'] = (pd.to_datetime(EDW_Dementia_Medication_df.DateOfVisit) - pd.to_datetime(EDW_Dementia_Medication_df.ContactDTSForEncounter)).dt.days
#RPDR_Dementia_Medication_df = RPDR_Dementia_Medication_df[RPDR_Dementia_Medication_df['dT'] >= -365]
#EDW_Dementia_Medication_df = EDW_Dementia_Medication_df[EDW_Dementia_Medication_df['dT'] >= -365]

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_Dementia_Medication_df[EDW_Dementia_Medication_df['FolderName'] == row['FolderName']]
    df2 = RPDR_Dementia_Medication_df[RPDR_Dementia_Medication_df['FolderName'] == row['FolderName']]
    if (len(df2) >= 1)|(len(df1) >= 1):
        criteria_df.at[index,'Dementia_Med'] =len(df2.Encounter_number.unique()) + len(df1.PatientEncounterID.unique())

EDW_Dementia_Prob_df = EDW_Prob_df[EDW_Prob_df.DiagnosisNM.astype(str).str.lower().str.contains(dementia_regex) & (EDW_Prob_df.ProblemStatusDSC=='Active') ]
EDW_Dementia_Prob_df = EDW_Dementia_Prob_df.dropna(subset=['DiagnosisDTS']).reset_index(drop=True)
#EDW_Dementia_Prob_df = EDW_Dementia_Prob_df[EDW_Dementia_Prob_df['DiagnosisDTS'].str.contains('-')].reset_index(drop=True)

EDW_Dementia_Prob_df['DiagnosisDTS'] = EDW_Dementia_Prob_df['DiagnosisDTS'].astype(str)
EDW_Dementia_Prob_df = EDW_Dementia_Prob_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']], on=['PatientID'])
EDW_Dementia_Prob_df['dT'] = (pd.to_datetime(EDW_Dementia_Prob_df.DateOfVisit) - pd.to_datetime(EDW_Dementia_Prob_df.DiagnosisDTS)).dt.days
#EDW_Dementia_Prob_df = EDW_Dementia_Prob_df[EDW_Dementia_Prob_df['dT'] >= -365]

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_Dementia_Prob_df[EDW_Dementia_Prob_df['FolderName'] == row['FolderName']]
    if len(df1) >= 1:
        criteria_df.at[index,'Dementia_Prob'] =len(df1.drop_duplicates())


## MCI criteria

criteria_df['MCI_Enc'] = 0
criteria_df['MCI_dT'] = np.nan
criteria_df['MCI_ICD'] = ""
criteria_df['MCI_Med'] = 0
criteria_df['MCI_Prob'] = 0

with open(os.path.join(data_path, 'medical_data/MCI_regex.txt'), 'r') as file:
    lines = file.read().split('\n')
MCI_regex = '|'.join(lines)

RPDR_MCI_df = RPDR_Dia_df[RPDR_Dia_df['Diagnosis_Name'].astype(str).str.lower().str.contains(MCI_regex)]
EDW_MCI_df = EDW_Enc_df[EDW_Enc_df['DiagnosisNM'].astype(str).str.lower().str.contains(MCI_regex)]
RPDR_MCI_df = RPDR_MCI_df.merge(criteria_df[['FolderName','EMPI','DateOfVisit']],on=['EMPI'])
EDW_MCI_df = EDW_MCI_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])

RPDR_MCI_df['dT'] = (pd.to_datetime(RPDR_MCI_df.DateOfVisit) - pd.to_datetime(RPDR_MCI_df.Date)).dt.days
EDW_MCI_df['dT'] = (pd.to_datetime(EDW_MCI_df.DateOfVisit) - pd.to_datetime(EDW_MCI_df.ContactDTSForEncounter)).dt.days
RPDR_MCI_df['abs_dT'] = np.abs(RPDR_MCI_df.dT)
EDW_MCI_df['abs_dT'] = np.abs(EDW_MCI_df.dT)
EDW_MCI_df = EDW_MCI_df.sort_values(by=['abs_dT'], ignore_index=True)
RPDR_MCI_df = RPDR_MCI_df.sort_values(by=['abs_dT'], ignore_index=True)
#RPDR_MCI_df = RPDR_MCI_df[RPDR_MCI_df['dT'] >= -365]
#EDW_MCI_df = EDW_MCI_df[EDW_MCI_df['dT'] >= -365]

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_MCI_df[EDW_MCI_df['FolderName'] == row['FolderName']]
    df2 = RPDR_MCI_df[RPDR_MCI_df['FolderName'] == row['FolderName']]
    criteria_df.at[index,'MCI_Enc'] = len(df2.Encounter_number.unique()) + len(df1.PatientEncounterID.unique())
    ICD_list = list(set(list(df1['DiagnosisNM']) + list(df2['Diagnosis_Name'])))
    if len(ICD_list) != 0:
        criteria_df.at[index,'MCI_ICD'] = ';'.join(ICD_list)
        criteria_df.at[index,'MCI_dT'] = pd.concat([df1[['dT','abs_dT']],df2[['dT','abs_dT']]]).sort_values(by=['abs_dT']).iloc[0]['dT']
        
RPDR_MCI_Medication_df = RPDR_Medication_df[RPDR_Medication_df['Diagnosis_Name'].astype(str).str.lower().str.contains(MCI_regex)]
EDW_MCI_Medication_df = EDW_Medication_df[EDW_Medication_df['DiagnosisNM'].astype(str).str.lower().str.contains(MCI_regex)]
RPDR_MCI_Medication_df = RPDR_MCI_Medication_df.merge(criteria_df[['FolderName','EMPI','DateOfVisit']],on=['EMPI'])
EDW_MCI_Medication_df = EDW_MCI_Medication_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])

RPDR_MCI_Medication_df['dT'] = (pd.to_datetime(RPDR_MCI_Medication_df.DateOfVisit) - pd.to_datetime(RPDR_MCI_Medication_df.Medication_Date)).dt.days
EDW_MCI_Medication_df['dT'] = (pd.to_datetime(EDW_MCI_Medication_df.DateOfVisit) - pd.to_datetime(EDW_MCI_Medication_df.ContactDTSForEncounter)).dt.days
#RPDR_MCI_Medication_df = RPDR_MCI_Medication_df[RPDR_MCI_Medication_df['dT'] >= -365]
#EDW_MCI_Medication_df = EDW_MCI_Medication_df[EDW_MCI_Medication_df['dT'] >= -365]

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_MCI_Medication_df[EDW_MCI_Medication_df['FolderName'] == row['FolderName']]
    df2 = RPDR_MCI_Medication_df[RPDR_MCI_Medication_df['FolderName'] == row['FolderName']]
    if (len(df2) >= 1)|(len(df1) >= 1):
        criteria_df.at[index,'MCI_Med'] = len(df2.Encounter_number.unique()) + len(df1.PatientEncounterID.unique())

EDW_MCI_Prob_df = EDW_Prob_df[EDW_Prob_df['DiagnosisNM'].astype(str).str.lower().str.contains(MCI_regex) & (EDW_Prob_df['ProblemStatusDSC'] =='Active') ].reset_index(drop=True)
EDW_MCI_Prob_df = EDW_MCI_Prob_df.dropna(subset=['DiagnosisDTS']).reset_index(drop=True)
EDW_MCI_Prob_df = EDW_MCI_Prob_df[EDW_MCI_Prob_df['DiagnosisDTS'].str.contains('-') ]
EDW_MCI_Prob_df['DiagnosisDTS'] = EDW_MCI_Prob_df['DiagnosisDTS'].astype(str)
EDW_MCI_Prob_df= EDW_MCI_Prob_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])
EDW_MCI_Prob_df['dT'] = (pd.to_datetime(EDW_MCI_Prob_df.DateOfVisit) - pd.to_datetime(EDW_MCI_Prob_df.DiagnosisDTS)).dt.days
#EDW_MCI_Prob_df = EDW_MCI_Prob_df[EDW_MCI_Prob_df['dT'] >= -365]

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    Study_EDW_MCI_Prob_df = EDW_MCI_Prob_df[EDW_MCI_Prob_df['FolderName'] == row['FolderName']]
    if len(Study_EDW_MCI_Prob_df) >= 1:
        criteria_df.at[index,'MCI_Prob'] =len(Study_EDW_MCI_Prob_df.drop_duplicates())


## Symptomatic Criteria

criteria_df['Symptomatic_Enc'] = 0
criteria_df['Symptomatic_dT'] = np.nan
criteria_df['Symptomatic_ICD'] = ""

with open(os.path.join(data_path, 'medical_data/symptomatic_regex.txt'), 'r') as file:
    lines = file.read().split('\n')
symptomatic_regex = '|'.join(lines)

RPDR_Symptomatic_df = RPDR_Dia_df[RPDR_Dia_df['Diagnosis_Name'].astype(str).str.lower().str.contains(symptomatic_regex)]
EDW_Symptomatic_df = EDW_Enc_df[EDW_Enc_df['DiagnosisNM'].astype(str).str.lower().str.contains(symptomatic_regex)]

RPDR_Symptomatic_df = RPDR_Symptomatic_df.merge(criteria_df[['FolderName','EMPI','DateOfVisit']],on=['EMPI'])
EDW_Symptomatic_df = EDW_Symptomatic_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])

RPDR_Symptomatic_df['dT'] = (pd.to_datetime(RPDR_Symptomatic_df.DateOfVisit) - pd.to_datetime(RPDR_Symptomatic_df.Date)).dt.days
EDW_Symptomatic_df['dT'] = (pd.to_datetime(EDW_Symptomatic_df.DateOfVisit) - pd.to_datetime(EDW_Symptomatic_df.ContactDTSForEncounter)).dt.days
RPDR_Symptomatic_df['abs_dT'] = np.abs(RPDR_Symptomatic_df.dT)
EDW_Symptomatic_df['abs_dT'] = np.abs(EDW_Symptomatic_df.dT)
EDW_Symptomatic_df = EDW_Symptomatic_df.sort_values(by=['abs_dT'], ignore_index=True)
RPDR_Symptomatic_df = RPDR_Symptomatic_df.sort_values(by=['abs_dT'], ignore_index=True)
#RPDR_Symptomatic_df = RPDR_Symptomatic_df[RPDR_Symptomatic_df['dT'] >= -365]
#EDW_Symptomatic_df = EDW_Symptomatic_df[EDW_Symptomatic_df['dT'] >= -365]

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_Symptomatic_df[EDW_Symptomatic_df['FolderName'] == row['FolderName']]
    df2 = RPDR_Symptomatic_df[RPDR_Symptomatic_df['FolderName'] == row['FolderName']]
    criteria_df.at[index,'Symptomatic_Enc'] = len(df2.Encounter_number.unique()) + len(df1.PatientEncounterID.unique())
    ICD_list = list(set(list(df1['DiagnosisNM']) + list(df2['Diagnosis_Name'])))
    if len(ICD_list) != 0:
        criteria_df.at[index,'Symptomatic_ICD'] = ';'.join(ICD_list)
        criteria_df.at[index,'Symptomatic_dT'] = pd.concat([df1[['dT','abs_dT']],df2[['dT','abs_dT']]]).sort_values(by=['abs_dT']).iloc[0]['dT']
        
        
## Dementia Subtypes

# AlzD
alz_regex = 'alzh'

criteria_df['AlzD_Enc'] = 0
criteria_df['AlzD_dT'] = np.nan
criteria_df['AlzD_ICD'] = ""
criteria_df['AlzD_Prob'] = 0

RPDR_AlzD_df = RPDR_Dia_df[RPDR_Dia_df['Diagnosis_Name'].astype(str).str.lower().str.contains(alz_regex)].reset_index(drop=True)
EDW_AlzD_df = EDW_Enc_df[EDW_Enc_df['DiagnosisNM'].astype(str).str.lower().str.contains(alz_regex)].reset_index(drop=True)
RPDR_AlzD_df = RPDR_AlzD_df.merge(criteria_df[['FolderName','EMPI','DateOfVisit']],on=['EMPI'])
EDW_AlzD_df = EDW_AlzD_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])
RPDR_AlzD_df['dT'] = (pd.to_datetime(RPDR_AlzD_df.DateOfVisit) - pd.to_datetime(RPDR_AlzD_df.Date)).dt.days
EDW_AlzD_df['dT'] = (pd.to_datetime(EDW_AlzD_df.DateOfVisit) - pd.to_datetime(EDW_AlzD_df.ContactDTSForEncounter)).dt.days
EDW_AlzD_df = EDW_AlzD_df.sort_values(by=['dT'], ignore_index=True)
RPDR_AlzD_df = RPDR_AlzD_df.sort_values(by=['dT'], ignore_index=True)

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_AlzD_df[EDW_AlzD_df['FolderName'] == row['FolderName']]
    df2 = RPDR_AlzD_df[RPDR_AlzD_df['FolderName'] == row['FolderName']]
    criteria_df.at[index,'AlzD_Enc'] = len(df2.Encounter_number.unique()) + len(df1.PatientEncounterID.unique())
    ICD_list = list(set(list(df1['DiagnosisNM']) + list(df2['Diagnosis_Name'])))
    if len(ICD_list) != 0:
        criteria_df.at[index,'AlzD_ICD'] = ';'.join(ICD_list)
        criteria_df.at[index,'AlzD_dT'] = pd.concat([df1[['dT']],df2[['dT']]]).sort_values(by=['dT']).iloc[0]['dT']
        
EDW_AlzD_Prob_df = EDW_Prob_df[EDW_Prob_df['DiagnosisNM'].astype(str).str.lower().str.contains(alz_regex) & (EDW_Prob_df['ProblemStatusDSC'] =='Active') ].reset_index(drop=True)
EDW_AlzD_Prob_df = EDW_AlzD_Prob_df.dropna(subset=['DiagnosisDTS']).reset_index(drop=True)
EDW_AlzD_Prob_df = EDW_AlzD_Prob_df[EDW_AlzD_Prob_df['DiagnosisDTS'].str.contains('-') ].reset_index(drop=True)
EDW_AlzD_Prob_df['DiagnosisDTS'] = EDW_AlzD_Prob_df['DiagnosisDTS'].astype(str)
EDW_AlzD_Prob_df= EDW_AlzD_Prob_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])
for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_AlzD_Prob_df[EDW_AlzD_Prob_df['FolderName'] == row['FolderName']]
    if len(df1) >= 1:
        criteria_df.at[index,'AlzD_Prob'] =len(df1.drop_duplicates())

# VaD
vad_regex = 'vasc.{0,14}demen'

criteria_df['VaD_Enc'] = 0
criteria_df['VaD_dT'] = np.nan
criteria_df['VaD_ICD'] = ""
criteria_df['VaD_Prob'] = 0
RPDR_VaD_df = RPDR_Dia_df[RPDR_Dia_df['Diagnosis_Name'].astype(str).str.lower().str.contains(vad_regex)].reset_index(drop=True)
EDW_VaD_df = EDW_Enc_df[EDW_Enc_df['DiagnosisNM'].astype(str).str.lower().str.contains(vad_regex)].reset_index(drop=True)
RPDR_VaD_df = RPDR_VaD_df.merge(criteria_df[['FolderName','EMPI','DateOfVisit']],on=['EMPI'])
EDW_VaD_df = EDW_VaD_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])
RPDR_VaD_df['dT'] = (pd.to_datetime(RPDR_VaD_df.DateOfVisit) - pd.to_datetime(RPDR_VaD_df.Date)).dt.days
EDW_VaD_df['dT'] = (pd.to_datetime(EDW_VaD_df.DateOfVisit) - pd.to_datetime(EDW_VaD_df.ContactDTSForEncounter)).dt.days
EDW_VaD_df = EDW_VaD_df.sort_values(by=['dT'], ignore_index=True)
RPDR_VaD_df = RPDR_VaD_df.sort_values(by=['dT'], ignore_index=True)

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_VaD_df[EDW_VaD_df['FolderName'] == row['FolderName']]
    df2 = RPDR_VaD_df[RPDR_VaD_df['FolderName'] == row['FolderName']]
    criteria_df.at[index,'VaD_Enc'] = len(df2.Encounter_number.unique()) + len(df1.PatientEncounterID.unique())
    ICD_list = list(set(list(df1['DiagnosisNM']) + list(df2['Diagnosis_Name'])))
    if len(ICD_list) != 0:
        criteria_df.at[index,'VaD_ICD'] = ';'.join(ICD_list)
        criteria_df.at[index,'VaD_dT'] = pd.concat([df1[['dT']],df2[['dT']]]).sort_values(by=['dT']).iloc[0]['dT']
        
EDW_VaD_Prob_df = EDW_Prob_df[RPDR_Dia_df['Diagnosis_Name'].astype(str).str.lower().str.contains(vad_regex) & (EDW_Prob_df['ProblemStatusDSC'] =='Active')].reset_index(drop=True)
EDW_VaD_Prob_df = EDW_VaD_Prob_df.dropna(subset=['DiagnosisDTS']).reset_index(drop=True)
EDW_VaD_Prob_df = EDW_VaD_Prob_df[EDW_VaD_Prob_df['DiagnosisDTS'].str.contains('-') ].reset_index(drop=True)
EDW_VaD_Prob_df['DiagnosisDTS'] = EDW_VaD_Prob_df['DiagnosisDTS'].astype(str)
EDW_VaD_Prob_df= EDW_VaD_Prob_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])
for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_VaD_Prob_df[EDW_VaD_Prob_df['FolderName'] == row['FolderName']]
    if len(df1) >= 1:
        criteria_df.at[index,'VaD_Prob'] =len(df1.drop_duplicates())

# FTD

ftd_regex = 'frontotemp.{0,14}demen'

criteria_df['FTD_Enc'] = 0
criteria_df['FTD_dT'] = np.nan
criteria_df['FTD_ICD'] = ""
criteria_df['FTD_Prob'] = 0
RPDR_FTD_df = RPDR_Dia_df[RPDR_Dia_df['Diagnosis_Name'].astype(str).str.lower().str.contains(ftd_regex)].reset_index(drop=True)
EDW_FTD_df = EDW_Enc_df[EDW_Enc_df['DiagnosisNM'].astype(str).str.lower().str.contains(ftd_regex)].reset_index(drop=True)
RPDR_FTD_df = RPDR_FTD_df.merge(criteria_df[['FolderName','EMPI','DateOfVisit']],on=['EMPI'])
EDW_FTD_df = EDW_FTD_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])
RPDR_FTD_df['dT'] = (pd.to_datetime(RPDR_FTD_df.DateOfVisit) - pd.to_datetime(RPDR_FTD_df.Date)).dt.days
EDW_FTD_df['dT'] = (pd.to_datetime(EDW_FTD_df.DateOfVisit) - pd.to_datetime(EDW_FTD_df.ContactDTSForEncounter)).dt.days
EDW_FTD_df = EDW_FTD_df.sort_values(by=['dT'], ignore_index=True)
RPDR_FTD_df = RPDR_FTD_df.sort_values(by=['dT'], ignore_index=True)

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_FTD_df[EDW_FTD_df['FolderName'] == row['FolderName']]
    Study_RPDR_FTD_df = RPDR_FTD_df[RPDR_FTD_df['FolderName'] == row['FolderName']]
    criteria_df.at[index,'FTD_Enc'] = len(Study_RPDR_FTD_df.Encounter_number.unique()) + len(df1.PatientEncounterID.unique())
    ICD_list = list(set(list(df1['DiagnosisNM']) + list(Study_RPDR_FTD_df['Diagnosis_Name'])))
    if len(ICD_list) != 0:
        criteria_df.at[index,'FTD_ICD'] = ';'.join(ICD_list)
        criteria_df.at[index,'FTD_dT'] = pd.concat([df1[['dT']],Study_RPDR_FTD_df[['dT']]]).sort_values(by=['dT']).iloc[0]['dT']
        
EDW_FTD_Prob_df = EDW_Prob_df[EDW_Prob_df['DiagnosisNM'].astype(str).str.lower().str.contains(ftd_regex) & (EDW_Prob_df['ProblemStatusDSC'] =='Active') ].reset_index(drop=True)
EDW_FTD_Prob_df = EDW_FTD_Prob_df.dropna(subset=['DiagnosisDTS']).reset_index(drop=True)
EDW_FTD_Prob_df = EDW_FTD_Prob_df[EDW_FTD_Prob_df['DiagnosisDTS'].str.contains('-') ]
EDW_FTD_Prob_df['DiagnosisDTS'] = EDW_FTD_Prob_df['DiagnosisDTS'].astype(str)
EDW_FTD_Prob_df= EDW_FTD_Prob_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])
for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    Study_EDW_FTD_Prob_df = EDW_FTD_Prob_df[EDW_FTD_Prob_df['FolderName'] == row['FolderName']]
    if len(Study_EDW_FTD_Prob_df) >= 1:
        criteria_df.at[index,'FTD_Prob'] =len(Study_EDW_FTD_Prob_df.drop_duplicates())

# DLB
dlb_regex = 'demen.{0,23}lewy.{0,20}bod|lewy.{0,20}bod.{0,20}demen'

criteria_df['DLB_Enc'] = 0
criteria_df['DLB_dT'] = np.nan
criteria_df['DLB_ICD'] = ""
criteria_df['DLB_Prob'] = 0
RPDR_DLB_df = RPDR_Dia_df[RPDR_Dia_df['Diagnosis_Name'].astype(str).str.lower().str.contains(dlb_regex)].reset_index(drop=True)
EDW_DLB_df = EDW_Enc_df[EDW_Enc_df['DiagnosisNM'].astype(str).str.lower().str.contains(dlb_regex)].reset_index(drop=True)
RPDR_DLB_df = RPDR_DLB_df.merge(criteria_df[['FolderName','EMPI','DateOfVisit']],on=['EMPI'])
EDW_DLB_df = EDW_DLB_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])
RPDR_DLB_df['dT'] = (pd.to_datetime(RPDR_DLB_df.DateOfVisit) - pd.to_datetime(RPDR_DLB_df.Date)).dt.days
EDW_DLB_df['dT'] = (pd.to_datetime(EDW_DLB_df.DateOfVisit) - pd.to_datetime(EDW_DLB_df.ContactDTSForEncounter)).dt.days
EDW_DLB_df = EDW_DLB_df.sort_values(by=['dT'], ignore_index=True)
RPDR_DLB_df = RPDR_DLB_df.sort_values(by=['dT'], ignore_index=True)

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_DLB_df[EDW_DLB_df['FolderName'] == row['FolderName']]
    df2 = RPDR_DLB_df[RPDR_DLB_df['FolderName'] == row['FolderName']]
    criteria_df.at[index,'DLB_Enc'] = len(df2.Encounter_number.unique()) + len(df1.PatientEncounterID.unique())
    ICD_list = list(set(list(df1['DiagnosisNM']) + list(df2['Diagnosis_Name'])))
    if len(ICD_list) != 0:
        criteria_df.at[index,'DLB_ICD'] = ';'.join(ICD_list)
        criteria_df.at[index,'DLB_dT'] = pd.concat([df1[['dT']],df2[['dT']]]).sort_values(by=['dT']).iloc[0]['dT']
        
EDW_DLB_Prob_df = EDW_Prob_df[EDW_Prob_df['DiagnosisNM'].astype(str).str.lower().str.contains(dlb_regex) & (EDW_Prob_df['ProblemStatusDSC'] =='Active') ].reset_index(drop=True)
EDW_DLB_Prob_df = EDW_DLB_Prob_df.dropna(subset=['DiagnosisDTS']).reset_index(drop=True)
EDW_DLB_Prob_df = EDW_DLB_Prob_df[EDW_DLB_Prob_df['DiagnosisDTS'].str.contains('-') ].reset_index(drop=True)
EDW_DLB_Prob_df['DiagnosisDTS'] = EDW_DLB_Prob_df['DiagnosisDTS'].astype(str)
EDW_DLB_Prob_df= EDW_DLB_Prob_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])
for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_DLB_Prob_df[EDW_DLB_Prob_df['FolderName'] == row['FolderName']]
    if len(df1) >= 1:
        criteria_df.at[index,'DLB_Prob'] =len(df1.drop_duplicates())


# PD
pd_regex = 'parki.{0,24}dise'

criteria_df['PD_Enc'] = 0
criteria_df['PD_dT'] = np.nan
criteria_df['PD_ICD'] = ""
criteria_df['PD_Prob'] = 0
RPDR_PD_df = RPDR_Dia_df[RPDR_Dia_df['Diagnosis_Name'].astype(str).str.lower().str.contains(pd_regex)].reset_index(drop=True)
EDW_PD_df = EDW_Enc_df[EDW_Enc_df['DiagnosisNM'].astype(str).str.lower().str.contains(pd_regex)].reset_index(drop=True)
RPDR_PD_df = RPDR_PD_df.merge(criteria_df[['FolderName','EMPI','DateOfVisit']],on=['EMPI'])
EDW_PD_df = EDW_PD_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])
RPDR_PD_df['dT'] = (pd.to_datetime(RPDR_PD_df.DateOfVisit) - pd.to_datetime(RPDR_PD_df.Date)).dt.days
EDW_PD_df['dT'] = (pd.to_datetime(EDW_PD_df.DateOfVisit) - pd.to_datetime(EDW_PD_df.ContactDTSForEncounter)).dt.days
EDW_PD_df = EDW_PD_df.sort_values(by=['dT'], ignore_index=True)
RPDR_PD_df = RPDR_PD_df.sort_values(by=['dT'], ignore_index=True)

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_PD_df[EDW_PD_df['FolderName'] == row['FolderName']]
    df2 = RPDR_PD_df[RPDR_PD_df['FolderName'] == row['FolderName']]
    criteria_df.at[index,'PD_Enc'] = len(df2.Encounter_number.unique()) + len(df1.PatientEncounterID.unique())
    ICD_list = list(set(list(df1['DiagnosisNM']) + list(df2['Diagnosis_Name'])))
    if len(ICD_list) != 0:
        criteria_df.at[index,'PD_ICD'] = ';'.join(ICD_list)
        criteria_df.at[index,'PD_dT'] = pd.concat([df1[['dT']],df2[['dT']]]).sort_values(by=['dT']).iloc[0]['dT']

EDW_PD_Prob_df = EDW_Prob_df[EDW_Prob_df['DiagnosisNM'].astype(str).str.lower().str.contains(pd_regex) & (EDW_Prob_df['ProblemStatusDSC'] =='Active') ].reset_index(drop=True)
EDW_PD_Prob_df = EDW_PD_Prob_df.dropna(subset=['DiagnosisDTS']).reset_index(drop=True)
EDW_PD_Prob_df = EDW_PD_Prob_df[EDW_PD_Prob_df['DiagnosisDTS'].str.contains('-') ].reset_index(drop=True)
EDW_PD_Prob_df['DiagnosisDTS'] = EDW_PD_Prob_df['DiagnosisDTS'].astype(str)
EDW_PD_Prob_df= EDW_PD_Prob_df.merge(criteria_df[['FolderName','PatientID','DateOfVisit']],on=['PatientID'])
for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    df1 = EDW_PD_Prob_df[EDW_PD_Prob_df['FolderName'] == row['FolderName']]
    if len(df1) >= 1:
        criteria_df.at[index,'PD_Prob'] =len(df1.drop_duplicates())

label_columns = [
#'True_Stage',
#'True_Certainty',
#'True_Diease',
#'Note',
'Predicted_Stage',
'Predicted_Certainty',
'Predicted_dT',
'Predicted_Disease']
label_df = pd.DataFrame(data=np.zeros((len(criteria_df),4))+np.nan, columns=label_columns)
criteria_df = pd.concat([label_df,criteria_df],axis=1) 
#criteria_df.to_excel('study_medical_table_V3.xlsx',index=False)

criteria_df['Predicted_Stage'] = None
criteria_df['Predicted_Disease'] = None
criteria_df['Predicted_Certainty'] = None

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    if row['Age'] < 50:
        criteria_df.at[index,'Predicted_Stage'] = 'Excluded'
    if row['Exclusion_Enc'] != 0:
        criteria_df.at[index,'Predicted_Stage'] = 'Excluded'
        
for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0):
    if row['Predicted_Stage'] =='Excluded':
        continue
    if row['CDR_Score'] >= 1:
        if (np.isnan(row['Predicted_dT'])) | (abs(row['Predicted_dT']) > abs(row['CDR_dT'])): 
            criteria_df.at[index,'Predicted_dT'] = row['CDR_dT']
            criteria_df.at[index,'Predicted_Stage'] = 'Dementia'
    if row['MMSE_Score'] < 25:
        if criteria_df.at[index,'Predicted_Stage']  == 'Dementia':
            criteria_df.at[index,'Predicted_Certainty'] = 'High'
        if (np.isnan(criteria_df.at[index,'Predicted_dT'])) | (abs(criteria_df.at[index,'Predicted_dT']) > abs(row['MMSE_dT'])): 
            criteria_df.at[index,'Predicted_dT'] = row['MMSE_dT']
            criteria_df.at[index,'Predicted_Stage'] = 'Dementia'
            if criteria_df.at[index,'Predicted_Certainty'] == None:
                criteria_df.at[index,'Predicted_Certainty'] = 'Low'
    if row['MoCA_Score'] < 20:
        if criteria_df.at[index,'Predicted_Stage'] == 'Dementia':
            criteria_df.at[index,'Predicted_Certainty'] = 'High'
        if (np.isnan(criteria_df.at[index,'Predicted_dT'])) | (abs(criteria_df.at[index,'Predicted_dT']) > abs(row['MoCA_dT'])): 
            criteria_df.at[index,'Predicted_dT'] = row['MoCA_dT']
            criteria_df.at[index,'Predicted_Stage'] = 'Dementia'
            if row['Predicted_Certainty'] == None:
                criteria_df.at[index,'Predicted_Certainty'] = 'Low'
    if (row['Dementia_Enc'] >= 1) & ((row['Symptomatic_Enc'] >= 1)|(row['Dementia_Med'])):
        if criteria_df.at[index,'Predicted_Stage'] == 'Dementia':
            criteria_df.at[index,'Predicted_Certainty'] = 'High'
        if (np.isnan(criteria_df.at[index,'Predicted_dT'])) | (abs(criteria_df.at[index,'Predicted_dT']) > abs(row['Dementia_dT'])): 
            criteria_df.at[index,'Predicted_dT'] = row['Dementia_dT']
            criteria_df.at[index,'Predicted_Stage'] = 'Dementia'
            if criteria_df.at[index,'Predicted_Certainty'] == None:
                criteria_df.at[index,'Predicted_Certainty'] = 'Low'

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0,leave=True):
    if row['Predicted_Stage'] =='Excluded':
        continue
    if row['CDR_Score'] == 0.5:
        if (np.isnan(criteria_df.at[index,'Predicted_dT'])) | (abs(criteria_df.at[index,'Predicted_dT']) > abs(row['CDR_dT'])): 
            criteria_df.at[index,'Predicted_dT'] = row['CDR_dT']
            criteria_df.at[index,'Predicted_Stage'] = 'MCI'
    if (row['MMSE_Score'] <= 27) & (row['MMSE_Score'] >= 25):
        if criteria_df.at[index,'Predicted_Stage']  == 'MCI':
            criteria_df.at[index,'Predicted_Certainty'] = 'High'
        if (np.isnan(criteria_df.at[index,'Predicted_dT'])) | (abs(criteria_df.at[index,'Predicted_dT']) > abs(row['MMSE_dT'])): 
            criteria_df.at[index,'Predicted_dT'] = row['MMSE_dT']
            criteria_df.at[index,'Predicted_Stage'] = 'MCI'
            if criteria_df.at[index,'Predicted_Certainty'] == None:
                criteria_df.at[index,'Predicted_Certainty'] = 'Low'
    if  (row['MoCA_Score'] >= 20)& (row['MoCA_Score'] <= 26):
        if criteria_df.at[index,'Predicted_Stage']  == 'MCI':
            criteria_df.at[index,'Predicted_Certainty'] = 'High'
        if (np.isnan(criteria_df.at[index,'Predicted_dT'])) | (abs(criteria_df.at[index,'Predicted_dT']) > abs(row['MoCA_dT'])): 
            criteria_df.at[index,'Predicted_dT'] = row['MoCA_dT']
            criteria_df.at[index,'Predicted_Stage'] = 'MCI'
            if criteria_df.at[index,'Predicted_Certainty'] == None:
                criteria_df.at[index,'Predicted_Certainty'] = 'Low'
    if (row['MCI_Enc'] >= 1) & ((row['Symptomatic_Enc'] >= 1)|(row['MCI_Med'])):
        if criteria_df.at[index,'Predicted_Stage']  == 'MCI':
            criteria_df.at[index,'Predicted_Certainty'] = 'High'
        if (np.isnan(criteria_df.at[index,'Predicted_dT'])) | (abs(criteria_df.at[index,'Predicted_dT']) > abs(row['MCI_dT'])): 
            criteria_df.at[index,'Predicted_dT'] = row['MCI_dT']
            criteria_df.at[index,'Predicted_Stage'] = 'MCI'
            if criteria_df.at[index,'Predicted_Certainty'] == None:
                criteria_df.at[index,'Predicted_Certainty'] = 'Low'

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0,leave=True):
    if row['Predicted_Stage'] is None:
        if row['Symptomatic_Enc'] >= 1:
            criteria_df.at[index,'Predicted_dT'] = row['Symptomatic_dT']
            criteria_df.at[index,'Predicted_Stage'] = 'Symptomatic'
        if row['MCI_Enc'] >= 1:
            criteria_df.at[index,'Predicted_dT'] = row['Symptomatic_dT']
            criteria_df.at[index,'Predicted_Stage'] = 'Symptomatic'
        if row['Dementia_Enc'] >= 1:
            criteria_df.at[index,'Predicted_dT'] = row['Symptomatic_dT']
            criteria_df.at[index,'Predicted_Stage'] = 'Symptomatic'

for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0,leave=True):
    if row['Predicted_Stage'] is None:
        criteria_df.at[index,'Predicted_Stage'] = 'No Dementia'
        
for index, row in tqdm(criteria_df.iterrows(),total=criteria_df.shape[0],position=0,leave=True):
    Enc_num = 0
    dT = float('inf')
    predicted_disease = row['Predicted_Disease']
    for disease in ['AlzD','VaD','FTD','DLB','PD']:
        if not np.isnan(row[disease + '_Enc']):
            if ((abs(row[disease + '_dT']) < abs(dT))):
                predicted_disease = disease
                Enc_num = row[disease + '_Enc']
                dT = row[disease + '_dT']
    if Enc_num != 0:
        criteria_df.at[index,'Predicted_Disease'] = predicted_disease

"""
#MCI Correction
for index, row in tqdm(criteria_df[(criteria_df['Predicted_Stage']=='Dementia')].iterrows(),total=criteria_df[criteria_df['Predicted_Stage']=='Dementia'].shape[0],position=0,leave=True):
    if np.isnan(row['Low_CDR_Score']) & np.isnan(row['High_MMSE_Score']) & np.isnan(row['High_MoCA_Score']):
        continue
    if row['Low_CDR_Score'] <= 0.5:
        if (np.isnan(row['Predicted_dT'])) | (row['Predicted_dT'] > row['Low_CDR_dT']): 
            criteria_df.at[index,'Predicted_dT'] = row['Low_CDR_dT']
            criteria_df.at[index,'Predicted_Stage'] = 'MCI'
    if row['High_MMSE_Score'] > 25:
        if (np.isnan(criteria_df.at[index,'Predicted_dT'])) | (criteria_df.at[index,'Predicted_dT'] > row['High_MMSE_dT']): 
            criteria_df.at[index,'Predicted_dT'] = row['High_MMSE_dT']
            criteria_df.at[index,'Predicted_Stage'] = 'MCI'
    if row['High_MoCA_Score'] > 20:
        if (np.isnan(criteria_df.at[index,'Predicted_dT'])) | (criteria_df.at[index,'Predicted_dT'] > row['High_MoCA_dT']): 
            criteria_df.at[index,'Predicted_dT'] = row['High_MoCA_dT']
            criteria_df.at[index,'Predicted_Stage'] = 'MCI'
    #if row['MCI_Enc'] >= 1:
    #    if criteria_df.at[index,'Predicted_dT'] > row['MCI_dT']:
    #        criteria_df.at[index,'Predicted_dT'] = row['MCI_dT']
    #        criteria_df.at[index,'Predicted_Stage'] = 'MCI'
"""
    
print('Dementia Cases:',len(criteria_df[criteria_df['Predicted_Stage'] =='Dementia' ] ))
print('MCI Cases:',len(criteria_df[criteria_df['Predicted_Stage'] =='MCI' ] ))
print('Symptomatic Cases:',len(criteria_df[criteria_df['Predicted_Stage'] =='Symptomatic' ] ))
print('Non-Dementia Cases:',len(criteria_df[criteria_df['Predicted_Stage'] =='No Dementia' ] ))
print('Excluded Cases:',len(criteria_df[criteria_df['Predicted_Stage'] =='Excluded' ] ))

print('High Certainty Dementia Cases:',len(criteria_df[(criteria_df['Predicted_Stage'] =='Dementia' ) & (criteria_df['Predicted_Certainty'] =='High')]))
print('Low Certainty Dementia Cases:',len(criteria_df[(criteria_df['Predicted_Stage'] =='Dementia' ) & (criteria_df['Predicted_Certainty'] =='Low')] ))
print('High Certainty MCI Cases:',len(criteria_df[(criteria_df['Predicted_Stage'] =='MCI' ) & (criteria_df['Predicted_Certainty'] =='High')]))
print('Low Certainty MCI Cases:',len(criteria_df[(criteria_df['Predicted_Stage'] =='MCI' ) & (criteria_df['Predicted_Certainty'] =='Low')] ))

print('AlzD:',len(criteria_df[criteria_df['Predicted_Disease'] == 'AlzD']))
print('VaD:',len(criteria_df[criteria_df['Predicted_Disease'] == 'VaD']))
print('FTD:',len(criteria_df[criteria_df['Predicted_Disease'] == 'FTD']))
print('DLB:',len(criteria_df[criteria_df['Predicted_Disease'] == 'DLB']))
print('PDD:',len(criteria_df[(criteria_df['Predicted_Disease'] == 'PD') & ((criteria_df['Predicted_Stage'] == 'MCI')|(criteria_df['Predicted_Stage'] == 'Dementia'))]))

criteria_df.to_excel('Dementia_MCI_table_Elissa.xlsx',index=False)


"""
Dementia Cases: 838
MCI Cases: 1208
Symptomatic Cases: 3076
Non-Dementia Cases: 8585
Excluded Cases: 9284
High Certainty Dementia Cases: 251
Low Certainty Dementia Cases: 576
High Certainty MCI Cases: 371
Low Certainty MCI Cases: 813
AlzD: 412
VaD: 105
FTD: 41
DLB: 41
PDD: 123
"""
