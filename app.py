#!/usr/bin/env python
# coding: utf-8

# In[2]:
import glob2 as glob2

import glob2 as glob2
import pandas as pd
import numpy as np
import time
import os
import re
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import pickle
from sklearn.metrics import classification_report
from sklearn import metrics
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler 
import plotly.graph_objects as go
from plotly import tools
import plotly.offline as py
import plotly.express as px
from bokeh.plotting import figure
import xlsxwriter

# In[3]:


import streamlit as st

# Глобальные переменные

progress_bar = st.progress(0)
progress_text = st.empty()

# Функции для выравнивания временного ряда

def upsampl_time (d1, delta):
  # upsamling - повышение разрешения каждые 10 секунд, вставляемая сторока заполняется NAN по другим признакам

    d1['Date/Time'] = pd.to_datetime(d1['Date/Time'])
    d1 = d1.set_index('Date/Time')
    d1 = d1.resample(delta).asfreq()
    d1 = d1.reset_index()
    return d1
def downsampl_time (d1, delta):
  # downsamling - понижение разрешения до 10 секунд, при схлопывании по другим признакам берется максимальное значение

    d1['Date/Time'] = pd.to_datetime(d1['Date/Time'])
    d1 = d1.set_index('Date/Time')
    d1 = d1.resample(delta).max()
    d1 = d1.reset_index()
    return d1


def sec_for_min(d1):
  # проставление "обрезанных" секунд при наличии данных для них 

    i=0
    t_old = pd.to_datetime(0)
    delta = timedelta(seconds = 10)
    for t in d1["Date/Time"]:
        if (t_old.time().second >= 50) and (t_old.time().second <= 59):
                delta = timedelta(seconds = 2)
        else: delta = timedelta(seconds = 10)
        #print(t_old, delta)
        if i !=0:
            if t.time().minute == t_old.time().minute:
                t = (t_old + delta)
                d1["Date/Time"][i]=t
        i += 1
        t_old = t
    return d1

#def walk_dir(path="\Downloads\0-bur\f"):
def walk_dir(uploaded_file):
    #st.write('ok', path)
    data_f = pd.DataFrame()
     # Датасет, который будем строить

    COLUMN_NAMES = ['Date/Time', 'StuckPipe', 'TVD', 'DEPT', 'CDEPTH', 'HDTH', 'BPOS', 'HKLD', 'STOR', 'FLWI', 'RPM',
                    'SPPA', 'ECD', 'DLS', 'INCL', 'AZIM', 'GR', 'APRS', 'BVEL', 'RIG_STATE', 'Stick_Slip_Ratio',
                    'StickPercentage', 'ESD']
    data = pd.DataFrame(columns=COLUMN_NAMES)
    # Переименуем дублирующиеся колонки

    recolumn_1 = {
        'APRS_RT': 'APRS',
        'AZIM_CONT_RT': 'AZIM',
        'AZIMQ': 'AZIM',
        'ECD_RT': 'ECD',
        'ECD_ECO_RT': 'ECD',
        'DEPTH': 'DEPT',
        'INCL_CONT_RT': 'INCL',
        'INCLQ': 'INCL',
        'DHAP_DH_ECO_RT': 'APRS',
        'GR_ARC_RT': 'GR',
        'GRMA_ECO_RT': 'GR',
        'RIG_STATE_EXACT': 'RIG_STATE',
        'STICKRATIO': 'Stick_Slip_Ratio',
        'STICK_RT': 'StickPercentage'
    }
    recolumn_2 = {
        'APRS_RT': 'APRS',
        'AZIM_CONT_RT': 'AZIM',
        'AZIMQ': 'AZIM',
        'ECD_RT': 'ECD',
        'ECD_ECO_RT': 'ECD',
        'DEPTH': 'DEPT',
        'INCL_CONT_RT': 'INCL',
        'INCLQ': 'INCL',
        'DHAP_DH_ECO_RT': 'APRS',
        'GR_ARC_RT': 'GR',
        'GRMA_ECO_RT': 'GR',
        'RIG_STATE_EXACT': 'RIG_STATE',
        'STICKRATIO': 'StickPercentage',
        'STICK_RT': 'StickPercentage'
    }
    #st.write('ok2', path)
    n=0
#    for dirname, _, filenames in os.walk(path):
    for filenames in [uploaded_file]:
        #st.write('Файл из директории: ', dir)
        for file in [filenames]:
            st.write('Подождите! Идет предобработка файла: ', file.name)
            progress_bar.progress(5)
            progress_text.text(f"Progress: {5}%")
            f_txt = os.path.split(dir_upl)[1].split(".")[1]
            if f_txt == 'xlsx' or f_txt == 'xls':
                data_f = pd.read_excel(file, skiprows=[1, 2])
            if f_txt == 'csv':
                data_f = pd.read_csv(file, skiprows=[1, 2])

            data_f['file'] = file
            if 'Stick_Slip_Ratio' in data_f.columns:
                data_f.rename(columns=recolumn_2, inplace=True)
            else:
                data_f.rename(columns=recolumn_1, inplace=True)
            hole = str(file).split('.')[0]  # добавляем номер буровой

            progress_bar.progress(15)
            progress_text.text(f"Progress: {15}%")

            data_f['Date/Time'] = pd.to_datetime(data_f['Date/Time']) #, format="%Y-%d-%m, %H:%M")
            data_f['TimeU'] = data_f['Date/Time'].apply(lambda t: int(t.timestamp()))
            max_delta=data_f['Date/Time'].diff().max().seconds
            min_delta=data_f['Date/Time'].diff().min().seconds
            if (max_delta == 60) and (min_delta ==0):
            #print ('60')
                data_f = sec_for_min(data_f)
                data_f = downsampl_time (data_f, '10S')
            elif (max_delta == 30) and (min_delta ==0):
            #print =  ( '30')
                data_f = sec_for_min(data_f)
                data_f = upsampl_time (data_f, '10S')
            elif max_delta == 10:
            #print ( '10')
                data_f = downsampl_time (data_f, '10S')
            else: 
            #print ( '2')        
                data_f = downsampl_time (data_f, '10S')
 
            progress_bar.progress(20)
            progress_text.text(f"Progress: {20}%")           
            
            # обрабатываем значения -999.25
            for col in data_f.columns[data_f.dtypes == 'float64']:
                # data_f[col] = data_f[col].apply(lambda n: np.nan if n == -999.25 else n).astype('float')
                data_f.loc[data_f[col] == -999.25, col] = np.nan
 
            progress_bar.progress(25)
            progress_text.text(f"Progress: {25}%")

            # приводим к общим единицам измерения
            data_f['hole'] = hole
            #hole_measures = pd.read_excel('Units & schema.xlsx', skiprows=[1])
            # Списки скважин, где требуется замена параметров
            #kft_lbf_list = hole_measures.loc[hole_measures['STOR'] == 'kft.lbf']['Скважина'].tolist()  # Крутящий момент STOR
            #bar_list = hole_measures.loc[hole_measures['SPPA'] == 'bar']['Скважина'].tolist()  # Давление SPPA и APRS
            #kPa_list = hole_measures.loc[hole_measures['SPPA'] == 'kPa']['Скважина'].tolist()  # Давление SPPA и APRS
            #if hole in kft_lbf_list:
            #    data_f['STOR'] = data_f['STOR'].astype('float').apply(lambda x: x * 0.737 if x != -999.25 else np.nan)
            #if hole in bar_list:
            #    data_f['APRS'] = data_f['APRS'].astype('float').apply(lambda x: x * 1.02 if x != -999.25 else np.nan)
            #    data_f['SPPA'] = data_f['SPPA'].astype('float').apply(lambda x: x * 1.02 if x != -999.25 else np.nan)
            #if hole in kPa_list:
            #    data_f['APRS'] = data_f['APRS'].astype('float').apply(lambda x: x * 0.01 if x != -999.25 else np.nan)
            #    data_f['SPPA'] = data_f['SPPA'].astype('float').apply(lambda x: x * 0.01 if x != -999.25 else np.nan)

            if ('220ST3' not in hole) & ('220ST4' not in hole):
                #st.write('Подождите1!' )
                data_f['STOR'] = data_f['STOR'].astype('float').apply(lambda x: x * 0.737 if x != -999.25 else np.nan)
            if ('204' in hole) | ('210ST' in hole) | ('216ST' in hole) | ('247' in hole):
                data_f['APRS'] = data_f['APRS'].astype('float').apply(lambda x: x * 1.02 if x != -999.25 else np.nan)
                data_f['SPPA'] = data_f['SPPA'].astype('float').apply(lambda x: x * 1.02 if x != -999.25 else np.nan)
            if ('241' in hole) | ('243' in hole) | ('278' in hole) | ('279' in hole):
                data_f['APRS'] = data_f['APRS'].astype('float').apply(lambda x: x * 0.01 if x != -999.25 else np.nan)
                data_f['SPPA'] = data_f['SPPA'].astype('float').apply(lambda x: x * 0.01 if x != -999.25 else np.nan)
                #st.write('Подождите2!' )
            # Замена NaN
            data_f['StuckPipe'] = data_f['StuckPipe'].fillna(0)

            data_f[['DEPT', 'HDTH']] = data_f[['DEPT', 'HDTH']].interpolate()
            data_f['DDEPT'] = data_f['DEPT'].diff()  # Новый признак изменения к предыдущему глубины долота

            data_f.loc[((data_f['RIG_STATE'].isna()) & (data_f['FLWI'] > 15) & (data_f['RPM'] > 15) & (
                        data_f['DDEPT'] > 0)), 'RIG_STATE'] = 1
            data_f.loc[((data_f['RIG_STATE'].isna()) & (data_f['FLWI'] <= 15) & (data_f['RPM'] <= 15) & (
                        data_f['DDEPT'] == 0)), 'RIG_STATE'] = 2
            data_f.loc[((data_f['RIG_STATE'].isna()) & (data_f['FLWI'] > 15) & (data_f['RPM'] <= 15) & (
                        data_f['DDEPT'] > 0)), 'RIG_STATE'] = 4
            data_f.loc[((data_f['RIG_STATE'].isna()) & (data_f['FLWI'] <= 15) & (data_f['RPM'] > 15) & (
                        data_f['DDEPT'] > 0)), 'RIG_STATE'] = 5
            data_f.loc[((data_f['RIG_STATE'].isna()) & (data_f['FLWI'] <= 15) & (data_f['RPM'] <= 15) & (
                        data_f['DDEPT'] > 0)), 'RIG_STATE'] = 6
            data_f.loc[((data_f['RIG_STATE'].isna()) & (data_f['FLWI'] > 15) & (data_f['RPM'] > 15) & (
                        data_f['DDEPT'] < 0)), 'RIG_STATE'] = 7
            data_f.loc[((data_f['RIG_STATE'].isna()) & (data_f['FLWI'] > 15) & (data_f['RPM'] <= 15) & (
                        data_f['DDEPT'] < 0)), 'RIG_STATE'] = 8
            data_f.loc[((data_f['RIG_STATE'].isna()) & (data_f['FLWI'] <= 15) & (data_f['RPM'] > 15) & (
                        data_f['DDEPT'] < 0)), 'RIG_STATE'] = 9
            data_f.loc[((data_f['RIG_STATE'].isna()) & (data_f['FLWI'] <= 15) & (data_f['RPM'] <= 15) & (
                        data_f['DDEPT'] < 0)), 'RIG_STATE'] = 10
            data_f.loc[((data_f['RIG_STATE'].isna()) & (data_f['FLWI'] > 15) & (data_f['RPM'] > 15) & (
                        data_f['DDEPT'] == 0)), 'RIG_STATE'] = 11
            data_f.loc[((data_f['RIG_STATE'].isna()) & (data_f['FLWI'] > 15) & (data_f['RPM'] <= 15) & (
                        data_f['DDEPT'] == 0)), 'RIG_STATE'] = 12
            data_f.loc[((data_f['RIG_STATE'].isna()) & (data_f['FLWI'] <= 15) & (data_f['RPM'] > 15) & (
                        data_f['DDEPT'] == 0)), 'RIG_STATE'] = 13
            data_f['RIG_STATE'] = data_f['RIG_STATE'].fillna(method='ffill')

            data_f.loc[(data_f['RIG_STATE'] == 2 & data_f['BPOS'].isna()), 'BPOS'] = 1
            data_f.loc[(data_f['RIG_STATE'] != 2 & data_f['BPOS'].isna()), 'BPOS'] = -data_f['DDEPT'].cumsum()

            data_f['BPOS'] = data_f['BPOS'].interpolate()
            data_f['BPOS'] = data_f['BPOS'].fillna(method='ffill')
            data_f['BPOS'] = data_f['BPOS'].fillna(method='bfill')

            data_f['HKLD'].fillna((data_f['HKLD'].rolling(10).mean()), inplace=True)
            data_f['STOR'].fillna((data_f['STOR'].rolling(10).mean()), inplace=True)

            FLWI_0_RIG = [2, 5, 6, 9, 10, 13, 14]  # Список статусов RIG_STATE, при которых FLWI = 0
            RPM_0_RIG = [2, 4, 6, 8, 10, 12, 14]  # То же для RPM = 0
            MOVE_RIG = [1, 3, 4, 5, 6, 7, 8, 9, 10]  # Список статусов RIG_STATE, при которых крюк двигается

            data_f.loc[((data_f['FLWI'].isna()) & (data_f['RIG_STATE'].isin(FLWI_0_RIG))), 'FLWI'] = 0
            data_f['FLWI'] = data_f['FLWI'].interpolate()

            data_f.loc[((data_f['RPM'].isna()) & (data_f['RIG_STATE'].isin(RPM_0_RIG))), 'RPM'] = 0
            data_f['RPM'] = data_f['RPM'].interpolate()

            data_f['SPPA'] = data_f['SPPA'].interpolate()

            data_f['ECD'] = data_f['ECD'].interpolate()
            data_f['ECD'] = data_f['ECD'].fillna(method='ffill')
            data_f['ECD'] = data_f['ECD'].fillna(method='bfill')

            data_f['DLS'] = data_f['DLS'].interpolate()
            data_f['DLS'] = data_f['DLS'].fillna(method='bfill')
            data_f['DLS'] = data_f['DLS'].fillna(method='ffill')

            if 'INCL' in data_f.columns:
                data_f[['INCL', 'AZIM']] = data_f[['INCL', 'AZIM']].interpolate()
                data_f[['INCL', 'AZIM']] = data_f[['INCL', 'AZIM']].fillna(method='ffill')
            if 'APRS' in data_f.columns:
                data_f['APRS'] = data_f['APRS'].interpolate()
                data_f['APRS'] = data_f['APRS'].fillna(method='ffill')
                data_f['APRS'] = data_f['APRS'].fillna(method='bfill')

            data_f.loc[((data_f['BVEL'].isna()) & (data_f['RIG_STATE'].isin(MOVE_RIG))), 'BVEL'] = data_f.loc[((data_f['BVEL'].isna()) & (data_f['RIG_STATE'].isin(MOVE_RIG)))]['DDEPT']/360
            data_f['BVEL'] = data_f['BVEL'].fillna(0)

            data_f['ESD'] = data_f['ESD'].interpolate()
            data_f['ESD'] = data_f['ESD'].fillna(method='ffill')
            data_f['ESD'] = data_f['ESD'].fillna(method='bfill')

            if 'GR' in data_f.columns:
                data_f['GR'] = data_f['GR'].interpolate()
                data_f['GR'] = data_f['GR'].fillna(method='bfill')
                data_f['GR'] = data_f['GR'].fillna(method='ffill')


            # Объединим Stick_Slip_Ratio и StickPercentage, пропущенные значения заполним 0
            data_f['Stick'] = 0
            data_f.loc[((data_f['Stick_Slip_Ratio']==0)&(data_f['StickPercentage']!=0)), 'Stick'] = data_f['StickPercentage']
            data_f.loc[((data_f['Stick_Slip_Ratio'] != 0) & (data_f['StickPercentage'] == 0)), 'Stick'] = data_f['Stick_Slip_Ratio']
            data_f['Stick'] = data_f['Stick'].fillna(0)

            data_f = data_f.fillna(0)
            # for col in data_f.columns:
                # data_f[col] = data_f[col].fillna(data_f[col].rolling(1000).mean())
                # data_f[col] = data_f[col].fillna(0)

            progress_bar.progress(50)
            progress_text.text(f"Progress: {50}%")

           # новые признаки по временному сдвигу
            data_f['DDEPT_1'] = data_f['DEPT'].diff() / data_f['DEPT'].shift()
            data_f['DDEPT_3'] = data_f['DEPT'].diff(3) / data_f['DEPT'].shift(3)
            data_f['DDEPT_6'] = data_f['DEPT'].diff(6) / data_f['DEPT'].shift(6)
            data_f['DDEPT_12'] = data_f['DEPT'].diff(12) / data_f['DEPT'].shift(12)
            data_f['DDEPT_18'] = data_f['DEPT'].diff(18) / data_f['DEPT'].shift(18)
            data_f['FDEPT_1'] = data_f['DEPT'].diff(-1) / data_f['DEPT'].shift(-1)
            data_f['FDEPT_3'] = data_f['DEPT'].diff(-3) / data_f['DEPT'].shift(-3)

            data_f['DBPOS'] = data_f['BPOS'].diff()  # Новый признак изменения к предыдущему высоты крюка
            data_f['DBPOS_1'] = data_f['BPOS'].diff() / data_f['BPOS'].shift()
            data_f['DBPOS_3'] = data_f['BPOS'].diff(3) / data_f['BPOS'].shift(3)
            data_f['DBPOS_6'] = data_f['BPOS'].diff(6) / data_f['BPOS'].shift(6)
            data_f['DBPOS_12'] = data_f['BPOS'].diff(12) / data_f['BPOS'].shift(12)
            data_f['DBPOS_18'] = data_f['BPOS'].diff(18) / data_f['BPOS'].shift(18)
            data_f['FBPOS_1'] = data_f['BPOS'].diff(-1) / data_f['BPOS'].shift(-1)
            data_f['FBPOS_3'] = data_f['BPOS'].diff(-3) / data_f['BPOS'].shift(-3)

            data_f['DHKLD'] = data_f['HKLD'].diff()  # Новый признак изменения к предыдущему веса крюка
            data_f['DHKLD_1'] = data_f['HKLD'].diff() / data_f['HKLD'].shift()
            data_f['DHKLD_3'] = data_f['HKLD'].diff(3) / data_f['HKLD'].shift(3)
            data_f['DHKLD_6'] = data_f['HKLD'].diff(6) / data_f['HKLD'].shift(6)
            data_f['DHKLD_12'] = data_f['HKLD'].diff(12) / data_f['HKLD'].shift(12)
            data_f['DHKLD_18'] = data_f['HKLD'].diff(18) / data_f['HKLD'].shift(18)
            data_f['FHKLD_1'] = data_f['HKLD'].diff(-1) / data_f['HKLD'].shift(-1)
            data_f['FHKLD_3'] = data_f['HKLD'].diff(-3) / data_f['HKLD'].shift(-3)

            data_f['DSPPA'] = data_f['SPPA'].diff()
            data_f['DSPPA_1'] = data_f['SPPA'].diff() / data_f['SPPA'].shift()
            data_f['DSPPA_3'] = data_f['SPPA'].diff(3) / data_f['SPPA'].shift(3)
            data_f['DSPPA_6'] = data_f['SPPA'].diff(6) / data_f['SPPA'].shift(6)
            data_f['DSPPA_12'] = data_f['SPPA'].diff(12) / data_f['SPPA'].shift(12)
            data_f['DSPPA_18'] = data_f['SPPA'].diff(18) / data_f['SPPA'].shift(18)
            data_f['FSPPA_1'] = data_f['SPPA'].diff(-1) / data_f['SPPA'].shift(-1)
            data_f['FSPPA_3'] = data_f['SPPA'].diff(-3) / data_f['SPPA'].shift(-3)

            data_f['DRPM'] = data_f['RPM'].diff()
            data_f['DRPM_1'] = data_f['RPM'].diff() / data_f['RPM'].shift()
            data_f['DRPM_3'] = data_f['RPM'].diff(3) / data_f['RPM'].shift(3)
            data_f['DRPM_6'] = data_f['RPM'].diff(6) / data_f['RPM'].shift(6)
            data_f['DRPM_12'] = data_f['RPM'].diff(12) / data_f['RPM'].shift(12)
            data_f['DRPM_18'] = data_f['RPM'].diff(18) / data_f['RPM'].shift(18)
            data_f['FRPM_1'] = data_f['RPM'].diff(-1) / data_f['RPM'].shift(-1)
            data_f['FRPM_3'] = data_f['RPM'].diff(-3) / data_f['RPM'].shift(-3)

            data_f['DBVEL'] = data_f['BVEL'].diff()
            data_f['DBVEL_1'] = data_f['BVEL'].diff() / data_f['BVEL'].shift()
            data_f['DBVEL_3'] = data_f['BVEL'].diff(3) / data_f['BVEL'].shift(3)
            data_f['DBVEL_6'] = data_f['BVEL'].diff(6) / data_f['BVEL'].shift(6)
            data_f['DBVEL_12'] = data_f['BVEL'].diff(12) / data_f['BVEL'].shift(12)
            data_f['DBVEL_18'] = data_f['BVEL'].diff(18) / data_f['BVEL'].shift(18)
            data_f['FBVEL_1'] = data_f['BVEL'].diff(-1) / data_f['BVEL'].shift(-1)
            data_f['FBVEL_3'] = data_f['BVEL'].diff(-3) / data_f['BVEL'].shift(-3)

            data_f['DSTOR_1'] = data_f['STOR'].diff() / data_f['STOR'].shift()
            data_f['DSTOR_3'] = data_f['STOR'].diff(3) / data_f['STOR'].shift(3)
            data_f['DSTOR_6'] = data_f['STOR'].diff(6) / data_f['STOR'].shift(6)
            data_f['DSTOR_12'] = data_f['STOR'].diff(12) / data_f['STOR'].shift(12)
            data_f['DSTOR_18'] = data_f['STOR'].diff(18) / data_f['STOR'].shift(18)
            data_f['FSTOR_1'] = data_f['STOR'].diff(-1) / data_f['STOR'].shift(-1)
            data_f['FSTOR_3'] = data_f['STOR'].diff(-3) / data_f['STOR'].shift(-3)

            progress_bar.progress(60)
            progress_text.text(f"Progress: {60}%")

            if '216ST2' in hole:
                data_f['SPPA_APRS'] = 0
            else:
                data_f['SPPA_APRS'] = data_f['SPPA'] / data_f['APRS']
            data_f['DSPPA_APRS_1'] = data_f['SPPA_APRS'].diff() / data_f['SPPA_APRS'].shift()
            data_f['DSPPA_APRS_3'] = data_f['SPPA_APRS'].diff(3) / data_f['SPPA_APRS'].shift(3)
            data_f['DSPPA_APRS_6'] = data_f['SPPA_APRS'].diff(6) / data_f['SPPA_APRS'].shift(6)
            data_f['DSPPA_APRS_12'] = data_f['SPPA_APRS'].diff(12) / data_f['SPPA_APRS'].shift(12)
            data_f['DSPPA_APRS_18'] = data_f['SPPA_APRS'].diff(18) / data_f['SPPA_APRS'].shift(18)
            data_f['FSPPA_APRS_1'] = data_f['SPPA_APRS'].diff(-1) / data_f['SPPA_APRS'].shift(-1)
            data_f['FSPPA_APRS_3'] = data_f['SPPA_APRS'].diff(-3) / data_f['SPPA_APRS'].shift(-3)

            data_f['DECD'] = data_f['ECD'].diff()

            data_f[['DDEPT', 'DBPOS', 'DHKLD', 'DSPPA', 'DECD']] = data_f[
                ['DDEPT', 'DBPOS', 'DHKLD', 'DSPPA', 'DECD']].fillna(0)
            shift_list = [
                'DDEPT_1', 'DDEPT_3', 'DDEPT_6', 'DDEPT_12', 'DDEPT_18', 'FDEPT_1', 'FDEPT_3',
                'DBPOS_1', 'DBPOS_3', 'DBPOS_6', 'DBPOS_12', 'DBPOS_18', 'FBPOS_1', 'FBPOS_3',
                'DHKLD_1', 'DHKLD_3', 'DHKLD_6', 'DHKLD_12', 'DHKLD_18', 'FHKLD_1', 'FHKLD_3',
                'DSPPA_1', 'DSPPA_3', 'DSPPA_6', 'DSPPA_12', 'DSPPA_18', 'FSPPA_1', 'FSPPA_3',
                'DSTOR_1', 'DSTOR_3', 'DSTOR_6', 'DSTOR_12', 'DSTOR_18', 'FSTOR_1', 'FSTOR_3',
                'DRPM_1', 'DRPM_3', 'DRPM_6', 'DRPM_12', 'DRPM_18', 'FRPM_1', 'FRPM_3',
                'DBVEL_1', 'DBVEL_3', 'DBVEL_6', 'DBVEL_12', 'DBVEL_18', 'FBVEL_1', 'FBVEL_3',
                'DSPPA_APRS_1', 'DSPPA_APRS_3', 'DSPPA_APRS_6', 'DSPPA_APRS_12', 'DSPPA_APRS_18',
                'FSPPA_APRS_1', 'FSPPA_APRS_3'
            ]
            for feat in shift_list:
                data_f[feat] = data_f[feat].fillna(0)

            if n == 0:
                data = data_f
                n += 1
            else:
                data = pd.concat([data, data_f], axis=0, ignore_index=True)
        data['STOR'].fillna((data['STOR'].mean()), inplace=True)
        data['ECD'] = data['ECD'].fillna(method='ffill')
        data['ECD'] = data['ECD'].fillna(method='bfill')
        data[['DLS', 'INCL', 'AZIM']] = data[['DLS', 'INCL', 'AZIM']].fillna(0)
        data['GR'].fillna((data['GR'].mean()), inplace=True)
        data['APRS'].fillna((data['APRS'].mean()), inplace=True)
        data['ESD'].fillna((data['ESD'].mean()), inplace=True)
        data = data.fillna(0)
        data['idx'] = data.index

        progress_bar.progress(70)
        progress_text.text(f"Progress: {70}%")

        df_range = pd.read_csv('range.csv')
        df_r = pd.pivot_table(df_range, index=["RANGE"])
        df_r['min_real'] = 0
        df_r['max_real'] = 0
        df_r = df_r.drop(['CDEPTH', 'StickPercentage', 'Stick_Slip_Ratio', 'TVD'], axis=0)
        print('ок')
        for rows in df_r.index:
            df_r['min_real'][rows] = data[rows].min()
            df_r['max_real'][rows] = data[rows].max()

        df_r['discard'] = df_r.apply(lambda n: 1 if n['max'] < n['max_real'] or n['min'] > n['min_real'] else 0, axis=1)

        # Создадим признак аномальных значений по атрибутам ['GR', 'BPOS', 'RPM', 'SPPA', 'STOR', 'ECD', 'INCL', 'APRS', 'Stick']
        data['ANOMAL'] = 0
        for rows in ['GR', 'BPOS', 'RPM', 'SPPA', 'STOR', 'ECD', 'INCL', 'APRS', 'Stick']:
            data.loc[data[rows] < df_r['min'][rows], 'ANOMAL'] += 1
            data.loc[data[rows] > df_r['max'][rows], 'ANOMAL'] += 1

        # Приведем значения выбросов к значениям с отклонением на 10% от граничных значений
        cat_cols = ['StuckPipe', 'RIG_STATE', 'hole', 'file_name', 'Date/Time']
        num_cols = list(set(df_r.index) - set(cat_cols))

        for col in num_cols:
            data.loc[data[col] > df_r['max'][col] * 1.5, col] = df_r['max'][col] * 1.5

        data.loc[data['Stick'] < (-300), 'Stick'] = -300

        # Приведем время к unix формату
        #data['Date/Time'] = pd.to_datetime(data['Date/Time'])
       # data['TimeU'] = data['Date/Time'].apply(lambda t: int(t.timestamp()))

        # Генерация новых признаков - разница между выбросом и пределом
        discard_list = ['SPPA', 'STOR', 'ECD', 'APRS', 'RPM', 'Stick']
        for rows in discard_list:
            IQR = data[rows].quantile(0.75) - data[rows].quantile(0.25)
            perc25 = data[rows].quantile(0.25)
            perc75 = data[rows].quantile(0.75)
            f = perc25 - 1.5 * IQR
            l = perc75 + 1.5 * IQR
            data[str('discard' + rows)] = 0
            data.loc[data[rows] > df_r['max'][rows], str('discard' + rows)] = data[rows].apply(lambda r: (r - df_r['max'][rows]))
            data.loc[data[rows] < 0, str('discard' + rows)] = data[rows].apply(lambda r: r)

        # Новый признак отношения расстояния до долота к общей длине скважины
        data['PDEPT'] = data['DEPT'] / data['HDTH']

        # Новый признак содержащий примерное отношение к формуле расчета силы давления на дно и стенки сосуда F= плотность*g*H*S
        # Зависимость GR и плотности обратная - чем больше Градусов API тем меньше плотность в кг/м3, поэтому берем 1/GR
        # http://oilreview.kiev.ua/tpga-mup/
        # Добавим признак, учитывающий угол наклона поверхности при давлении на дно.

        data['F'] = data['DEPT'] / data['GR']
        data['F_all'] = data['F'] * data['INCL'].apply(lambda u: np.sin(u))
        data['F_all'] = data['F_all'].fillna(method='bfill')

        # Новый признак - 'F_all' - вспомогательный, потом удалить
        # INCL - зенитный угол, отклонение касательной ствола скважины от вертикали (градус);
        # DLS - интенсивность искривления скважины (30 м от забоя по HDTH) (градус/30 м);
        # AZIM - бесполезная информация в данной задаче, поскольку показывает только направление по отношению к Северу по поверхности а не глубине
        # сумма INCL + DLS по идее должна быть в пределах 90 градусов. Если то больше 90, то меньше, то это образуется некое "вихляние" скважины
        # сыделаем из этого отдельный признак

        data['conner'] = data['INCL'] + data['DLS']
        data['conner_delta'] = data['conner'].apply(lambda u: 1 if u > 90 else 0)

        # Давление раствора на забое APRS складывается из давления насосов SPPA и гидростатического давления. Гидростатическое давление -
        # давление столба жидкости на дно сосуда (зависит от плотности бурового раствора ESD и абсолютной глубины ствола TVD).
        # Идея: Принимаем гидростатическое давление за const. Снижение APRS относительно SPPA может свидетельствовать об обрушении стенок скважины и возрастания вероятности аварии на обратной проработке.
        # Вывод: Вводим новую переменную как отношение APRS к SPPA.

        data['SPPA_APRS'] = data['SPPA_APRS'].fillna(0)

        # GR - естественная радиоактивность пород в скважине. Значительной радоактивностью обладают глины >200. Песчаники наоборот, имеют низкую <60.
        data['GR_type'] = 1
        data.loc[data['GR'] < 60, 'GR_type'] = 0
        data.loc[data['GR'] > 200, 'GR_type'] = 2

        # Бинарные переменные по наличию выбросов на интервале
        # Вылавливаются выбросы и ставится единица, если больше или меньше квантилей.
        # Так как разброс по скважинам отличается, то сделал в разрезе скважин
        # Корреляция с целевой переменной найдена по следующим переменным
      
  
        data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
        data.to_csv('stream_test.csv')
    progress_bar.progress(100)
    progress_text.text(f"Progress: {100}%")
    return data

# Функции обработки

def read_files(uploaded_file):
    """
    Функция для чтения файлов 
    """
    df = walk_dir(uploaded_file)

    return df

#def load_nn():
    #model = open('./Downloads/0-bur'+'/model_rez.model')
 #   model = pickle.load(open('./Downloads/0-bur'+'/model_rez.model', 'rb'))
 #   @catboost.load_model(model, './Downloads/0-bur'+'/model_rez.model')
 #   return model


param_list = ['DBPOS', 'ANOMAL', 'DDEPT_6', 'FSPPA_1', 'HKLD', 'DBVEL_18', 'PDEPT', 'DSPPA_APRS_6', 'DDEPT_12',
          'RIG_STATE', 'DBVEL_12', 'DHKLD_3', 'FSTOR_3', 'DSPPA_1', 'ESD', 'DRPM', 'DSTOR_1', 'DECD', 'conner',
          'DRPM_18', 'DRPM_6', 'DSPPA_APRS_1', 'DDEPT_18', 'FBPOS_3', 'FSPPA_APRS_3', 'F_all', 'idx', 'FSPPA_3',
          'DSPPA_18', 'FBVEL_3', 'DSPPA_APRS_18', 'DLS', 'DSTOR_18', 'ECD', 'STOR', 'RPM', 'SPPA_APRS', 'DEPT', 
          'DHKLD_12', 'hole', 'FHKLD_3', 'DSPPA_3', 'DHKLD_1', 'FDEPT_3', 'FSPPA_APRS_1', 'DDEPT_3', 'FBPOS_1', 
          'BPOS', 'GR_type', 'GR', 'DHKLD', 'DSPPA_APRS_3', 'FBVEL_1', 'DBVEL_6', 'DSPPA_APRS_12', 'DSTOR_12', 
          'DSPPA_12', 'DSPPA', 'DBPOS_3', 'DBPOS_1', 'SPPA', 'APRS', 'DRPM_12', 'DSPPA_6', 'DBPOS_6', 'F', 'FLWI',
          'DDEPT_1', 'FDEPT_1', 'BVEL', 'DBPOS_18', 'DDEPT', 'DSTOR_6', 'HDTH', 'FHKLD_1', 'DBPOS_12', 'INCL']


#['F', 'STOR', 'SPPA_APRS', 'FSTOR_3', 'DSPPA_3', 'DSPPA_18', 'DSPPA_6', 'FDEPT_3', 'ANOMAL', 
#              'DBPOS_6', 'DBVEL_6', 'HKLD', 'DBVEL_12', 'DHKLD_12', 'DBVEL_18', 'FBPOS_3', 'DRPM', 'RPM', 
#              'ESD', 'FSPPA_3', 'DSTOR_18', 'DSPPA_12', 'BPOS', 'FLWI', 'DBPOS_18', 'FHKLD_3', 'DSPPA_APRS_3',
#              'DBVEL_3', 'DDEPT_18', 'DRPM_18', 'ECD', 'PDEPT', 'DSTOR_12', 'FSPPA_APRS_3', 'SPPA', 
#              'DHKLD_18', 'DHKLD_6', 'DSPPA', 'DEPT', 'F_all', 'DBPOS_12']

param_base = ['Date/Time', 'TVD', 'DEPT', 'CDEPTH', 'HDTH', 'BPOS', 'HKLD', 'STOR', 'FLWI', 'RPM',
                    'SPPA', 'ECD', 'DLS', 'INCL', 'AZIM', 'GR', 'APRS', 'BVEL', 'RIG_STATE', 'Stick_Slip_Ratio',
                    'StickPercentage', 'ESD']

#Выводим рекомендации к ней
st.title('Авторазметка прихватов')


# @st.cache(allow_output_mutation=True)


def file_selector(folder_path='./bur'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox(' Выберете файл из списка: ', filenames)
    return os.path.join(folder_path, selected_filename)

def metrics_pr(pr):
    n=0
    list_1=[]
    list_sheet=[]
    m2=pr.index.max()
    m1=pr.index.min()
    #pr[m2+2]=0
    print(n)
    for i in range(m1, m2+1):
        if (pr[i]==1):
            n +=1
            if n==1:
                list_sheet.append(i)
            if i == m2:
                list_1.append(n)
        else: 
            if n!= 0:
                list_1.append(n)
                n=0
    return pd.Series(list_1), list_sheet

# Выбор файла из выпадающего списка

#filename = file_selector()
#st.write('Chosen Image `%s`' % filename)

#if filename:
    
    #st.write('*Выбран файл *`%s`' % filename)


# In[12]:

# Выбор файла через проводник
#st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader('Загрузите ваш файл:')
#@st.cache(allow_output_mutation=True)


if uploaded_file is not None:
    #Загружаем данные
    file_upl = uploaded_file.name
    dir_upl = os.path.abspath(file_upl)
    dir = os.path.dirname(dir_upl)  # путь корневого каталога
    full_name = os.path.basename(dir_upl)
    file_name = os.path.splitext(full_name)[0]
    st.write("Выбран файл", file_upl)
   # st.write("Выбрана директория", dir)
   # st.write("Полный путь", dir_upl )
 #  f_txt = os.path.split(dir_upl)[1].split(".")[1]  # dir_upl.splittext(full_name)[1]
 #   st.write("Расширение файла", f_txt )
    selected_obr = st.button('1. Начать предобработку файла')   
    if selected_obr:
        df  =  read_files(uploaded_file)   #('C:/Users/79265/Downloads/0-bur/f')  #+ str(uploaded_file)) #/0-bur/data
#    st.write('The image is %s with %.5f%% probability' % (class_info[predict_rank[0][0]], (predict_rank[1][0]*100) ))
        st.write('Успешно завершена предобработка файла: ', file_upl)
        st.dataframe(df.head(5))
#@st.cache(df)
    selected_mod = st.button('2. Произвести разметку и выгрузку файла')
    selected_chek = st.checkbox('с графиками')
    if selected_mod:
        df  =  pd.read_csv('stream_test.csv')
        #st.dataframe(df.head(5))
        df_res = df[param_base].copy()
        #scaler = MinMaxScaler()
        #df = scaler.fit_transform(df[param_list]) 
        df = df[param_list]
# Обработка файла моделью
        model = pickle.load(open('finalized_model17.pkl', 'rb'))   #('finalized_model7.pkl', 
        pred = model.predict_proba(df)[:, 1]
        pred2 = pd.Series(pred).apply(lambda n: 1 if n>0.51 else 0) #0.572 578 для 17-норм 575-по 3 много 5.8 - сам прихват на 5 единиц
        #st.write('Разметка')
# Постпроцессинг
        predict_list=pred2.copy()
        for i in range(1,len(pred2) - 4):  # промежуток в 3 нуля = 1
            if predict_list[i]==0 and predict_list[i-1]==1 and predict_list[i+1]==1:
                predict_list[i]=1
            if predict_list[i]==0 and predict_list[i-1]==1 and predict_list[i+2]==1:
                predict_list[i]=1
            if predict_list[i]==0 and predict_list[i-1]==1 and predict_list[i+3]==1:
                predict_list[i]=1
            if predict_list[i]==0 and predict_list[i-1]==1 and predict_list[i+4]==1:
                predict_list[i]=1
        #st.write('Разметка 3')
        for i in range(1,len(pred2) - 3):  # 3 подряд единички = 0
            if predict_list[i]==1 and predict_list[i-1]==0 and predict_list[i+1]==0:
                predict_list[i]=0
            if predict_list[i]==1 and predict_list[i-1]==0 and predict_list[i+1]==1 and predict_list[i+2]==0:
                predict_list[i]=0
                predict_list[i+1]=0
            if predict_list[i]==1 and predict_list[i-1]==0 and predict_list[i+1]==1 and predict_list[i+2]==1 and predict_list[i+3]==0:
                predict_list[i]=0
                predict_list[i+1]=0
                predict_list[i+2]=0
#            if predict_list[i]==1 and predict_list[i-1]==0 and predict_list[i+1]==1 and predict_list[i+2]==1 and predict_list[i+3]==0:
#                predict_list[i]=0
#                predict_list[i+1]=0
#                predict_list[i+2]=0

        #st.write('Разметка2')
        pred2=predict_list
        df_res['StuckPipeX'] = pred2
        st.dataframe(df_res.head(5))
        df_res['id']=df_res.index
       # st.write('Разметка прихватов успешно завершена')
        path_res = 'rez'+file_name+'.xlsx'

        list_1, list_sheet = metrics_pr(pred2)
        st.write( list(list_1))  
        count_pr = list_1[(list_1>0)&(list_1>5)].count()
        count_vb = list_1[(list_1>0)&(list_1<6)].count()
        st.write('Количество прихватов длиной от 1 минуты: ', count_pr)
        for i in range(len(list_sheet)):
            if list_1[i]>5: 
                st.write(df_res['Date/Time'].loc[list_sheet[i]])
        st.write('Количество прихватов длиной менее 1 минуты: ', count_vb)
        for i in range(len(list_sheet)):
            if list(list_1)[i]<6: 
                st.write(df_res['Date/Time'].loc[list_sheet[i]])
        st.write('Процент прерываний авторазметки: ', count_vb/(count_pr+count_vb)*100)
       # df_res['StuckPipeX'].px()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        y=pred2
        x=pred2.index
        fig, ax1 = plt.subplots(1, 1, figsize=(20, 10), dpi=50)
        ax1.plot(x, y, color='tab:blue')
        ax1.set_xlabel('Количество прихватов на диапазоне', fontsize=40)
        fig.tight_layout()
        #st.plotly_chart(fig)
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write('Дождитесь сообщения об окончании разметки.')
        
        if selected_chek:
           with pd.ExcelWriter(path_res, engine='xlsxwriter') as wb:    
               df_res.to_excel(wb, sheet_name='Sheet1', index=False)
               sheet = wb.sheets['Sheet1']

               for i in list_sheet:
                chart = wb.book.add_chart({'type': 'line'})
                chart.add_series({
                'categories': ['Sheet1', i-10, 23, i+40, 23],   
                'values':     ['Sheet1', i-10, 5, i+40, 5], 
                'name': 'BPOS',
                'y2_axis': True,
                'line':       {'color': 'red'}})

                chart.add_series({
                #'categories': ['Sheet1', i-10, 23, i+40, 23],   
                'values':     ['Sheet1', i-10, 7, i+40, 7], 
                'name': 'STOR',
               # 'y2_axis': True,
                'line':       {'color': 'blue'}})

                chart.add_series({
                #'categories': ['Sheet1', i-10, 23, i+40, 23],   
                'values':     ['Sheet1', i-10, 9, i+40, 9], 
                'name': 'RPM',
               # 'y2_axis': True,
                'line':       {'color': 'green'}})

                chart.add_series({
                #'categories': ['Sheet1', i-10, 23, i+40, 23],   
                'values':     ['Sheet1', i-10, 11, i+40, 11], 
                'name': 'ECD',
                'y2_axis': True,
                'line':       {'color': 'black'}})

                chart.add_series({
                #'categories': ['Sheet1', i-10, 23, i+40, 23],   
                'values':     ['Sheet1', i-10, 10, i+40, 10], 
                'name': 'SPPA',
                #'y2_axis': True,
                'line':       {'color': 'yellow'}})

                chart.add_series({
                #'categories': ['Sheet1', i-10, 23, i+40, 23],   
                'values':     ['Sheet1', i-10, 6, i+40, 6], 
                'name': 'HKLD',
               # 'y2_axis': True,
                'line':       {'color': 'brown'}})

                chart.set_y_axis({'name': 'BPOS/STOR/RPM/SPPA/HKLD'})
                chart.set_y2_axis({'name': 'ECD'})

                coord = str('Z'+str(i))
                sheet.insert_chart(coord, chart)   

           wb.save()

        else:
           df_res.to_excel(path_res)

        st.write('Файл с разметкой прихватов сформирован: ', dir+'\\'+path_res)
 



