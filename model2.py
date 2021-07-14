import pickle
from datetime import timedelta

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import itertools

from IPython.core.display import display
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

st.header('Предсказание прихвата на буровой')
st.subheader('время предупреждающего сигнала около 60 сек')
st.write("(Работа с множеством файлов, анализ последней ситуации)")

#model1 = "ctboost_predict_model60_6_New.pkl"
#model = "ctboost_predict_model60_6_all.pkl"
model = "ctboost_model_12_07.pkl" # Костя модель
#model = "ctboost_predict-60-1.pkl" # модель 60 сек тики перед таргетом
#model = "ctboost_predict-60-60.pkl" # модель 60 сек сдвиг

pickle_in = open(model, 'rb')
ctboost_model = pickle.load(pickle_in)
data = pd.DataFrame()

#Insert a file uploader that accepts multiple files at a time:
files = st.file_uploader("Choose a EXCEL file", accept_multiple_files=True, type=["xlsx"])

for path in files:
    button = st.button(path.name)
    filename = path.name
    #data = pd.read_csv(uploaded_file)
    #data1 = pd.read_excel(path, skiprows=[1, 2]) # режет 2 строки
    data1 = pd.read_excel(path)
    # data = data.append(data1, ignore_index= True)
    data = pd.concat([data, data1], ignore_index=True)
    st.write(data.shape)
    st.write(data.head(50))


    # button_start = st.button('Начать анализ в указанных параметрах')
    if button:
        st.write("Начинаем анализ... {}".format(path.name))
        #st.write(data.shape)
        file = os.path.split('/')[-1].split('.')[0].split('-')[0]

        # Функции для выравнивания временного ряда
        def upsampl_time(d1, delta):
            # upsamling - повышение разрешения каждые 10 секунд, вставляемая сторока заполняется NAN по другим признакам
            d1['Date/Time'] = pd.to_datetime(d1['Date/Time'])
            d1 = d1.set_index('Date/Time')
            d1 = d1.resample(delta).asfreq()
            d1 = d1.reset_index()
            return d1

        def downsampl_time(d1, delta):
            # downsamling - понижение разрешения до 10 секунд, при схлопывании по другим признакам берется максимальное значение
            d1['Date/Time'] = pd.to_datetime(d1['Date/Time'])
            d1 = d1.set_index('Date/Time')
            d1 = d1.resample(delta).max()
            d1 = d1.reset_index()
            return d1

        def sec_for_min(d1):
            # проставление "обрезанных" секунд при наличии данных для них
            i = 0
            t_old = pd.to_datetime(0)
            delta = timedelta(seconds=10)
            for t in d1["Date/Time"]:
                if (t_old.time().second >= 50) and (t_old.time().second <= 59):
                    delta = timedelta(seconds=2)
                else:
                    delta = timedelta(seconds=10)
                # print(t_old, delta)
                if i != 0:
                    if t.time().minute == t_old.time().minute:
                        t = (t_old + delta)
                        d1["Date/Time"][i] = t
                i += 1
                t_old = t
            return d1

        def walk_dir(path=path, data_f=data):
            #     COLUMN_NAMES = ['Date/Time', 'StuckPipe', 'TVD', 'DEPT', 'CDEPTH', 'HDTH', 'BPOS', 'HKLD', 'STOR',
            #                     'FLWI', 'RPM', 'SPPA', 'ECD', 'DLS', 'INCL', 'AZIM', 'GR', 'APRS', 'BVEL', 'RIG_STATE',
            #                     'Stick_Slip_Ratio', 'StickPercentage', 'ESD']

            #     data = pd.DataFrame(columns=COLUMN_NAMES)
            #     data_f = pd.DataFrame()
            #     data_f = pd.read_excel(path)
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

            file = filename.split('/')[-1].split('.')[0].split('-')[0]
            #     n=0
            # file = path
            data_f['file'] = file
            if 'Stick_Slip_Ratio' in data_f.columns:
                data_f.rename(columns=recolumn_2, inplace=True)
            else:
                data_f.rename(columns=recolumn_1, inplace=True)

            # приводим интервал времени к единому
            # data_f['Date/Time'] = pd.to_datetime(data_f['Date/Time'])
            data_f = sec_for_min(data_f)
            # max_delta = data_f['Date/Time'].diff().max().seconds
            # min_delta = data_f['Date/Time'].diff().min().seconds
            # if (max_delta == 60) and (min_delta == 0):
            #     # print ('60')
            #     data_f = sec_for_min(data_f)
            #     data_f = downsampl_time(data_f, '10S')
            # elif (max_delta == 30) and (min_delta == 0):
            #     # print =  ( '30')
            #     data_f = sec_for_min(data_f)
            #     data_f = upsampl_time(data_f, '10S')
            # elif max_delta == 10:
            #     # print ( '10')
            #     data_f = downsampl_time(data_f, '10S')
            # else:
            #     # print ( '2')
            #     data_f = downsampl_time(data_f, '10S')

            # data_f['Date/Time'] = pd.to_datetime(data_f['Date/Time'], format='%H:%M:%S').astype(str)
            # data_f['Date/Time'] = pd.to_datetime(data_f['Date/Time']).dt.time

                # обрабатываем значения -999.25
            for col in data_f.columns[data_f.dtypes == 'float64']:
                # data_f[col] = data_f[col].apply(lambda n: np.nan if n == -999.25 else n).astype('float')
                data_f.loc[data_f[col] == -999.25, col] = np.nan
            # приводим к общим единицам измерения
            data_f['hole'] = file
            hole_measures = pd.read_excel('Units & schema.xlsx', skiprows=[1])

            # Списки скважин, где требуется замена параметров
            # Крутящий момент STOR
            kft_lbf_list = hole_measures.loc[hole_measures['STOR'] == 'kft.lbf']['Скважина'].tolist()
            # Давление SPPA и APRS
            bar_list = hole_measures.loc[hole_measures['SPPA'] == 'bar']['Скважина'].tolist()
            # Давление SPPA и APRS
            kPa_list = hole_measures.loc[hole_measures['SPPA'] == 'kPa']['Скважина'].tolist()

            if file in kft_lbf_list:
                data_f['STOR'] = data_f['STOR'].astype('float'). \
                    apply(lambda x: x * 0.737 if x != -999.25 else np.nan)
            if file in bar_list:
                data_f['APRS'] = data_f['APRS'].astype('float'). \
                    apply(lambda x: x * 1.02 if x != -999.25 else np.nan)
                data_f['SPPA'] = data_f['SPPA'].astype('float'). \
                    apply(lambda x: x * 1.02 if x != -999.25 else np.nan)
            if file in kPa_list:
                data_f['APRS'] = data_f['APRS'].astype('float'). \
                    apply(lambda x: x * 0.01 if x != -999.25 else np.nan)
                data_f['SPPA'] = data_f['SPPA'].astype('float'). \
                    apply(lambda x: x * 0.01 if x != -999.25 else np.nan)

            # Замена NaN
            data_f['StuckPipe'] = data_f['StuckPipe'].fillna(0)

            data_f[['DEPT', 'HDTH']] = data_f[['DEPT', 'HDTH']].interpolate()
            data_f['DDEPT'] = data_f['DEPT'].diff()  # Новый признак изменения к предыдущему глубины долота

            data_f.loc[((data_f['RIG_STATE'].isna()) & (data_f['FLWI'] > 15) & (data_f['RPM'] > 15) &
                        (data_f['DDEPT'] > 0)), 'RIG_STATE'] = 1
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
            # data_f.loc[(data_f['RIG_STATE'] != 2 & data_f['BPOS'].isna()), 'BPOS'] = -data_f['DDEPT'].cumsum()

            data_f['BPOS'] = data_f['BPOS'].interpolate()
            data_f['BPOS'] = data_f['BPOS'].fillna(method='ffill')
            data_f['BPOS'] = data_f['BPOS'].fillna(method='bfill')

            data_f['HKLD'].fillna((data_f['HKLD'].rolling(10).mean()), inplace=True)
            data_f['STOR'].fillna((data_f['STOR'].rolling(10).mean()), inplace=True)

            FLWI_0_RIG = [2, 5, 6, 9, 10, 13, 14]  # Список статусов RIG_STATE, при которых FLWI = 0
            RPM_0_RIG = [2, 4, 6, 8, 10, 12, 14]  # То же для RPM = 0
            MOVE_RIG = [1, 3, 4, 5, 6, 7, 8, 9, 10]  # Список статусов RIG_STATE, когда крюк двигается

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

            data_f.loc[((data_f['BVEL'].isna()) & (data_f['RIG_STATE'].isin(MOVE_RIG))), 'BVEL'] = \
                data_f.loc[((data_f['BVEL'].isna()) & (data_f['RIG_STATE'].isin(MOVE_RIG)))]['DDEPT'] / 360
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
            data_f.loc[((data_f['Stick_Slip_Ratio'] == 0) & (data_f['StickPercentage'] != 0)), 'Stick'] = \
                data_f['StickPercentage']
            data_f.loc[~((data_f['Stick_Slip_Ratio'] == 0) & (data_f['StickPercentage'] != 0)), 'Stick'] = \
                data_f['Stick_Slip_Ratio']
            data_f['Stick'] = data_f['Stick'].fillna(0)

            for col in data_f.columns:
                # data_f[col] = data_f[col].fillna(data_f[col].rolling(1000).mean())
                data_f[col] = data_f[col].fillna(0)

            # новые признаки по временному сдвигу
            data_f['DDEPT_1'] = data_f['DEPT'].diff() / data_f['DEPT'].shift()
            data_f['DDEPT_3'] = data_f['DEPT'].diff(3) / data_f['DEPT'].shift(3)
            data_f['DDEPT_6'] = data_f['DEPT'].diff(6) / data_f['DEPT'].shift(6)
            data_f['DDEPT_12'] = data_f['DEPT'].diff(12) / data_f['DEPT'].shift(12)
            data_f['DDEPT_18'] = data_f['DEPT'].diff(18) / data_f['DEPT'].shift(18)
            data_f['FDEPT_1'] = data_f['DEPT'].diff(-1) / data_f['DEPT'].shift(-1)
            data_f['FDEPT_3'] = data_f['DEPT'].diff(-3) / data_f['DEPT'].shift(-3)

            # Новый признак изменения к предыдущему высоты крюка
            data_f['DBPOS'] = data_f['BPOS'].diff()
            data_f['DBPOS_1'] = data_f['BPOS'].diff() / data_f['BPOS'].shift()
            data_f['DBPOS_3'] = data_f['BPOS'].diff(3) / data_f['BPOS'].shift(3)
            data_f['DBPOS_6'] = data_f['BPOS'].diff(6) / data_f['BPOS'].shift(6)
            data_f['DBPOS_12'] = data_f['BPOS'].diff(12) / data_f['BPOS'].shift(12)
            data_f['DBPOS_18'] = data_f['BPOS'].diff(18) / data_f['BPOS'].shift(18)
            data_f['FBPOS_1'] = data_f['BPOS'].diff(-1) / data_f['BPOS'].shift(-1)
            data_f['FBPOS_3'] = data_f['BPOS'].diff(-3) / data_f['BPOS'].shift(-3)

            # Новый признак изменения к предыдущему веса крюка
            data_f['DHKLD'] = data_f['HKLD'].diff()
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

            data = data_f

            data_f['STOR'].fillna((data_f['STOR'].mean()), inplace=True)
            data_f['ECD'] = data_f['ECD'].fillna(method='ffill')
            data_f['ECD'] = data_f['ECD'].fillna(method='bfill')
            data_f[['DLS', 'INCL', 'AZIM']] = data_f[['DLS', 'INCL', 'AZIM']].fillna(0)
            data_f['GR'].fillna((data_f['GR'].mean()), inplace=True)
            data_f['APRS'].fillna((data_f['APRS'].mean()), inplace=True)
            data_f['ESD'].fillna((data_f['ESD'].mean()), inplace=True)
            data_f = data_f.fillna(0)

            df_range = pd.read_csv('range.csv')
            df_r = pd.pivot_table(df_range, index=["RANGE"])
            df_r['min_real'] = 0
            df_r['max_real'] = 0
            df_r = df_r.drop(['CDEPTH', 'StickPercentage', 'Stick_Slip_Ratio', 'TVD'], axis=0)

            for rows in df_r.index:
                df_r['min_real'][rows] = data_f[rows].min()
                df_r['max_real'][rows] = data_f[rows].max()

            df_r['discard'] = df_r.apply(lambda n: 1 if n['max'] < n['max_real'] or \
                                                        n['min'] > n['min_real'] else 0, axis=1)

            # Создадим признак аномальных значений по атрибутам ['GR', 'BPOS', 'RPM', 'SPPA', 'STOR', 'ECD', 'INCL', 'APRS', 'Stick']
            data_f['ANOMAL'] = 0
            for rows in ['GR', 'BPOS', 'RPM', 'SPPA', 'STOR', 'ECD', 'INCL', 'APRS', 'Stick']:
                data_f.loc[data_f[rows] < df_r['min'][rows], 'ANOMAL'] += 1
                data_f.loc[data_f[rows] > df_r['max'][rows], 'ANOMAL'] += 1

            # Приведем значения выбросов к значениям с отклонением на 10% от граничных значений
            cat_cols = ['StuckPipe', 'RIG_STATE', 'hole', 'file_name', 'Date/Time']
            num_cols = list(set(df_r.index) - set(cat_cols))

            for col in num_cols:
                data_f.loc[data_f[col] > df_r['max'][col] * 1.5, col] = df_r['max'][col] * 1.5

            data_f.loc[data_f['Stick'] < (-300), 'Stick'] = -300

            # Приведем время к unix формату
            # data_f['Date/Time'] = pd.to_datetime(data_f['Date/Time'])
            # data_f['TimeU'] = data_f['Date/Time'].apply(lambda t: int(t.timestamp()))

            # Генерация новых признаков - разница между выбросом и пределом
            discard_list = ['SPPA', 'STOR', 'ECD', 'APRS', 'RPM', 'Stick']
            for rows in discard_list:
                IQR = data_f[rows].quantile(0.75) - data_f[rows].quantile(0.25)
                perc25 = data_f[rows].quantile(0.25)
                perc75 = data_f[rows].quantile(0.75)
                f = perc25 - 1.5 * IQR
                l = perc75 + 1.5 * IQR
                data_f[str('discard' + rows)] = 0
                data_f.loc[data_f[rows] > df_r['max'][rows], str('discard' + rows)] = \
                    data_f[rows].apply(lambda r: (r - df_r['max'][rows]))
                data_f.loc[data_f[rows] < 0, str('discard' + rows)] = data_f[rows].apply(lambda r: r)

            # Новый признак отношения расстояния до долота к общей длине скважины
            data_f['PDEPT'] = data_f['DEPT'] / data_f['HDTH']

            # Новый признак содержащий примерное отношение к формуле расчета силы давления на дно и стенки сосуда F= плотность*g*H*S
            # Зависимость GR и плотности обратная - чем больше Градусов API тем меньше плотность в кг/м3, поэтому берем 1/GR
            # http://oilreview.kiev.ua/tpga-mup/
            # Добавим признак, учитывающий угол наклона поверхности при давлении на дно.

            data_f['F'] = data_f['DEPT'] / data_f['GR']
            data_f['F_all'] = data_f['F'] * data_f['INCL'].apply(lambda u: np.sin(u))
            data_f['F_all'] = data_f['F_all'].fillna(method='bfill')

            # Новый признак - 'F_all' - вспомогательный, потом удалить
            # INCL - зенитный угол, отклонение касательной ствола скважины от вертикали (градус);
            # DLS - интенсивность искривления скважины (30 м от забоя по HDTH) (градус/30 м);
            # AZIM - бесполезная информация в данной задаче, поскольку показывает только направление по отношению к Северу по поверхности а не глубине
            # сумма INCL + DLS по идее должна быть в пределах 90 градусов. Если то больше 90, то меньше, то это образуется некое "вихляние" скважины
            # сыделаем из этого отдельный признак

            data_f['conner'] = data_f['INCL'] + data_f['DLS']
            data_f['conner_delta'] = data_f['conner'].apply(lambda u: 1 if u > 90 else 0)

            # Давление раствора на забое APRS складывается из давления насосов SPPA и гидростатического давления. Гидростатическое давление -
            # давление столба жидкости на дно сосуда (зависит от плотности бурового раствора ESD и абсолютной глубины ствола TVD).
            # Идея: Принимаем гидростатическое давление за const. Снижение APRS относительно SPPA может свидетельствовать об обрушении стенок скважины и возрастания вероятности аварии на обратной проработке.
            # Вывод: Вводим новую переменную как отношение APRS к SPPA.

            data_f['SPPA_APRS'] = data_f['SPPA_APRS'].fillna(0)

            # GR - естественная радиоактивность пород в скважине. Значительной радоактивностью обладают глины >200. Песчаники наоборот, имеют низкую <60.
            data_f['GR_type'] = 1
            data_f.loc[data_f['GR'] < 60, 'GR_type'] = 0
            data_f.loc[data_f['GR'] > 200, 'GR_type'] = 2

            # Бинарные переменные по наличию выбросов на интервале
            # Вылавливаются выбросы и ставится единица, если больше или меньше квантилей.
            # Так как разброс по скважинам отличается, то сделал в разрезе скважин
            # Корреляция с целевой переменной найдена по следующим переменным
            # рассчет производится на основе исторических данных без заглядывания вперед (30 сек)
            bin_dict_new = {
                'DDEPT_3': 'DEPT_back3',
                'DSPPA_3': 'SPPA_back3',
                'DSPPA_APRS_3': 'SPPA_APRS_back3',
                'DBVEL_3': 'BVEL_back3',
                'DHKLD_3': 'HKLD_back3',
                'DBPOS_3': 'BPOS_back3',
                'DRPM_3': 'RPM_back3',
                'DSTOR_3': 'STOR_back3'
            }

            for val in bin_dict_new.values():
                data_f[val] = 0

            hole_dict = {}

            for hole in data_f.hole.unique():
                hole_dict[hole] = {}
                for key, val in bin_dict_new.items():
                    new_dict = {}
                    hole_dict[hole][key] = [val]
                    for i in range(89, 100):
                        for j in range(1, 8):
                            df1 = data_f.loc[(data_f['hole'] == hole) & (data_f['RIG_STATE'] == 7)]
                            df1.loc[(df1[key] >= df1[key].quantile(i / 100)), val] = 1
                            df1.loc[(df1[key] <= df1[key].quantile(j / 100)), val] = 1
                            if np.isnan(df1[val].corr(df1['StuckPipe'])):
                                new_dict[0] = [i / 100, j / 100]
                            else:
                                new_dict[abs(df1[val].corr(df1['StuckPipe']))] = [i / 100, j / 100]
                    max_corr = max(new_dict.keys())
                    hole_dict[hole][key].append(new_dict[max_corr])

            for hole, value in hole_dict.items():
                for key, val in value.items():
                    df2 = data_f.loc[(data_f['hole'] == hole) & (data_f['RIG_STATE'] == 7)]
                    df2.loc[(df2[key] >= df2[key].quantile(val[1][0])), val[0]] = 1
                    df2.loc[(df2[key] <= df2[key].quantile(val[1][1])), val[0]] = 1
                    data_f.loc[data_f.index.isin(df2.loc[df2[val[0]] == 1].index), [val[0]]] = 1

            # Те же самые бинарные переменные, на основе исторических данных (10 сек)
            bin_dict_new = {
                'DDEPT_1': 'DEPT_back1',
                'DSPPA_1': 'SPPA_back1',
                'DSPPA_APRS_1': 'SPPA_APRS_back1',
                'DBVEL_1': 'BVEL_back1',
                'DHKLD_1': 'HKLD_back1',
                'DBPOS_1': 'BPOS_back1',
                'DRPM_1': 'RPM_back1',
                'DSTOR_1': 'STOR_back1'
            }

            for val in bin_dict_new.values():
                data_f[val] = 0

            hole_dict = {}

            for hole in data_f.hole.unique():
                hole_dict[hole] = {}
                for key, val in bin_dict_new.items():
                    new_dict = {}
                    hole_dict[hole][key] = [val]
                    for i in range(89, 100):
                        for j in range(1, 8):
                            df1 = data_f.loc[(data_f['hole'] == hole) & (data_f['RIG_STATE'] == 7)]
                            df1.loc[(df1[key] >= df1[key].quantile(i / 100)), val] = 1
                            df1.loc[(df1[key] <= df1[key].quantile(j / 100)), val] = 1
                            if np.isnan(df1[val].corr(df1['StuckPipe'])):
                                new_dict[0] = [i / 100, j / 100]
                            else:
                                new_dict[abs(df1[val].corr(df1['StuckPipe']))] = [i / 100, j / 100]
                    max_corr = max(new_dict.keys())
                    hole_dict[hole][key].append(new_dict[max_corr])

            for hole, value in hole_dict.items():
                for key, val in value.items():
                    df2 = data_f.loc[(data_f['hole'] == hole) & (data_f['RIG_STATE'] == 7)]
                    df2.loc[(df2[key] >= df2[key].quantile(val[1][0])), val[0]] = 1
                    df2.loc[(df2[key] <= df2[key].quantile(val[1][1])), val[0]] = 1
                    data_f.loc[data_f.index.isin(df2.loc[df2[val[0]] == 1].index), [val[0]]] = 1

            data_f = data_f.replace([np.inf, -np.inf], np.nan).fillna(0)
            data_f.to_csv('data.csv', index=False)


        walk_dir()
        data = pd.read_csv('data.csv')
        #data = data_f
        display(data)

        # display(data)
        # признаки берем те же, что и в обучении были (могут меняться):
        X = data[[
            'DEPT', 'HDTH', 'BPOS', 'HKLD', 'STOR', 'FLWI', 'RPM',
            'SPPA', 'ECD', 'DLS', 'INCL', 'AZIM', 'GR', 'APRS', 'BVEL', 'RIG_STATE',
            'ESD', 'DDEPT', 'Stick', 'DDEPT_1', 'DDEPT_3',
            'DDEPT_6', 'DDEPT_12', 'DDEPT_18',
            'DBPOS', 'DBPOS_1', 'DBPOS_3',
            'DBPOS_6', 'DBPOS_12', 'DBPOS_18',
            'DHKLD', 'DHKLD_1', 'DHKLD_3', 'DHKLD_6', 'DHKLD_12', 'DHKLD_18',
            'DSPPA', 'DSPPA_1', 'DSPPA_3', 'DSPPA_6',
            'DSPPA_12', 'DSPPA_18', 'DRPM', 'DRPM_1', 'DRPM_3', 'DRPM_6', 'DRPM_12',
            'DRPM_18', 'DBVEL', 'DBVEL_1', 'DBVEL_3', 'DBVEL_6', 'DBVEL_12', 'DBVEL_18',
            'DSTOR_1', 'DSTOR_3', 'DSTOR_6', 'DSTOR_12', 'DSTOR_18',
            'SPPA_APRS', 'DSPPA_APRS_1', 'DSPPA_APRS_3', 'DSPPA_APRS_6',
            'DSPPA_APRS_12', 'DSPPA_APRS_18', 'DECD', 'ANOMAL', 'discardSPPA', 'discardSTOR',
            'discardECD', 'discardStick', 'discardAPRS', 'discardRPM',
            'PDEPT', 'F', 'F_all', 'conner', 'conner_delta', 'GR_type',
            'DEPT_back3', 'SPPA_back3', 'SPPA_APRS_back3',
            'BVEL_back3', 'HKLD_back3', 'BPOS_back3',
            'RPM_back3', 'STOR_back3', 'DEPT_back1', 'SPPA_back1',
            'SPPA_APRS_back1', 'BVEL_back1', 'HKLD_back1',
            'BPOS_back1', 'RPM_back1',
            'STOR_back1'
        ]].values
        # нормализация: # убрать ее в модели - без нее в катбуст лучше результаты:
        #sc = StandardScaler()
        #X_norm = sc.fit_transform(X)
        X_norm = X
        #data = data[:1445]  # 186 #500
        #X = X[:1445]

        # y = data['StuckPipe60_6'].values

        # сделаем предикт Stuckpipe 60:
        y_predict = ctboost_model.predict(X_norm)
        y_predict_proba = ctboost_model.predict_proba(X_norm)
        # df7 = pd.DataFrame(y_predict)
        # data["predict_class"] = df7
        # print(data["predict_class"].value_counts())
        # df9 = pd.DataFrame(y_predict_proba[1])
        # data["predict_proba"] = df9

        # Постпроцессинг. Замена одиноких нолей, использование порога вероятности.
        # df5 = pd.DataFrame()
        porog = 0.3  # задаем порог

        df5 = pd.DataFrame((_ for _ in itertools.zip_longest(*y_predict_proba))).T
        df5.rename(columns={0: 'predict_0', 1: 'predict_1'}, inplace=True)
        df5['mark'] = 0
        df5.loc[df5['predict_1'] > porog, 'mark'] = 1
        predict_list = df5['mark'].tolist()

        for i in range(len(predict_list) - 4):
            if predict_list[i] == 0 and predict_list[i - 1] == 1 and predict_list[i + 1] == 1:
                predict_list[i] = 1
            if predict_list[i] == 0 and predict_list[i - 1] == 1 and predict_list[i + 2] == 1:
                predict_list[i] = 1

            if predict_list[i]==1 and predict_list[i-1]==0 and predict_list[i+1]==0:
                predict_list[i]=0
            if predict_list[i]==1 and predict_list[i-1]==0 and predict_list[i+2]==0:
                predict_list[i]=0
            if predict_list[i]==1 and predict_list[i-1]==0 and predict_list[i+3]==0:
                predict_list[i]=0
            if predict_list[i]==1 and predict_list[i-1]==0 and predict_list[i+4]==0:
                predict_list[i]=0

        # df7 = pd.DataFrame(y_predict)
        data["predict_class"] = pd.Series(predict_list).values
        data['predict_1'] = df5['predict_1']

        # продолжение исходного кода...

        #### игры с порогом (Павел)
        # y_predict_proba1 = (y_predict_proba[:, 1] > 0.3)
        # df77 = pd.DataFrame(y_predict_proba1)
        # # data["predict_class"] = df7ctboost_predict_model-30_1.pkl
        # data["predict_class"] = df77
        #print(data["predict_proba"].value_counts())

        n = data.index.max()
        X_valid = X[n]
        feat_list = ['BPOS', 'HKLD', 'STOR', 'ECD', "FLWI", "RPM", "SPPA"]
        Time = data["Date/Time"][n]
        hole_number = data["hole"][n]
        id_index = data.index[n]
        Stuckpipe = data["StuckPipe"][n]
        BPOS = data["BPOS"][n]
        HKLD = data["HKLD"][n]
        SPPA = data["SPPA"][n]
        FLWI = data["FLWI"][n]
        SPPA = data["SPPA"][n]
        STOR = data["STOR"][n]
        RPM = data["RPM"][n]
        ECD = data["ECD"][n]

        y_predict_proba_1 = data['predict_1'][n]# вариант с вероятностью
        y_predict_1 = data["predict_class"][n]

        if y_predict_1 == 1 and y_predict_proba_1 > 0.3:
            os.system('say "Опасность прихвата через 60 секунд!"')
            # espeak.synth("Hello world.")
            st.header("Внимание! Высокая вероятность прихвата: {0:0.2f}".format(
                y_predict_proba_1))

            st.write((Time))
            st.write("")
            st.write("ID: {}".format(id_index))
            # st.write("Разметка Stuckpipe: {}".format(Stuckpipe))
            st.write("Hole: {}".format(hole_number))
            st.write("BPOS: {0:0.2f}".format(BPOS))
            st.write("HKLD: {0:0.2f}".format(HKLD))
            st.write("STOR: {0:0.2f}".format(STOR))
            st.write("FLWI: {0:0.2f}".format(FLWI))
            st.write("RPM: {0:0.2f}".format(RPM))
            st.write("SPPA: {0:0.2f}".format(SPPA))
            st.write("ECD: {0:0.2f}".format(ECD))

        else:
            os.system('say "Норма"')
            st.subheader("Показания датчиков в норме.")
            st.write(Time)
            # out_green("Норма. Вероятность прихвата: {0:0.2f}".format(
            # y_predict_proba[1]))
            st.write("ID: {}".format(id_index))
            # st.write("Разметка Stuckpipe: {}".format(Stuckpipe))
            st.write("Hole: {}".format(hole_number))
            st.write("BPOS: {0:0.2f}".format(BPOS))
            st.write("HKLD: {0:0.2f}".format(HKLD))
            st.write("STOR: {0:0.2f}".format(STOR))
            st.write("FLWI: {0:0.2f}".format(FLWI))
            st.write("RPM: {0:0.2f}".format(RPM))
            st.write("SPPA: {0:0.2f}".format(SPPA))
            st.write("ECD: {0:0.2f}".format(ECD))


        # Отрисовка временных рядов:
        def plt_time_shift(data, time_col, col_list, hole_id):

            for i in range(len(col_list)):
                st.set_option('deprecation.showPyplotGlobalUse', False)
                x = data[time_col]
                # x = data.index
                # x = data[time_col].to_series().dt.strftime('%Y-%m-%d')
                y1 = data[col_list[i]]
                y2 = data["predict_class"]  # лучше заменить на y_predict!!!

                # Plot Line1 (Left Y Axis)
                fig, ax1 = plt.subplots(1, 1, figsize=(20, 10), dpi=50)
                ax1.plot(x, y1, color='tab:red')

                ax2 = ax1.twinx()  # добавляем вторую ось у на ту же ось х:
                ax2.plot(x, y2, color='tab:blue')

                # Левая шкала у1
                ax1.set_xlabel('Time', fontsize=40)
                ax1.tick_params(axis='x', rotation=90, labelsize=25)
                ax1.set_ylabel(f'{col_list[i]}', color='tab:red', fontsize=30)
                ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red')
                ax1.grid(alpha=.4)

                # Правая шкала у2
                ax2.set_ylabel("Сигнал прихвата", color='tab:blue', fontsize=30)
                ax2.tick_params(axis='y', labelcolor='tab:blue')
                ax2.set_xticks(np.arange(0, len(x), 60))
                ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize': 40})
                ax2.set_title(f'Зависимость {col_list[i]} от Сигнала по скважине №{hole_id}',
                              fontsize=40)

                fig.tight_layout()
                # plt.show()
                st.pyplot()
                st.set_option('deprecation.showPyplotGlobalUse', False)


        for hole in data.hole.unique():
            plt_time_shift(data, 'Date/Time', feat_list, hole)





