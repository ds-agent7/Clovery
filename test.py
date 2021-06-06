import pickle

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from catboost import CatBoostClassifier

from sklearn.preprocessing import StandardScaler

st.header('Предсказание прихвата на буровой')
st.subheader('время сигнала около 120 сек')
# загрузка csv файла:
#@st.cache()
w = st.file_uploader("Upload a CSV file", type="csv")
if w:
    data = pd.read_csv(w)

#file = "ctboost_predict_model60_6_New.pkl"
#file = "https://github.com/ds-agent7/Clovery/blob/main/ctboost_predict_model60_6_New.pkl"
#file = "https://drive.google.com/file/d/1rZ8BpoFMjh66tUNuEi9NEwQnOn8ieFUb/view?usp=sharing"

    file = "ctboost_predict_model60_6_New.pkl"

    pickle_in = open(file, 'rb')
    ctboost_model = pickle.load(pickle_in)

    X =data[['GR', 'SPPA_APRS', 'ECD', 'APRS', 'RPM', 'FLWI','STOR', 'BPOS', 'SPPA',
          'DDEPT_3', 'DDEPT_6', 'conner',
        'F', 'PDEPT', 'DDEPT_12', 'DDEPT_18','DBPOS_3','DBPOS_6', 'DBPOS_12', 'DBPOS_18', 'DEPT']].values

    # нормализация:
    sc= StandardScaler()
    X_norm = sc.fit_transform(X)
    y = data['StuckPipe60_6'].values

    y_predict = ctboost_model.predict(X_norm)
    y_predict_proba = ctboost_model.predict_proba(X_norm)
    df7 = pd.DataFrame(y_predict)
    data["predict_class"] = df7


    # Отрисовка временных рядов:

    def plt_time_shift(data, time_col, col_list, hole_id):

        for i in range(len(col_list)):
            st.set_option('deprecation.showPyplotGlobalUse', False)
            x = data[time_col]
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
            #plt.show()
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)


    data = data[:1000] #186 #500
    X = X[:1000]


    n = data.index.max()
    X_valid = X[n]
    feat_list = [ 'BPOS','HKLD','STOR','ECD',"FLWI","RPM","SPPA"]
    Time = data["Date/Time"][n]
    hole_number = data["hole"][n]
    id_index = data.index[n]
    Stuckpipe = data["StuckPipe"][n]

    #y_predict = ctboost_model.predict(X_valid)  # вариант с категорией
    y_predict_proba = ctboost_model.predict_proba(X_norm[n]) # вариант с вероятностью
    #if y_predict==1:
    if y_predict_proba[1] > 0.5:
        os.system('say "Опасность прихвата через 2 минуты!"')
        st.header ("Внимание! Вероятность прихвата: {0:0.2f}".format(
                y_predict_proba[1]))

        st.write((Time))
        st.write("")
        #st.write("Разметка Stuckpipe: {}".format(Stuckpipe))
        st.write("ID: {}".format(id_index))
        st.write("Hole: {}".format(hole_number))
        st.write("BPOS: {0:0.2f}".format(X_valid[2]))
        st.write("HKLD: {0:0.2f}".format(X_valid[3]))
        st.write("STOR: {0:0.2f}".format(X_valid[4]))
        st.write("FLWI: {0:0.2f}".format(X_valid[5]))
        st.write("RPM: {0:0.2f}".format(X_valid[6]))
        st.write("SPPA: {0:0.2f}".format(X_valid[7]))
        st.write("ECD: {0:0.2f}".format(X_valid[8]))
        st.write("")
        st.write("")
        st.write("")
        time.sleep(0.1) # время задержки

    else:
        os.system('say "Норма"')
        st.subheader("Показания датчиков в норме.")
        # out_green("Норма. Вероятность прихвата: {0:0.2f}".format(
        # y_predict_proba[1]))
        st.write("ID: {}".format(id_index))
        #st.write("Разметка Stuckpipe: {}".format(Stuckpipe))
        st.write(Time)
        st.write("")
        st.write("")

    for hole in data.hole.unique():
        plt_time_shift(data, 'Date/Time', feat_list, hole)
