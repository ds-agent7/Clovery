import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

st.header('Предсказание прихвата на буровой (построчно)')
st.subheader('время сигнала около 120 сек')
# загрузка csv файла:
w = st.file_uploader("Upload a CSV file", type="csv")
if w:
    import pandas as pd

    data = pd.read_csv(w)
    #st.write(data)  вывод датасета на экран

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

    #st.write("ok")

    y_predict = ctboost_model.predict(X_norm)
    y_predict_proba = ctboost_model.predict_proba(X_norm)
    #st.write(pred)


    data = data[:195] #186 #500
    X = X[:195]
    n = data.index.max()

    for i in range(0, n + 1):
        X_valid = X[i]
        X_n = X_norm[i]
        Time = data["Date/Time"][i]
        # BPOS = data["BPOS"][i]
        hole_number = data["hole"][i]
        id_index = data.index[i]
        Stuckpipe = data["StuckPipe"][i]  # только для TEST
        # y_predict = ctboost_model.predict(X_valid)  # вариант с категорией
        y_predict_proba = ctboost_model.predict_proba(X_n)  # вариант с вероятностью
        # if y_predict==1:
        if y_predict_proba[1] > 0.5:
            os.system('say "Опасность прихвата 2 минуты"')
            st.write("ID: {}".format(id_index))
            st.header("Внимание! Вероятность прихвата: {0:0.2f}".format(
                y_predict_proba[1]))
            st.write((Time))
            st.write("Разметка Stuckpipe: {}".format(Stuckpipe))  # только для TEST

            print("")
            st.write("Hole: {}".format(hole_number))
            st.write("BPOS: {0:0.2f}".format(X_valid[2]))
            st.write("HKLD: {0:0.2f}".format(X_valid[3]))
            st.write("STOR: {0:0.2f}".format(X_valid[4]))
            st.write("FLWI: {0:0.2f}".format(X_valid[5]))
            st.write("RPM: {0:0.2f}".format(X_valid[6]))
            st.write("SPPA: {0:0.2f}".format(X_valid[7]))
            st.write("ECD: {0:0.2f}".format(X_valid[8]))
            st.write("_")
            st.write("_")
            time.sleep(1.0)  # время задержки
        else:
            st.write("ID: {}".format(id_index))
            st.subheader("Показания датчиков в норме.")
            # out_green("Норма. Вероятность прихвата: {0:0.2f}".format(
            # y_predict_proba[1]))
            st.write(Time)
            st.write("Разметка Stuckpipe: {}".format(Stuckpipe))  # только для TEST
            st.write("_ ")
            st.write("_ ")
            time.sleep(0.2)