# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 11:50:15 2025

@author: amina
"""

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

path="student_health_data.csv"
df= pd.read_csv(path)
model=SVR(kernel='linear', C=1.0, epsilon=0.1)


X= df[["Project_Hours","Stress_Level_Biosensor"]]
y=df["Age"]

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)
model.fit(x_train,y_train)

y_pred= model.predict(X)

#RSS: residual_sum_of_squares

tablo_predict_actual= pd.DataFrame({"actual":y,"predicted":y_pred})

RSS= sum((tablo_predict_actual["actual"]-tablo_predict_actual["predicted"])**2)
print("RSS:",RSS)

#TSS: Total Sum Of squares

TSS= sum((tablo_predict_actual["actual"]-np.mean(tablo_predict_actual["actual"]))**2)
print("TSS:",TSS)

#R²
R_squared= 1-(RSS/TSS)
print("R_squared:",R_squared) #the model is good enough to 
#predict tge mean of for each yi

#f_statistic
f= ((TSS-RSS)/2)/(RSS/(1000-2-1))
#f=0.4, the features choosen don't have a significant importance
#to predict y