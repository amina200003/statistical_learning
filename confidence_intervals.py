# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:26:11 2025

@author: amina
"""

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
path="student_health_data.csv"
df= pd.read_csv(path)

data=list(df["Heart_Rate"])

mean= np.mean(data)
ecart_type= np.std(data)
n= len(data)

#Calcul de l'intervalle de confiance à 95%
confiance= 0.95
Z_normal= st.norm.ppf((1+confiance)/2)
marge_error= Z_normal*(ecart_type/np.sqrt(n))

#We use Loi Normale because n>30 but if n<30 use loi de Student:
    #t_value = st.t.ppf((1 + confidence) / 2, df=n-1)
    
#borne de l'intervalle
lower_bound= mean-marge_error
upper_bound= mean+marge_error

#The interval is [69.51-70.69] 

#Lets make predictions to see if the true values really fall into this interval
X= df[["Study_Hours","Project_Hours"]]
y= df["Heart_Rate"]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model= LinearRegression()
# model.fit(X_train,y_train)
# y_pred= model.predict(X_test) 

#check if y_test/actual values are in the intervals
#inside_interval= (y_test>=lower_bound)&(y_test<=upper_bound)

#percentage_inside= np.mean(inside_interval)*100 #3%, the interval is too tight 

#with another model
scalar= StandardScaler()
Xscaled= scalar.fit_transform(X)
model2= SVR(kernel='rbf', C=1.0, epsilon=0.1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(Xscaled, y, test_size=0.2, random_state=42)



model2.fit(X_train2, y_train2)


y_pred2 = model2.predict(X_test2)

inside_interval= (y_test2>=lower_bound)&(y_test2<=upper_bound)

percentage_inside2= np.mean(inside_interval)*100 #SAME