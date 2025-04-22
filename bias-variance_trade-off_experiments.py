# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:28:25 2025

@author: amina
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns
from sklearn import svm


path="student_health_data.csv"
df= pd.read_csv(path)

#check if nothing is missing or duplicated
missing= df.isnull().sum()
duplicat=df.duplicated().sum()

#print(missing, duplicat) 

#Is there a trend between heart rate and age ?
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["Heart_Rate"],y=df["Stress_Level_Biosensor"],alpha=0.6)
plt.xlabel("Heart rate")
plt.ylabel("Stress level")
plt.title("Correlation heart rate vs stress level")
plt.grid(True)
plt.show() #there is some outliers

#Estimating bias and variance

X= df[["Heart_Rate"]].values
Y= df[["Age"]].values


models={"Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
         "SVM": svm.SVC()                                    
   }

result={}

for name_model, model in models.items():
    predictions= cross_val_predict(model, X,Y,cv=10)
    
    # bias²
    bias_squared= np.mean((np.mean(predictions)-Y)**2)
    
    #variance
    variance= np.var(predictions) # = np.mean((predictions-np.mean(prediction)**2))
    
    #Total error (MSE)
    total_error= mean_squared_error(Y,predictions) # = bias_squared + variance + irreducible error
    
    result[name_model]={
        "bias²": round(bias_squared,2),
        "variance":round(variance,2),
        "total error(MSE)":round(total_error,2)}
    
bias_variance_df= pd.DataFrame(result)
bias_variance_df # I expected that SVM would have a lower bias² than the other two models since it's a neural network and it doesn't 
#predict by making assumptions like the Linear Regression. SVM's MSE is quite high == might overfit. 
# Linear Regression has a score of 0 for variance which was expected. The random forest is more balanced