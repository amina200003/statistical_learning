# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 18:29:39 2025

@author: amina
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

model= LinearRegression()
df= pd.read_csv("5.Py.1.csv")
y= df["y"]
X=df[["X1","X2"]]

# Fit a linear regression model with response "y" and features "X1" and "X2". What is the standard error for B1 ?
X_sm = sm.add_constant(X)


model2 = sm.OLS(y, X_sm).fit()


print(model2.summary())


standard_error_B1 = model2.bse["X1"]
print("Standard Error for B1:", standard_error_B1)

all= [df["y"],df["X1"],df["X2"]]

#plot the data using "df.plot()"
df.plot()

#use the (standard) bootstrap to estimate B1. To within 10%,

n_iterations= 1000
n_sample= len(df)
b1_coeff= []

#boostrap loop
for i in range(n_iterations):
    sample_ind= np.random.choice(n_sample,n_sample,replace=True)
    X_sample= X.iloc[sample_ind]
    y_sample=y.iloc[sample_ind]
    
    model.fit(X_sample,y_sample)
    b1_coeff.append(model.coef_[0])
    
bootstrap= np.std(b1_coeff)
print("Bootstrap estimate of standard error for B1:", bootstrap)
 
#Use blocks of 100 contiguous observations, and resample ten whole blocks with replacement then paste them together to construct each bootstrap time series   
block_size = 100
num_blocks = 10


n_blocks_total = len(df) // block_size

b1_coeffs_block = []

for _ in range(n_iterations):
    
    block_indices = np.random.choice(n_blocks_total, num_blocks, replace=True)
    
    
    slices = [slice(i * block_size, (i + 1) * block_size) for i in block_indices]
    
  
    new_rows = np.r_[tuple(slices)]
    
    
    X_sample = X.iloc[new_rows]
    y_sample = y.iloc[new_rows]


    model = LinearRegression()
    model.fit(X_sample, y_sample)
    
    b1_coeffs_block.append(model.coef_[0])  # Coef for X1


block_bootstrap_se_b1 = np.std(b1_coeffs_block)
print("Block bootstrap SE for B1:", block_bootstrap_se_b1)
