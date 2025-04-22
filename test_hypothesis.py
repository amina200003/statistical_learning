# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:36:10 2025

@author: amina
"""

import numpy as np
import pandas as pd
import scipy.stats as st

path="student_health_data.csv"
df= pd.read_csv(path)

#h0= The feature Heart-rate(sample) is equal to the population mean =80
#ha= the sample mean is != 80 

heart_rate= df["Heart_Rate"]
pop_mean=80
sample_size= len(heart_rate)

mean_sample= np.mean(heart_rate)
standard_derivation_sample= np.std(heart_rate,ddof=1)

z_score= (mean_sample-pop_mean)/(standard_derivation_sample/np.sqrt(sample_size))

p_value= 2*(1-st.norm.cdf(abs(z_score)))

if p_value<0.05:
    print("H0 is rejected")
else:
    print("H0 is right")