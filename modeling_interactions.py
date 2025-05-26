# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:59:33 2025

@author: amina
"""

import pandas as pd
import statsmodels.formula.api as wth
import chardet

# Read the first 10000 bytes to guess encoding
with open('sales_data_sample.csv', 'rb') as f:
    result = chardet.detect(f.read(10000))
    #print(result)

data = pd.read_csv('sales_data_sample.csv', encoding=result['encoding'])

val= pd.DataFrame({
    "quantity": list(data["QUANTITYORDERED"]),
    "sales":list(data["SALES"]),
    "order_num":list(data["ORDERNUMBER"])})

model= wth.ols("order_num~ quantity + sales + quantity:sales",data=val).fit()
# fits a linear regression, the goal is to predict the order number using sales,quantity and an interaction between these two
#print(model.summary())

#get coeff
beta_quantity= model.params["quantity"]
beta_interaction= model.params["quantity:sales"]

#function to computa the effect of quantity at given sales
def effec_quantity(sales_val):
    return beta_quantity + beta_interaction*sales_val

#example
effect_sales_at_250= effec_quantity(250)
print(f"effect of quantity when sales=250:{effect_sales_at_250}")