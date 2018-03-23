#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 22:50:19 2018

@author: geoffrey.kip
"""

# Define functions needed for analysis
#find missing variables
def missing(dataset):
    print(dataset.apply(lambda x: sum(x.isnull().values), axis = 0))

def frequency(dataset):
        for col in dataset:
            print(dataset.groupby(col).size())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

meo_data= pd.read_csv("beijing_17_18_meo.csv")
aq_data= pd.read_csv("beijing_17_18_aq.csv")

meo_data.describe()
meo_data.dtypes
missing(meo_data)

#Replace missing values with the mean
meo_data["wind_direction"]= np.where(meo_data["wind_direction"].isnull(),meo_data["wind_direction"].mean(),meo_data["wind_direction"])
meo_data["wind_speed"]= np.where(meo_data["wind_speed"].isnull(),meo_data["wind_speed"].mean(),meo_data["wind_speed"])
missing(meo_data)

#check frequency distribution of all categorical variables
frequency(meo_data.select_dtypes(include=[np.object]))

#Select only continuos columns
quant_columns= meo_data.select_dtypes(include=[np.float64,np.int64]).columns.drop(['longitude', 'latitude','wind_direction'])
#check the distribution of continuos variables
for i, col in enumerate(meo_data[quant_columns]):
    plt.figure(i)
    sns.distplot(meo_data[col])
    
#Remove outliers eg. Remove any values from the quant columns that are 3 sd from the mean
meo_data= meo_data[(np.abs(stats.zscore(meo_data[quant_columns])) < 3).all(axis=1)]
    
#Scale all the continuos columns 
meo_data1= meo_data[meo_data.columns.difference(quant_columns)]
scaler= StandardScaler()
scaled_cols  = pd.DataFrame(scaler.fit_transform(meo_data[quant_columns]),columns=quant_columns)
meo_data= pd.merge(meo_data1, scaled_cols, left_index=True, right_index=True)

#Recheck distribution after removing outliers and scaling columns
for i, col in enumerate(meo_data[quant_columns]):
    plt.figure(i)
    sns.distplot(meo_data[col])
    