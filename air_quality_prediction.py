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
from datetime import datetime
 
meo_data= pd.read_csv("beijing_17_18_meo.csv")
aq_data= pd.read_csv("beijing_17_18_aq.csv")

##Meteorlogical data feature engineering

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
    
# Drop latitude and longitude as this can be inferred by the station id
meo_data= meo_data.drop(["latitude", "longitude"], axis=1)
categorical_data= meo_data.select_dtypes(include=[np.object])

# Do some feature engineering on dates. I am guessing time of the year plays a big role in air quality
day_of_week = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).weekday()
month = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).month
week_number = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).strftime('%V')

# Code seasons
seasons = [0,0,1,1,1,2,2,2,3,3,3,0] #dec - feb is winter, then spring, summer, fall etc
season = lambda x: seasons[(datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).month-1)]
 
# sleep: 12-5, 6-9: breakfast, 10-14: lunch, 14-17: dinner prep, 17-21: dinner, 21-23: deserts!
times_of_day = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5 ]
time_of_day = lambda x: times_of_day[(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)]


# Create new date variables
categorical_data['day_of_week'] = categorical_data['utc_time'].map(day_of_week)
categorical_data['month'] = categorical_data['utc_time'].map(month)
categorical_data['week_number'] = categorical_data['utc_time'].map(week_number)
categorical_data['season'] = categorical_data['utc_time'].map(season)
#categorical_data['time_of_day'] = categorical_data['utc_time'].map(time_of_day)
categorical_data= categorical_data.drop(["utc_time"],axis=1)

# Do some binary encoding for time variables
#categorical_data["weather_cat"] = categorical_data["weather"].astype('category').cat.codes

# one hot encode station_id and weather
categorical_data= pd.get_dummies(categorical_data, columns=["station_id"])
categorical_data= pd.get_dummies(categorical_data, columns=["weather"])

# Drop categorical variables from meo_data
meo_data= meo_data.drop(["weather"], axis=1)

# Code type of station

meo_data['urban_station']=np.where(meo_data['station_id'].isin(["dongsi_meo",
"tiantan_meo",
"guanyuan_meo",
"wanshouxigong_meo",
"aotizhongxin_meo",
"nongzhanguan_meo",
"wanliu_meo",
"beibuxinqu_meo",
"zhiwuyuan_meo",
"fengtaihuayuan_meo",
"yungang_meo",
"gucheng_meo"]),1,0)
    
meo_data['suburban_station']=np.where(meo_data['station_id'].isin(["fangshan_meo",
"daxing_meo",
"yizhuang_meo",
"tongzhou_meo",
"shunyi_meo",
"pingchang_meo",
"mentougou_meo",
"pinggu_meo",
"huairou_meo",
"miyun_meo",
"yanqin_meo"]),1,0)
    
meo_data['other_station']=np.where(meo_data['station_id'].isin(["dingling_meo",
"badaling_meo",
"miyunshuiku_meo",
"donggaocun_meo",
"yongledian_meo",
"yufa_meo",
"liulihe_meo"]),1,0)
    
meo_data['station_near_traffic']=np.where(meo_data['station_id'].isin(["qianmen_meo",
"yongdingmennei_meo",
"xizhimenbei_meo",
"nansanhuan_meo",
"dongsihuan_meo"]),1,0)

meo_data_prepped= pd.merge(meo_data, categorical_data, left_index=True, right_index=True)

# Split station id for merging later to air quality
meo_data_prepped["station_id"]= meo_data_prepped["station_id"].apply(lambda x: x.split("_")[0])

# Air quality data feature engineering
aq_data_train= aq_data.drop(["PM2.5" , "PM10", "O3"], axis=1)
aq_data_train["station_id"]= aq_data_train["stationId"].apply(lambda x: x.split("_")[0])
aq_data_train= aq_data_train.drop(["stationId"],axis=1)

train_data = pd.merge(meo_data_prepped, aq_data_train,  how='left', 
                     left_on=['station_id','utc_time'], right_on = ['station_id','utc_time'])
