# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 18:06:47 2023

@author: rking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import audb
import opensmile
from sklearn.preprocessing import RobustScaler

from scipy.stats import normaltest
from scipy.stats import anderson

db = audb.load('emodb') #load database

df = db.tables['emotion'].df #load table with emotions


#OpenSmile package to extract audio features, select the configurations and the group of features

smile = opensmile.Smile(
   #feature_set=opensmile.FeatureSet.eGeMAPSv02,
   feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)


feats_df = smile.process_files(df.index) #load the features


# Normalize all data

feats_df_r = RobustScaler().fit(feats_df)
feats_df_n = feats_df_r.transform(feats_df)
feats_df2 = pd.DataFrame(feats_df_n)



#teste of normality
stats = []
cvalue = []
i=0
    
for columns in feats_df:
    res = anderson(feats_df2.iloc[:,i])
    i = i+1
    stats.append(res.statistic)
    cvalue.append(res.critical_values[2]) # choose 5% of significance level

norm = []
notnormind = []
normind = []
normname = []
notnormname = []
    
for i in range(len(cvalue)):
    if stats[i] < cvalue[i]:
        #normind.append(i)
        normind.append(feats_df2.columns[i])
        normname.append(feats_df.columns[i])
    else:
        notnormind.append(feats_df2.columns[i])
        notnormname.append(feats_df.columns[i])

notnormalselected = feats_df2.drop(normname,axis=1)

normalselected = feats_df2.drop(notnormname,axis=1)        

notnormalselectedn = feats_df.drop(normname,axis=1)

normalselectedn = feats_df.drop(notnormname,axis=1)  
        
print(normalselectedn.columns) #print the normal features

print(notnormalselectedn.columns) # print the not normal features