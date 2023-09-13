# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:02:50 2023

@author: RafaelKingeski
"""

#from sklearnex import patch_sklearn
#patch_sklearn()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import audb
import opensmile
from scipy.stats import anderson
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn import svm
from sklearn.model_selection import cross_val_score

db = audb.load('emodb') #load database

df = db.tables['emotion'].df #load table with emotions


#OpenSmile package to extract audio features, select the configurations and the group of features
smile = opensmile.Smile(
   #feature_set=opensmile.FeatureSet.eGeMAPSv02,
   feature_set=opensmile.FeatureSet.ComParE_2016,
   feature_level=opensmile.FeatureLevel.Functionals,
)


feats_df = smile.process_files(df.index) #load the features


# Normalize data robust scaler
'''
from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
Scale features using statistics that are robust to outliers.

This Scaler removes the median and scales the data according 
to the quantile range (defaults to IQR: Interquartile Range). 
The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).

Centering and scaling happen independently on each feature by computing the relevant statistics
on the samples in the training set. Median and interquartile range are then stored to be used 
on later data using the transform method.

Standardization of a dataset is a common requirement for many machine learning estimators.
Typically this is done by removing the mean and scaling to unit variance.
However, outliers can often influence the sample mean / variance in a negative way. 
In such cases, the median and the interquartile range often give better results.
'''

def normalize(data):
    
    data_r = RobustScaler().fit(data)
    data_n = data_r.transform(data)
    data_normalized = pd.DataFrame(data_n, columns = data.columns)

    return data_normalized

#normalize all features

feats_dfn = normalize(feats_df)


#separates parameters into male and female

datadb = db["files"]["speaker"].get(index=db["emotion"].index, map="gender")

datadb = datadb.map({'male': 'Male', 'female': 'Female'})

gender = pd.DataFrame(datadb)


male= []
female =[]

for i in range(len(df)):
    if gender.iloc[i][0] == 'Male':
        male.append(i)
    else:
        female.append(i)
        

feats_male = feats_df.drop(feats_df.index[female])

feats_female = feats_df.drop(feats_df.index[male])




##function to test the normality

def distribuition(data, confiability):

    if (confiability == 85):
        conf = 0
    if (confiability == 90):
        conf = 1
    if (confiability == 95):
        conf = 2
    if (confiability == 97.2):
        conf = 3
    if (confiability == 99):
        conf = 4
        
        
    stats = []
    cvalue = []
    i=0
    
    for columns in data:
        res = anderson(data.iloc[:,i])
        stats.append(res.statistic)
        cvalue.append(res.critical_values[conf])
        i=i+1
    
    norm_index = []
    norm_name = []
    notnorm_index = []
    notnorm_name = []
    k=0
    j=0    
    
    for i in range(len(cvalue)):
        if stats[i] < cvalue[i]:
            norm_name.append(data.columns[i])
            norm_index.append(data.columns.get_loc(norm_name[j]))
            j=j+1
        else:
            notnorm_name.append(data.columns[i])
            notnorm_index.append(data.columns.get_loc(notnorm_name[k]))
            k=k+1
            
    return norm_index, norm_name, notnorm_index, notnorm_name


#normality for all features

norm_index, norm_name, notnorm_index, notnorm_name = distribuition(feats_df,95)



## Distribuition test for male features

norm_index_male, norm_name_male, notnorm_index_male, notnorm_name_male = distribuition(feats_male,95)


## Distribuition test for male features

norm_index_female, norm_name_female, notnorm_index_female, notnorm_name_female = distribuition(feats_female,95)


'''
icaselectedf = []
pcaselectedf = []

icaselectedf = feats_female.drop(normfind,axis=1)

pcaselectedf = feats_female.drop(notnormfind,axis=1)
'''

print("Number of features with normal distribuition for male and female: ",len(norm_index))
print("Number of female features with normal distribuition: ",len(norm_index_male))
print("Number of male features with normal distribuition: ",len(norm_index_male))

print("Number of features with not_normal distribuition for male and female: ",len(notnorm_index))
print("Number of female features with not-normal distribuition: ",len(notnorm_index_male))
print("Number of male features with not-normal distribuition: ",len(notnorm_index_male))


## Graph of all datas, gender, and distribuition by gender



# make the percentage and values for plot

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


gender_counts = gender['gender'].value_counts()
labels_gender = gender_counts.index
sizes_gender = gender_counts.values

labels_1 = 'Normal', 'Not Normal'
fracs_1 = [len(norm_index_female), len(notnorm_index_female)]

labels_2 = 'Normal', 'Not Normal'
fracs_2 = [len(norm_index_male), len(notnorm_index_male)]


fig, axs = plt.subplots(1, 3, figsize=(15, 5))


axs[0].pie(sizes_gender, labels=labels_gender, autopct=make_autopct(sizes_gender), startangle=90)
axs[0].set_title('Audio Gender')


axs[1].pie(fracs_1, labels=labels_1, autopct=make_autopct(fracs_1), startangle=90)
axs[1].set_title('Female Features Distribuition')


axs[2].pie(fracs_2, labels=labels_2, autopct=make_autopct(fracs_2), startangle=90)
axs[2].set_title('Male Features Distribution')


plt.tight_layout()


plt.show()


#ordering the features by variance


vardata = feats_df.var()

ordervar = vardata.sort_values(ascending=False)


ordervar = pd.DataFrame(ordervar)

feats_df_ordbyvar = feats_df[ordervar.index]


#robsut scaler normalization in features ordered by variance

feats_dfn_ordbyvar = normalize(feats_df_ordbyvar)


#male features normalized

feats_dfn_male_normaldist = feats_dfn.drop(notnorm_name_male,axis=1)

feats_dfn_male_notnormaldist = feats_dfn.drop(norm_name_male,axis=1)

#female features normalized

feats_dfn_female_normaldist = feats_dfn.drop(notnorm_name_female,axis=1)

feats_dfn_female_notnormaldist = feats_dfn.drop(norm_name_female,axis=1)



