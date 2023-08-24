# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:13:10 2023

@author: rking

"""

import pandas as pd
import matplotlib.pyplot as plt
import audb
import opensmile
from scipy.stats import anderson
from sklearn.preprocessing import RobustScaler


db = audb.load('emodb') #Carrega a base de dados

df = db.tables['emotion'].df #cria uma tabela com as emoções


#df.emotion.value_counts().plot(kind='pie') # plota o gráfico de pizza do número de emoções da base


#chama a biblioteca que calcula os parametros baseado no GeMAPS
smile = opensmile.Smile(
   #feature_set=opensmile.FeatureSet.eGeMAPSv02,
   feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)


feats_df = smile.process_files(df.index) #faz a extracao dos parmateros

feats_df.shape #tamanho da matriz de paramteros



feats_df_r = RobustScaler().fit(feats_df)
feats_df_n = feats_df_r.transform(feats_df)
feats_df2 = pd.DataFrame(feats_df_n)
#feats_df = feats_df2


colors = {'happiness':'red', 'neutral':'green', 'sadness':'blue', 'fear':'yellow', 'boredom':'pink', 'disgust':'black', 'anger':'purple'}

#fig = plt.figure(figsize=(12, 12))
#ax = fig.add_subplot(projection='3d')

#ax.scatter(feats_df2[1], feats_df2[10], feats_df2[13], c=df['emotion'].map(colors))

#plt.show()

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
        
#plt.hist(feats_df.iloc[:,41])


feats_male = feats_df.drop(feats_df.index[female])

feats_female = feats_df.drop(feats_df.index[male])


## Selecionando parametros normais para audios masculinos

statsm = []
cvaluem = []
i = 0
    
for columns in feats_df:
    resm = anderson(feats_male.iloc[:,i])
    i = i+1
    statsm.append(resm.statistic)
    cvaluem.append(resm.critical_values[2])


normmind = []
normnamem = []
notnormmind = []
notnormnamem = []
    
for i in range(len(cvaluem)):
    if statsm[i] < cvaluem[i]:
        normmind.append(feats_male.columns[i])
        normnamem.append(feats_df.columns[i])
    else:
        notnormmind.append(feats_male.columns[i])
        notnormnamem.append(feats_male.columns[i])
        
#plt.hist(feats_df.iloc[:,41])

icaselectedm = []
pcaselectedm = []

icaselectedm = feats_male.drop(notnormmind,axis=1)

pcaselectedm = feats_male.drop(normmind,axis=1)


## Selecionando parametros normais para audios femininos

statsf = []
cvaluef = []
i = 0


for columns in feats_df:
    resf = anderson(feats_female.iloc[:,i])
    i = i+1
    statsf.append(resf.statistic)
    cvaluef.append(resf.critical_values[2])


normfind = []
normnamef = []
notnormfind = []
notnormnamef = []
    
for i in range(len(cvaluem)):
    if statsf[i] < cvaluef[i]:
        normfind.append(feats_female.columns[i])
        normnamef.append(feats_df.columns[i])
    else:
        notnormfind.append(feats_female.columns[i])
        notnormnamef.append(feats_df.columns[i])
        
#plt.hist(feats_df.iloc[:,41])

icaselectedf = []
pcaselectedf = []

icaselectedf = feats_female.drop(normfind,axis=1)

pcaselectedf = feats_female.drop(notnormfind,axis=1)


print("Number of female features with normal distribuition: ",len(normfind))
print("Number of male features with normal distribuition: ",len(normmind))


# Graoh of all datas, gender, and distribuition by gender



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
fracs_1 = [len(normfind), len(notnormfind)]

labels_2 = 'Normal', 'Not Normal'
fracs_2 = [len(normmind), len(notnormmind)]


fig, axs = plt.subplots(1, 3, figsize=(15, 5))


axs[0].pie(sizes_gender, labels=labels_gender, autopct=make_autopct(sizes_gender), startangle=90)
axs[0].set_title('Audio Gender')


axs[1].pie(fracs_1, labels=labels_1, autopct=make_autopct(fracs_1), startangle=90)
axs[1].set_title('Female Features Distribuition')


axs[2].pie(fracs_2, labels=labels_2, autopct=make_autopct(fracs_2), startangle=90)
axs[2].set_title('Male Features Distribution')


plt.tight_layout()


plt.show()