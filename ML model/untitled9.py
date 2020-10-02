# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:36:13 2019

@author: Dell
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils.vis_utils import model_to_dot



import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
from tensorflow.python.keras.backend import set_session
from keras.models import Sequential, load_model
MODEL_NAME="Menstruation_cycle_prediction"


# In[ ]:



dataset = pd.read_csv('C:/Users/Dell/Desktop/tempppp/data.csv')
dataset.dropna(axis=0)
dataset = dataset.replace(r'^\s+$', np.nan, regex=True)
dataset

data=dataset.drop(axis=1,labels=['MeanBleedingIntensity','MeanMensesLength','Group','CycleNumber','CycleWithPeakorNot','ReproductiveCategory','FirstDayofHigh','TotalNumberofHighDays','TotalHighPostPeak','TotalNumberofPeakDays','TotalDaysofFertility','TotalFertilityFormula','NumberofDaysofIntercourse','IntercourseInFertileWindow','UnusualBleeding','PhasesBleeding','IntercourseDuringUnusBleed','AgeM','Maristatus','MaristatusM','Yearsmarried','Wedding','Religion','ReligionM','Ethnicity','EthnicityM','Schoolyears','SchoolyearsM','OccupationM','IncomeM','Reprocate','Numberpreg','Livingkids','Miscarriages','Abortions','Medvits','Medvitexplain','Gynosurgeries','LivingkidsM','Boys','Girls','MedvitsM','MedvitexplainM','Urosurgeries','Breastfeeding','Method','Prevmethod','Methoddate','Whychart','Nextpreg','NextpregM','Spousesame','SpousesameM','Timeattemptpreg','MensesScoreDay12','MensesScoreDay13','MensesScoreDay14','MensesScoreDay15','MensesScoreDayNine','MensesScoreDayTen','MensesScoreDay11'])
data.isnull().sum()

data['MeanCycleLength'].fillna(method='ffill',inplace=True)
data['Age'].fillna(method='ffill',inplace=True)
data['Height'].fillna(method='ffill',inplace=True)
data['Weight'].fillna(method='ffill',inplace=True)
data['BMI'].fillna(method='ffill',inplace=True)

data['LengthofMenses'].fillna(method='ffill',inplace=True)
data['TotalMensesScore'].fillna(method='ffill',inplace=True)

data['MensesScoreDayOne'].fillna(value=0,inplace=True)
data['MensesScoreDayTwo'].fillna(value=0,inplace=True)
data['MensesScoreDayThree'].fillna(value=0,inplace=True)
data['MensesScoreDayFour'].fillna(value=0,inplace=True)
data['MensesScoreDayFive'].fillna(value=0,inplace=True)
data['MensesScoreDaySix'].fillna(value=0,inplace=True)
data['MensesScoreDayEight'].fillna(value=0,inplace=True)
data['MensesScoreDaySeven'].fillna(value=0,inplace=True)

data['LengthofLutealPhase'].fillna(value=13,inplace=True)
data['EstimatedDayofOvulation'].fillna(value=14,inplace=True)

data.isnull().sum()

data.reindex(np.random.permutation(data.index))



#Test and Train data

model_data=data.drop(axis=1,labels=['ClientID','LengthofCycle'])
y=data['LengthofCycle']
X_train, X_test, y_train, y_test = train_test_split(model_data, y, test_size=0.2)
print(list(X_train.columns.values))

