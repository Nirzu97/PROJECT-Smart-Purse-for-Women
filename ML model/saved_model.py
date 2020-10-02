# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 02:42:56 2019

@author: Dell
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
MODEL_NAME="Menstruation_cycle_prediction"

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

model_data=data.drop(axis=1,labels=['ClientID','LengthofCycle'])
y=data['LengthofCycle']
X_train, X_test, y_train, y_test = train_test_split(model_data, y, test_size=0.2)

std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)

print(list(model_data.columns))


n_hidden_1 = 9 
n_hidden_2 = 5 
n_hidden_3 = 3 
n_input = 17 
output_layer = 1 

def multilayer_perceptron():
    model = Sequential()
    model.add(Dense(n_hidden_1, activation=tf.nn.relu, input_dim=n_input,name="layer1")),
    model.add(Dense(n_hidden_2, activation=tf.nn.relu,name="layer2")),
    model.add(Dense(n_hidden_3, activation=tf.nn.relu,name="layer3")),
    model.add(Dense(output_layer,name="layer4"))
    optimizer = keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mean_squared_error',optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = multilayer_perceptron()

best_model_file = "prediction.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)

print('Training model...')
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000
history = model.fit(
  X_train, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot(),best_model])
print('Traing finished.')

best_model_file = "prediction.h5"
print('Loading the best model...')
model = load_model(best_model_file)
print('Best Model loaded!')

test_predictions = model.predict(X_test).flatten()

print(X_test[20],test_predictions[20])

acc=model.evaluate(X_test, y_test, verbose=0)
print(acc)

# serialize model to JSON
model_json = model.to_json()
with open("new_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("new_model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('new_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("new_model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
optimizer = keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, amsgrad=False)
loaded_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))






