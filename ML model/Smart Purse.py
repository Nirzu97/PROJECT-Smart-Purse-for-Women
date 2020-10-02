
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen,NoTransition
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Rectangle, Color 
import datetime as dt
from datetime import datetime
from collections import OrderedDict
import csv

import pandas as pd
from firebase import firebase
from firebaseretreivedate import store_date,get_previous_date,store_features,get_features

import keras
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json

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

kv = Builder.load_file("my.kv")

user_id="5"

features = dict()

class CanvasWidget1(Screen):
    def on_enter(self):
        sm.current = "AllOptions"

class CanvasWidget2(Screen):
    def on_enter(self):
        sm.current = "CalendarScreen"
        
class CanvasWidget3(Screen):
    def on_enter(self):
         sm.current = "Screen1"
        
class CanvasWidget4(Screen):
    def on_enter(self):
        sm.current  = "PurseConfiguration"
        
class CanvasWidget5(Screen):
    def on_enter(self):
        sm.current  = "Screen2"
        
class CanvasWidget6(Screen):
    def on_enter(self):
        sm.current  = "Screen3"

class CanvasWidget7(Screen):
    def on_enter(self):
        sm.current  = "Screen4"
    

#page1    
class EnterPreviousDate(Screen):
    date=ObjectProperty(None)
  
    def storeDate(self):
        
        previousdate=self.date.text
        date = str(datetime.now())
        store_date("4",previousdate,date)
        self.reset()
        sm.current = "CanvasWidget1"
        
        
    def reset(self):
        self.date.text=""
        

#page2
class AllOptions(Screen):
    
    def showdates(self):
        sm.current = "CanvasWidget2"
        
        
    def trackhealth(self):
        sm.current = "CanvasWidget3"
       
    
    def configPurse(self):
        sm.current = "CanvasWidget4"
        


class CalendarScreen(Screen):
    
    date1=ObjectProperty(None)
    date2=ObjectProperty(None)
    date3=ObjectProperty(None)
    date4=ObjectProperty(None)
    date5=ObjectProperty(None)
    
    def on_enter(self):
        self.predictDates()
    
    def predictDates(self):
       
        now = str(get_previous_date(user_id))
        dataset = get_features(user_id)
        value = int(loaded_model.predict(dataset))
        print("value: " + str(value))
        print("date" + str(now))
        date_select = datetime.strptime(now, '%d-%m-%Y')
        date=date_select + dt.timedelta(days=value)
        strdate=date.strftime('%d-%m-%Y')
        self.date1.text=strdate
        
        date=date + dt.timedelta(days=value)
        strdate=date.strftime('%d-%m-%Y')
        self.date2.text=strdate
        
        date=date + dt.timedelta(days=value)
        strdate=date.strftime('%d-%m-%Y')
        self.date3.text=strdate
        
        date=date + dt.timedelta(days=value)
        strdate=date.strftime('%d-%m-%Y')
        self.date4.text=strdate
        
        date=date + dt.timedelta(days=value)
        strdate=date.strftime('%d-%m-%Y')
        self.date5.text=strdate
        
    def get_back(self):
        sm.current = "CanvasWidget1"
        
    
class PurseConfiguration(Screen):
    label_date = ObjectProperty(None)
    def on_enter(self):
        now = str(get_previous_date(user_id))
        dataset = get_features(user_id)
        value = int(loaded_model.predict(dataset))
        date_select = datetime.strptime(now, '%d-%m-%Y')
        date=date_select + dt.timedelta(days=value)
        strdate=date.strftime('%d-%m-%Y')
        self.label_date.text=strdate
        
    def on_spinner_select(self,text):
        sm.current = "CanvasWidget1"
        #data

#page3
class Screen1(Screen):
    f1 = ObjectProperty(None)
    f2 = ObjectProperty(None)
    f3 = ObjectProperty(None)
    f4 = ObjectProperty(None)
    def nextScreen2(self):
        features['Mean Cycle Length'] = self.f1.text
        features['Estimated Day of Ovulation'] = self.f2.text
        features['Length of Luteal Phase'] = self.f3.text
        features['Length of Menses'] = self.f4.text
        sm.current = "CanvasWidget5"

class Screen2(Screen):
    f5 = ObjectProperty(None)
    f6 = ObjectProperty(None)
    f7 = ObjectProperty(None)
    f8 = ObjectProperty(None)
    
    def nextScreen3(self):
        features['Menses Score Day One'] = self.f5.text
        features['Menses Score Day Two'] = self.f6.text
        features['Menses Score Day Three'] = self.f7.text
        features['Menses Score Day Four'] = self.f8.text
        sm.current = "CanvasWidget6"
        
class Screen3(Screen):
    f9 = ObjectProperty(None)
    f10 = ObjectProperty(None)
    f11 = ObjectProperty(None)
    f12 = ObjectProperty(None)
    def nextScreen4(self):
        features['Menses Score Day Five'] = self.f9.text
        features['Menses Score Day Six'] = self.f10.text
        features['Menses Score Day Seven'] = self.f11.text
        features['Menses Score Day Eight'] = self.f12.text
        sm.current = "CanvasWidget7"

class Screen4(Screen):
    f13 = ObjectProperty(None)
    f14 = ObjectProperty(None)
    f15 = ObjectProperty(None)
    f16 = ObjectProperty(None)
    def submitData(self):
        features['Total Menses Score'] = self.f13.text
        features['Age'] = self.f14.text
        features['Height'] = self.f15.text
        features['Weight'] = self.f16.text
        features['BMI'] = str(float(self.f16.text)/float(self.f15.text))
        features['timestamp'] = date = str(datetime.now())
        store_features(features,user_id)
        sm.current =  "CanvasWidget1"

sm = ScreenManager(transition = NoTransition())
screens = [EnterPreviousDate(name="EnterPreviousDate"),AllOptions(name = "AllOptions"),CalendarScreen(name = "CalendarScreen"),Screen1(name = "Screen1"),Screen2(name = "Screen2"),Screen3(name = "Screen3"),Screen4(name = "Screen4"), PurseConfiguration(name = "PurseConfiguration"), CanvasWidget1(name = "CanvasWidget1"),CanvasWidget2(name = "CanvasWidget2"),CanvasWidget3(name = "CanvasWidget3"),CanvasWidget4(name = "CanvasWidget4"),CanvasWidget5(name = "CanvasWidget5"),CanvasWidget6(name = "CanvasWidget6"),CanvasWidget7(name = "CanvasWidget7")]
for screen in screens:
    sm.add_widget(screen)

sm.current = "EnterPreviousDate"
class MyMainApp(App):
    def build(self):
        return sm

if __name__ == "__main__":
    MyMainApp().run()
