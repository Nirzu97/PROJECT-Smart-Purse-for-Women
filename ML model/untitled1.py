# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 00:29:38 2019

@author: Dell
"""

from predictionmodel import *

dataset = pd.read_csv('C:/Users/Dell/Desktop/GUI/databook.csv')
value = model.predict(dataset)
print(value,"\n")