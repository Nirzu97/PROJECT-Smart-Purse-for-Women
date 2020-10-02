# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 17:35:05 2019

@author: Dell
"""


import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("smart-purse-for-women-firebase-adminsdk-kiw8g-45db989e2d.json")
firebase_admin.initialize_app(cred)
db=firestore.client()
