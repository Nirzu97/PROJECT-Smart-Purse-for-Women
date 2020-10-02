# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 17:22:00 2019

@author: Dell
"""

import pyrebase
config = {
        "apiKey": "AIzaSyDZ8-ZU-rIrdiUFwOGdidR0p4OMdtvG6ng",
        "authDomain": "smart-purse-for-women.firebaseapp.com",
        "databaseURL": "https://smart-purse-for-women.firebaseio.com",
        "storageBucket": "smart-purse-for-women.appspot.com",
        "serviceAccount": "smart-purse-for-women-firebase-adminsdk-kiw8g-45db989e2d.json"
}
firebase = pyrebase.initialize_app(config)

auth = firebase.auth()
#authenticate a user
user = auth.sign_in_with_email_and_password("shailly7035@gmail.com", "123456")

def store_date(user_id,date,timestamp):
    data = {
            'user_id' : str(user_id),
            'date' : str(date),
            'time' : str(timestamp)
            }
    result = firebase.post('/smart-purse-for-women/PreviousDate', data)
    print(result)