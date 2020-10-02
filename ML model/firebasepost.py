

from firebase import firebase
from datetime import datetime
firebase = firebase.FirebaseApplication("https://smart-purse-for-women.firebaseio.com/", None)
def save_user_key(user_id):
    data={'user_id': user_id,
          'key' : key}
    
def get_all_userkey(user_id):
    alldata=firebase.get('smart-purse-for-women/PreviousDate',None)
    
def store_date(user_id,date,timestamp):
    data = {
            'user_id' : str(user_id),
            'date' : str(date),
            'time' : str(timestamp)
            }

    result = firebase.post('/smart-purse-for-women/PreviousDate', data)
    print(result)
    
def get_previous_date(user_id):
    result = firebase.get('smart-purse-for-women/PreviousDate',user_id)
    print(result)
        

get_previous_date("2")

def update_previous_date(user_id,date):
    #dict key after /
    #,key,changedvale
    firebase.put('/smart-purse-for-women/PreviousDate')

def delete_previous_date(user_id):
    firebase.delete('/smart-purse-for-women/PreviousDate')