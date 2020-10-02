import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pandas as pd
import re

# Fetch the service account key JSON file contents
cred = credentials.Certificate('smart-purse-for-women-firebase-adminsdk-kiw8g-45db989e2d.json')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smart-purse-for-women.firebaseio.com'
})

# As an admin, the app has access to read and write all data, regradless of Security Rules
ref = db.reference('smart-purse-for-women')
users_ref = ref.child('PreviousDate')
features_ref = ref.child("Features")

def get_previous_date(id):
    result = users_ref.child(str(id)).get()
    if result:
        print(result['date'])
        return(result['date'])
    else:
        return -1

def store_date(user_id,date,timestamp):
    data = {
            'user_id' : str(user_id),
            'date' : str(date),
            'time' : str(timestamp)
            }
    if get_previous_date(user_id):
        result = users_ref.child(user_id).update(data)
    else:
        result = users_ref.child(user_id).set(data)

#store_date("5","22-11-2019",str(datetime.now())) 
       

features ={'Age': '21', 'BMI': '0.5034013605442177', 'Estimated Day of Ovulation': '12', 'Height': '147', 'Length of Luteal Phase': '11', 'Length of Menses': '5', 'Mean Cycle Length': '28', 'Menses Score Day Eight': '0', 'Menses Score Day Five': '1', 'Menses Score Day Four': '2', 'Menses Score Day One': '3', 'Menses Score Day Seven': '0', 'Menses Score Day Six': '0', 'Menses Score Day Three': '3', 'Menses Score Day Two': '5', 'Total Menses Score': '0', 'Weight': '74', 'timestamp': '2019-10-20 23:34:53.930209'}        
#get_previous_date(1)
     
def store_features(features,user_id):
    features_ref.child(str(user_id) + '/'+str(re.sub('\W+','',str(datetime.now())))).set(features)
    
#store_features(features,"5")   
def get_features(user_id):
    result = features_ref.child(str(user_id)).order_by_child("timestamp").limit_to_last(1).get()
    result = pd.DataFrame.from_dict(result)
    new_index = ['Mean Cycle Length', 'Estimated Day of Ovulation', 'Length of Luteal Phase', 'Length of Menses', 'Menses Score Day One', 'Menses Score Day Two', 'Menses Score Day Three', 'Menses Score Day Four', 'Menses Score Day Five', 'Menses Score Day Six', 'Menses Score Day Seven', 'Menses Score Day Eight', 'Total Menses Score', 'Age', 'Height', 'Weight', 'BMI']
    result = result.reindex(new_index)
    std = StandardScaler()
    result = std.fit_transform(result)
    result = pd.DataFrame(result)
    result =result.transpose()
    return(result)
get_features("5")