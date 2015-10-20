import requests
import json

uid = 'iceui2'
devid = 'iceui2'
appid = 'ingredients'
api_key = 'ebnwnutvy3rvyw86z8cayxxh'
session_id = None
URL = "http://api.foodessentials.com"

def createsession():
     global session_id
     params = {'uid'     : uid,
               'devid'   : devid,
               'appid'   : appid,
               'f'       : 'json',
               'api_key' : api_key,
               }

     data = requests.get(URL + '/createsession', params=params).json()
     session_id = data['session_id']

def ingredientsearch(q, n=10, s=0):
     params = {'q' : q,
               'n' : n,
               's' : s,
               'sid' : session_id,
               'f' : 'json',
               'api_key' : api_key,
               }
     data = requests.get(URL + '/ingredientsearch', params=params).json()
     return data


def searchprods(q, n=1000, s=0):
     params = {'q' : q,
               'n' : n,
               's' : s,
               'sid' : session_id,
               'f' : 'json',
               'api_key' : api_key,
               }
     data = requests.get(URL + '/searchprods', params=params).json()
     return data


def labelarray(upc, n=1000, s=0):
     params = {'u' : upc,
               'n' : n,
               's' : s,
               'sid' : session_id,
               'f' : 'json',
               'api_key' : api_key,
               }
     data = requests.get(URL + '/labelarray', params=params).json()
     return data
