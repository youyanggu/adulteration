import requests
import json

uid = 'iceui3'
devid = 'iceui3'
appid = 'ingredients'
#api_key = 'ebnwnutvy3rvyw86z8cayxxh'
api_key = 'etvxk3gnmvsm56kqv644uqks'
session_id = None
URL = "http://api.foodessentials.com"
calls = 0

def counter():
     global calls
     calls += 1

def reset_calls():
     global calls
     calls = 0

def get_calls():
     return calls

def multiple_tries(func, times, exceptions):
    for _ in range(times):
        try:
            return func()
        except Exception as e:
            if not isinstance(e, exceptions):
                raise # reraises unexpected exceptions 
    raise # reraises if attempts are unsuccessful

def get_session_id():
     return session_id

def start_session():
     counter()
     params = {'sid' : session_id,
               'f' : 'json',
               'api_key' : api_key,
               }
     response = requests.get(URL + '/getprofile', params=params)
     try:
          response.json()
     except ValueError:
          createsession()


def createsession():
     global session_id
     counter()
     params = {'uid'     : uid,
               'devid'   : devid,
               'appid'   : appid,
               'f'       : 'json',
               'api_key' : api_key,
               }

     data = requests.get(URL + '/createsession', params=params).json()
     session_id = data['session_id']

def ingredientsearch(q, n=10, s=0):
     counter()
     params = {'q' : q,
               'n' : n,
               's' : s,
               'sid' : session_id,
               'f' : 'json',
               'api_key' : api_key,
               }
     tries = 5 if n<100 else 2
     func = requests.get(URL + '/ingredientsearch', params=params).json
     data = multiple_tries(func, tries, ValueError)
     return data


def searchprods(q, n=1000, s=0):
     counter()
     params = {'q' : q,
               'n' : n,
               's' : s,
               'sid' : session_id,
               'f' : 'json',
               'api_key' : api_key,
               }
     tries = 5 if n<100 else 2
     func = requests.get(URL + '/searchprods', params=params).json
     data = multiple_tries(func, tries, ValueError)
     return data


def labelarray(upc, n=1000, s=0):
     counter()
     params = {'u' : upc,
               'n' : n,
               's' : s,
               'sid' : session_id,
               'f' : 'json',
               'api_key' : api_key,
               }
     tries = 5 if n<100 else 2
     func = requests.get(URL + '/labelarray', params=params).json
     data = multiple_tries(func, tries, ValueError)
     return data
