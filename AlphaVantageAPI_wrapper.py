#Python source module for chatbot that can answer queries specific to financial data (Using AlphaVantage's API)

import alpha_vantage
import requests
import json
import spacy
import nltk
from nltk.stem.snowball import SnowballStemmer
import random
import datetime
import numpy as np




NO_INTENT = 0
FIND_STOCK = 0x01
MULTI_STOCK = 0x02
FIND_CRYPTO = 0x03
COMPARE_CRYPTO = 0x04
FIND_FOREX_DEF = 0x05
FIND_FOREX_COMP = 0x06
FIND_SYMBOL = 0x07
Intent = NO_INTENT

api_IDs = ["6CHNDYO185X4Z7WT", "7H0W1931RHA3C9JI", "Q37OHKTGZWO1JX4Y", "EQBOMNU11PWHHX73", "S8L4OFNYQ6SN0MC5",
        "YIB8FOG7A8J693HB", "O02JCU824IZ4URBX", "1MVGFL6E66S9OPUU", "72UQ6SBEN8FNBUR4", "WE4M447GEFUWG8M9", "0CIVPUWC9Z1C5T9C"]


def get_info(intent, entity, time = 0):
        
        api_ID = random.choice(api_IDs)
        api_url_base = "https://www.alphavantage.co/query?"
        #https://2.python-requests.org/en/master/user/quickstart/#make-a-request
 

        if (intent == FIND_STOCK):
                while(True):
                        data = {
                        "function" : "GLOBAL_QUOTE",
                        "symbol" : entity,
                        "apikey" : random.choice(api_IDs)  
                        }    #datatype is json
                        r = requests.get(api_url_base, params=data)
                        json_dict = r.json()
                        if next(iter(json_dict)) == "Note": #This means API key limit has been hit <---
                                continue
                        val = json_dict["Global Quote"]["05. price"] 
                        break                                           #Fixed both API key limit bug and MICROSOFT bug
                return val
        elif (intent == FIND_SYMBOL):
                data = {
                "function" : "SYMBOL_SEARCH",
                "keywords" : entity,
                "apikey" : api_ID
                }
                r = requests.get(api_url_base, params=data)
                json_dict = r.json()
                val = json_dict['bestMatches'][0]["1. symbol"]
                try:         
                        get_info(FIND_STOCK, val)
                except:
                        val = json_dict['bestMatches'][1]["1. symbol"]
                        get_info(FIND_STOCK, val)
                        
                
                return val


def get_time_series(entity, start, end, output_size="compact", datatype="json"):
        api_ID = random.choice(api_IDs)
        api_url_base = "https://www.alphavantage.co/query?"

        data = {
        "function" : "TIME_SERIES_DAILY",
        "symbol" : entity,
        "outputsize" : output_size,
        "datatype" : datatype,
        "apikey" : api_ID
        }

        r = requests.get(api_url_base, params=data)
        json_dict = r.json()
        #for date in json_dict["Time Series (Daily)"][0]:
        val = json_dict["Time Series (Daily)"]

        start_date = datefromstr(end)
        end_date = datefromstr(start)

        dates = []
        for key in val:
                d_key = datefromstr(key)
                if d_key > start_date:
                        continue
                elif d_key <= start_date and d_key >= end_date:
                        dates.append(key)
                elif d_key < end_date:
                        break
        dates.reverse()
        arr = np.zeros(len(dates))
        for x in range(0, len(dates)):
                arr[x] = json_dict["Time Series (Daily)"][dates[x]]["4. close"]

        return arr


def datefromstr(string):
        return datetime.date.fromisoformat(string)

def main():
        #reply = get_info(FIND_SYMBOL, "tesla", 0)
       
        val = get_time_series("MSFT", "2019-01-01", "2019-08-30", output_size="full")     
        print(val)


if __name__ == "__main__":
    main()