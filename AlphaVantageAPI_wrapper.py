#Python source module for chatbot that can answer queries specific to financial data (Using AlphaVantage's API)

import alpha_vantage
import requests
import json
import spacy
import nltk
from nltk.stem.snowball import SnowballStemmer
import random




NO_INTENT = 0
FIND_STOCK = 0x01
MULTI_STOCK = 0x02
FIND_CRYPTO = 0x03
COMPARE_CRYPTO = 0x04
FIND_FOREX_DEF = 0x05
FIND_FOREX_COMP = 0x06
FIND_SYMBOL = 0x07
Intent = NO_INTENT


#def converse()
#def findIntent()
#def findContext()





def get_info(intent, entity, time = 0):
        api_IDs = ["9BIXEUS4SXWELIF6", "1YCO", "BJUZBFVW3MCJMVEK", "ACN2C50AH4AXT6R9", "1UE1RVLUHR1J74IX", "1AL8HE1HSV09Z9G9",
        "5P3TDJ9HMA6STLWR", "SY77VO8AJ9FJ31J7", "NJ8EZR9UNGVZHYKS", "6UNG6F7RUTUH82ZN", "V3U92VZ0EI7LWG96", "9BCN9HYK75BZGIBI"]
        api_ID = random.choice(api_IDs)
        api_url_base = "https://www.alphavantage.co/query?"
        #https://2.python-requests.org/en/master/user/quickstart/#make-a-request
 

        if (intent == FIND_STOCK):
                data = {
                "function" : "GLOBAL_QUOTE",
                "symbol" : entity,
                "apikey" : api_ID  
                }    #datatype is json
                r = requests.get(api_url_base, params=data)
                json_dict = r.json()
                #print(r)
                #print(entity)
                val = json_dict["Global Quote"]["05. price"]
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
                try:         ############################################FIX THIS##################### AHHHHHH################
                        get_info(FIND_STOCK, val)
                except:
                        val = json_dict['bestMatches'][1]["1. symbol"]
                        get_info(FIND_STOCK, val)
                        
                
                return val
        #elif(intent == FIND_CRYPTO):










def main():
        #with open("intents.json") as json_data:
        #        intents = json.load(json_data)
        reply = get_info(FIND_SYMBOL, "MICROSOFT", 0)
        print(reply)


if __name__ == "__main__":
    main()