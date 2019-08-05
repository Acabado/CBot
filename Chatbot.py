#What is this project?

import spacy
import nltk
import numpy as np
import random
import keras
from keras.layers import Input, Dense, Activation
from keras.models import Model, load_model
from nltk.stem.snowball import SnowballStemmer
from Chatbot import get_info
import Chatbot

NO_INTENT = 0
FIND_STOCK = 0x01
MULTI_STOCK = 0x02
FIND_CRYPTO = 0x03
COMPARE_CRYPTO = 0x04
FIND_FOREX_DEF = 0x05
FIND_FOREX_COMP = 0x06
FIND_SYMBOL = 0x07


stemmer = SnowballStemmer(language="english")

# Import json file with all programmed intents and patterns.

import json
with open('C:/Users/varun/Desktop/Work/intents.json') as json_file:
    data = json.load(json_file) #returns a python dict

def stem_and_lower(list_words):
    list_words = [stemmer.stem(w.lower()) for w in list_words]
    return list_words

def load(data):
    nlp = spacy.load('en_core_web_sm')
    words = [] # This will be useful for our Bag of Words implementation
    intent_types = []
    documents = [] # List of (tokenized pattern, intent) tuples. Will be used when training the model
    ignore_words = ['!', '?']

    for intent in data['intents']:
        for pattern in intent['patterns']: # Loops through individuals sentences of that pattern
            # Tokenize each word in the sentence
            doc = nlp(pattern)
            w = [] # List of words in all patterns
            for token in doc:
                w.append(token.text)
            
            # Add all desired words to the word list
            words.extend(w)
            # Add this tuple to our "corpus"
            documents.append((w, intent['tag']))
            # Add to intent_types
        if intent['tag'] not in intent_types:
                intent_types.append(intent['tag'])

    # Stem and lower each word
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    return words, intent_types, documents


def train_prep(words, intent_types, documents): # We need to give our model something it can understand
    training = []
    
    # Use BoW encoding as dataset is small and domain specific.
    for doc in documents:
        # Initialize the bag of words 
        bow = []
        pattern_words = doc[0] # doc[0] is the list of words in a sample intent sentence
        pattern_words = stem_and_lower(pattern_words)

        for word in words:
            bow.append(1) if word in pattern_words else bow.append(0) # bow contains the Bag of Words translated sentence

        output_array = [0] * len(intent_types)
        output_array[intent_types.index(doc[1])] = 1
        output_list = list(output_array)
        
        training.append((bow, output_list))


    #shuffle and turn into array
    random.shuffle(training)
    training = np.array(training)
    
    train_x = [doc[0] for doc in training]
    train_y = [doc[1] for doc in training]
    return train_x, train_y


def clean_up_sentence(sentence):
    # Tokenize the pattern
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    sentence_words = []
    for token in doc:
        sentence_words.append(token.text)
    # Stem and Lowercase each word
    sentence_words = stem_and_lower(sentence_words)
    return sentence_words

def convert2bow(sentence, words): # Converts user input into bow format
    sentence = clean_up_sentence(sentence)
    bow = [0]*len(words)
    for word in sentence:
        if word in words:
            bow[words.index(word)] = 1
    return(np.array(bow))


ERROR_THRESHOLD = 0
 
def decipher_intent(sentence, words, intent_types, states, userID):
    model = load_model('chatbot_model1.h5')
    bag = convert2bow(sentence, words)
    bag = np.reshape(bag,(1,len(words))) # Fixed ValueError: expected dense_1_input to have shape (43,) but got array with shape (1,)
    results = model.predict(bag)[0]
    res = []

    for i,k in enumerate(results):
        if k > ERROR_THRESHOLD:
            res.append((i,k))
    
    temp = []
    #print(intent_types)
    #print(res)
    if states[userID] != "expect_target":
        res[11] = (11, 0)
    if states[userID] != "expect_entity":
        res[10] = (10, 0)
        
       

    for j in res:
        intent = intent_types[j[0]]
        temp.append((intent, j[1]))
    temp.sort(key=lambda x: x[1], reverse=True)

    return temp[0][0]

def respond(): # Tells chatbot what to respond with
    return

def process_sentence(sentence, intent, states, entityHolder, targetHolder, userID):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    entities = []
    ret = "none"
    for ent in doc.ents: 
        entities.append((ent.text, ent.label_))
    #We'll assume for now that the max number of entities is one (Comparison capabilities will be added later)
    if intent == "find_stock":
        symbol = get_info(FIND_SYMBOL, entities[0][0])
        val = get_info(FIND_STOCK, symbol)
        return val
            
        #print(symbol)
        #res = get_info(FIND_STOCK, symbol)
    elif intent == "find_symbol":
        symbol = get_info(FIND_SYMBOL, entities[0][0])
        print(symbol)
        return symbol
      
    return ret

    # The States dict works as follows: Each user can only be in one "state" at a time (hence the 1 to 1 mapping).
    # The different states are default, expect_entity(which means we expect an entity from the user),
    # entity_present and query present. 

def main():
    states = {}
    entityHolder = {}
    targetHolder = {}
    states['USER_1'] = "expect_entity"
    entityHolder['USER_1'] = "N/A"
    targetHolder['USER_1'] = "N/A"
    
    #print(states)
    words, intent_types, documents = load(data)
    train_x, train_y = train_prep(words, intent_types, documents)
    train_x = np.array(train_x) # Fixed 'list' object has no attribute 'shape' error <---
    train_y = np.array(train_y)
    
    #print(np.shape(train_x))
    #print(np.shape(convert2bow("Hello Bitcoin Litecoin Microsoft Google Thanks", words)))
    
    # Create and save our model
    '''model = keras.Sequential()
    model.add(Dense(8, input_dim=len(words)))
    model.add(Dense(8))
    model.add(Dense(len(train_y[0])))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=500, batch_size=8, verbose=1)
    model.save("chatbot_model1.h5")'''
    #intent = ""
    
    #sent = input("")
    sent = "Jeep"
    intent = decipher_intent(sent, words, intent_types, states, "USER_1")
    print(intent)

    print(process_sentence(sent, intent, states, entityHolder, targetHolder, "USER_1"))

    '''if intent != "say_hi" and intent != "say_bye":
        #find the entity and it's symbol (MSFT for Microsoft and BTC for Bitcoin)
        '''
    






if __name__ == "__main__":
    main()