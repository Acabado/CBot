#What is this project?

import spacy
import nltk
import numpy as np
import random
import keras
from keras.layers import Input, Dense, Activation
from keras.models import Model, load_model
from nltk.stem.snowball import SnowballStemmer


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


ERROR_THRESHOLD = 0.25
 
def decipher_intent(sentence, words, intent_types):
    model = load_model('chatbot_model1.h5')
    bag = convert2bow(sentence, words)
    bag = np.reshape(bag,(1,43)) # Fixed ValueError: expected dense_1_input to have shape (43,) but got array with shape (1,)
    results = model.predict(bag)[0]
    res = []

    for i,k in enumerate(results):
        if k > ERROR_THRESHOLD:
            res.append((i,k))
    
    temp = []
    for j in res:
        intent = intent_types[j[0]]
        temp.append((intent, j[1]))
    temp.sort(key=lambda x: x[1], reverse=True)
    return temp

def respond(): # Tells chatbot what to respond with
    return

def main():
    words, intent_types, documents = load(data)
    train_x, train_y = train_prep(words, intent_types, documents)
    train_x = np.array(train_x) # Fixed 'list' object has no attribute 'shape' error <---
    train_y = np.array(train_y)
    
    print(np.shape(train_x))
    print(np.shape(convert2bow("Hello Bitcoin Litecoin Microsoft Google Thanks", words)))
    
    # Create and save our model
    '''model = keras.Sequential()
    model.add(Dense(8, input_dim=len(words)))
    model.add(Dense(8))
    model.add(Dense(len(train_y[0])))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=500, batch_size=8, verbose=1)
    model.save("chatbot_model1.h5")'''
    
    print(decipher_intent("Find the stock price of Apple", words, intent_types))
    






if __name__ == "__main__":
    main()
