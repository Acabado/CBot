# Reinforcement Learning for Stock Market (Time-Series) Prediction
# We need a class for the agent that is making the actions
# Agent observes the environment (current state) and makes an action
# State changes because of this action (or just changes bc environment is constantly changing)
# Agent receives reward
# The three actions are buy, cash and hold
# Stock cannot be bought if stock is owned (simplifying the problem)
# Algorithm to maximize reward needed. Alpha = learning rate, Epsilon = exploration rate, gamma = discount rate
# episodes, epsilon_decay, epsilon_min

from collections import deque
from AlphaVantageAPI_wrapper import get_time_series
import random
import numpy as np

import keras
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Activation, GRU
from keras.optimizers import Adam

#REWARDS POLICY:
# For CASH, 0
# For BUY, P_t+1 - P_t - c   (c is the transaction fee to buy a stock)
# For HOLD, P_t+1 - P_t

#time interval is one day (we're dealing with closing figures)

train_start_date = "2009-01-01"
train_end_date = "2014-01-01"
test_start_date = "2015-01-01"
test_end_date = "2016-01-01"


class Agent:
    def __init__(self, state, state_size, model_name=""):
        self.state = state #Includes the price of prev x closing prices, time of day, sentiment?, other factors
        self.state_size = state_size
        self.model_name = model_name
        self.invent = "empty"
        self.actions = ["BUY", "CASH", "HOLD"] # CASH is either DO NOT BUY (if no stock is owned) 
        self.memory = deque(maxlen = 10000)    # , or SELL existing stock (if stock is owned)
        
        
        self.alpha = 0.001 # learning rate
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def _model(self): # to predict the q value when the Agent doesn't want to be adventerous.
        model = Sequential()    # In other words, when np.random.rand() < epsilon
        model.add(Dense(units=64, input_dim=self.state_size*1, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(GRU(units=8, dropout=0.25, recurrent_dropout=0.25))
        model.add(Dense(units=len(self.actions), activation="linear")) #why linear? don't ask idk yet
        model.compile(loss="mse", optimizer=Adam(lr=0.001)) #lr = learning rate
        
        return model
        
        
    def act(self):
        if np.random.rand() < self.epsilon:
            res = random.choice(self.actions)
            print(res)
            return res
        else:
            return #ask model to predict which of the three actions provides the greatest q-value and pick that

    def remember(self, state, action, reward, next_state, final_state):
        self.memory.append((state, action, reward, next_state, final_state))


def get_state(train_set, state_num):
    state = np.zeros(100)
    max_len = train_set.size - 100
    
    if state_num + 100 > max_len:
        return state

    for x in range(state_num, state_num + 100):
        state[x - state_num] = train_set[x]
    return state

def train(arr):
    #first we look at state
    #then we "act". How we act is based on how good future rewards will be (how do we incentivise this in code)
    #then we observe 
    return arr
 

def main():
  train_set = get_time_series("MSFT", train_start_date, train_end_date, output_size="full")
  print(train_set)

  ret = get_state(train_set, 256)
  print(ret)
    
    
if __name__ == "__main__":
    main()

