import pandas as pd
import tensorflow as tf
import facts_preprocessing
import keras
from keras import layers
from keras.preprocessing.sequence import pad_sequences
import json


data = pd.read_csv("clean_data.csv",encoding="Windows-1252")

# what we're going to use:
# facts, first_party_won, first_party, second_party, issue_area, ideologies

# each fact -> vector of shape(250,50) (250 words long)
# ideology -> vector of shape(9)
# first_party_won -> 1 = True, 0 = False (no processing needed)
# first_party -> vector, maybe character-based
# second_party -> vector, maybe character-based

# issue_area -> vector of size (2,50); issue is typically 2 words long

# output:
# first_party_won


def load_data():
  
  vectorizer = facts_preprocessing.create_vectorization_model(data["facts"])
  data["facts"] = data["facts"].apply(lambda x: facts_preprocessing.vectorize_string(x,vectorizer))
  data["first_party"] = data["first_party"].apply(lambda x: facts_preprocessing.vectorize_string(x,vectorizer))
  data["second_party"] = data["second_party"].apply(lambda x: facts_preprocessing.vectorize_string(x,vectorizer))
  data["first_party_won"] = data["first_party_won"].apply(lambda x: 1 if x == True else 0)
  data["ideologies"] = data["ideologies"].apply(lambda x: json.loads(x))

  return data, vectorizer

def get_complete_data():
  X = pad_sequences(data["facts"],maxlen=250,padding="post", truncating="post")
  FP = pad_sequences(data["first_party"],maxlen=20,padding="post", truncating="post")
  SP = pad_sequences(data["second_party"],maxlen=20,padding="post", truncating="post")
  I = pad_sequences(data["ideologies"], maxlen = 9)
  Y = data["first_party_won"]

batch_size = 4

def build_model():
  text_input = layers.Input(shape=(250,100))
  conv_layer_1 = layers.Conv1D(64,3,padding="same",activation="relu")(text_input)
  pool_1 = layers.MaxPool1D()(conv_layer_1)
  conv_layer_2 = layers.Conv1D(64,3,padding="same",activation="relu")(pool_1)
  pool_2 = layers.MaxPool1D(pool_size=2)(conv_layer_2)
  conv_layer_3 = layers.Conv1D(128,3,padding="same",activation="relu")(pool_2)
  text_pool = layers.GlobalMaxPool1D()(conv_layer_3)

  ideology_input = layers.Input(shape=(9))
  i_pool = layers.Dense(100)(ideology_input) 

  
  first_party_input = layers.Input(shape=(20,100))
  fp_conv = layers.Conv1D(64,3,padding="same",activation="relu")(first_party_input)
  fp_pool = layers.GlobalAvgPool1D()(fp_conv)

  second_party_input = layers.Input(shape=(20,100))
  s_conv = layers.Conv1D(64,3,padding="same",activation="relu")(second_party_input)
  s_pool = layers.GlobalAvgPool1D()(s_conv)

  combined = layers.concatenate([text_pool,fp_pool,s_pool,i_pool])

  dense_1 = layers.Dense(32,activation="relu")(combined)
  dense_2 = layers.Dense(16,activation="relu")(dense_1)
  dense_3 = layers.Dense(8,activation="relu")(dense_2)
  output = layers.Dense(1,activation="sigmoid")(dense_3)

  return keras.Model(inputs=(text_input,first_party_input,second_party_input,ideology_input),outputs=output)
