from numpy import vectorize
import pandas as pd
import tensorflow as tf
import facts_preprocessing
import keras
from keras import layers
from keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
from keras.preprocessing.text import Tokenizer


data = pd.read_csv("clean_data.csv",encoding="Windows-1252").dropna()

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
batch_size = 4
embedding_dim = 100
seq_length = 1000
#num_tokens = 21962
max_features = 75

def tokenize_with_keras(text_data):
  tokenizer = Tokenizer(num_words=max_features, split=' ',char_level=True)
  tokenizer.fit_on_texts(text_data.values)
  #X = tokenizer.texts_to_sequences(text_data.values)
  #X = pad_sequences(X, padding = "post",truncating="post",maxlen = seq_length)
  return tokenizer

def load_data():
  
  #vectorizer = facts_preprocessing.create_vectorization_model(data["facts"])
  vectorizer = tokenize_with_keras(data["facts"])
  #data["facts"] = data["facts"].apply(lambda x: facts_preprocessing.vectorize_string(x,vectorizer))
  data["facts"] = vectorizer.texts_to_sequences(data["facts"].values)
  #data["first_party"] = data["first_party"].apply(lambda x: facts_preprocessing.vectorize_string(x,vectorizer))
  #data["second_party"] = data["second_party"].apply(lambda x: facts_preprocessing.vectorize_string(x,vectorizer))
  data["first_party_won"] = data["first_party_won"].apply(lambda x: 1 if x == True else 0)
  data["ideologies"] = data["ideologies"].apply(lambda x: json.loads(x))

  return data, vectorizer

def get_complete_data():
  X = pad_sequences(data["facts"],maxlen=seq_length,padding="post", truncating="post")
  #FP = pad_sequences(data["first_party"],maxlen=20,padding="post", truncating="post")
  #SP = pad_sequences(data["second_party"],maxlen=20,padding="post", truncating="post")
  I = pad_sequences(data["ideologies"], maxlen = 9)
  Y = data["first_party_won"]
  return X, I, Y


embedding_matrix = np.zeros((max_features, embedding_dim))

# Prepare embedding matrix
def build_embedding(vectorizer):
  hits = 0
  misses = 0
  for word, i in vectorizer.wv.key_to_index.items():
      embedding_vector = vectorizer.wv[word]
      if embedding_vector is not None:
          # Words not found in embedding index will be all-zeros.
          # This includes the representation for "padding" and "OOV"
          embedding_matrix[i] = embedding_vector
          hits += 1
      else:
          misses += 1
  print("Converted %d words (%d misses)" % (hits, misses))

def build_model():
  text_input = layers.Input(shape=(seq_length))
  #embedding = layers.Embedding(num_tokens, embedding_dim, embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False,)(text_input)
  embedding = layers.Embedding(max_features, embedding_dim,input_length=seq_length)(text_input)
  conv_layer_1 = layers.Conv1D(1024,8,padding="same",activation="relu")(embedding)
  pool_1 = layers.MaxPool1D()(conv_layer_1)
  norm_1 = layers.BatchNormalization()(pool_1)
  conv_layer_2 = layers.Conv1D(2048,8,padding="same",activation="relu")(norm_1)
  pool_2 = layers.MaxPool1D(pool_size=2)(conv_layer_2)
  norm_2 = layers.BatchNormalization()(pool_2)
  conv_layer_3 = layers.Conv1D(2048,8,padding="same",activation="relu")(norm_2)
  text_pool = layers.GlobalMaxPool1D()(conv_layer_3)

  ideology_input = layers.Input(shape=(9))
  i_1 = layers.Dense(100,activation="relu")(ideology_input) 
  i_2 = layers.Dense(50,activation="relu")(i_1)
  i_3 = layers.Dense(20,activation="relu")(i_2)  

  
  first_party_input = layers.Input(shape=(20,100))
  fp_conv = layers.Conv1D(64,3,padding="same",activation="relu")(first_party_input)
  fp_pool = layers.GlobalAvgPool1D()(fp_conv)

  second_party_input = layers.Input(shape=(20,100))
  s_conv = layers.Conv1D(64,3,padding="same",activation="relu")(second_party_input)
  s_pool = layers.GlobalAvgPool1D()(s_conv)

  #combined = layers.concatenate([text_pool,fp_pool,s_pool,i_pool])
  combined = layers.concatenate([text_pool,i_3])

  dense_1 = layers.Dense(1024,activation="relu")(combined)
  dense_2 = layers.Dense(128,activation="relu")(dense_1)
  norm_d = layers.BatchNormalization()(dense_2)
  dense_3 = layers.Dense(16,activation="relu")(norm_d)
  output = layers.Dense(1,activation="sigmoid")(dense_3)

  #return keras.Model(inputs=(text_input,first_party_input,second_party_input,ideology_input),outputs=output)
  #tf.metrics.BinaryAccuracy(threshold=0.0)
  mod = keras.Model(inputs=(text_input, ideology_input),outputs=output)
  mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  mod.summary()
  return mod

def train_model():
  data, vec = load_data()
  print("loaded data")
  X, I, Y = get_complete_data()
  print("vectorized data")
  model = build_model()
  print("built model")
  #build_embedding(vec)
  #print("built embeddings")
  model.fit(x=(X,I), y = Y, batch_size=batch_size,epochs=10,validation_split=0.3)
  #model.predict(x=[])
  return model

def test_model(fact, ideology, vectorizer, model):
  model.predict(x=[facts_preprocessing.vectorize_string(fact,vectorizer), ideology])
