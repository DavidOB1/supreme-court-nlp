import pandas as pd
import tensorflow as tf
import facts_preprocessing

def load_data():
  data = pd.read_csv("clean_data.csv")

  vectorizer = facts_preprocessing.create_vectorization_model(data["facts"])
  data["facts"] = data["facts"].apply(lambda x: facts_preprocessing.vectorize_string(x,vectorizer))