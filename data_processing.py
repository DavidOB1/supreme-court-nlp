import nltk
import pandas as pd
import re

# this file handles the legal data and produces a dataset that can be directly put into an ML library

data = pd.read_csv("justice.csv")

# this subset is just the first party, second party, facts, the result, and the area of debate 
data[data.columns[[6,7,8,12,15]]]

# TODO
'''
1. handle the given data
  1a. look at bigrams/ngrams
  1b. possibly remove punctuation
  1c. remove sentences within the facts column that directly say what the result is
2. augment the data with:
  2a. relevant law
    allows us to frame the question as: given this case and given this law, what's the right answer?
    2ai. need a tagging mechanism to find the actual law from the document
    2aii. this is where the use of GPT-2 might be helpful
  2b. the entire U.S. Code? 2a is likely the better option
  2c. political leaning of the judges

'''
# this function will clean a given fact
def clean_facts(fact):
  #remove the <p> from each fact
  return fact[3:-5]

# cleans up the facts of the cases
data["facts"] = data["facts"].apply(lambda x: clean_facts(x))
