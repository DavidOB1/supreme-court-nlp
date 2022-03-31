import string
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec, FastText
import pandas as pd

# removes punctuation from a string
def remove_punc(s):
    return "".join(x for x in s if x not in string.punctuation)


# creates a model to vectorize texts
# expects a list of strings, returns a vectorization model
def create_vectorization_model(texts):
    raw_text = ' '.join(texts)
    formatted_text = []

    # Iterates through all sentences in raw text
    for sent in sent_tokenize(raw_text):
        formatted_text.append(word_tokenize(remove_punc(sent.lower())))

    model = Word2Vec(formatted_text, min_count=1, window=5)
    return model

# Model created, run following on each facts section
# Loop through each case in cases, which represents all cases

# vectorizes a single string using a given model
def vectorize_string(s, model):
    word_vectors = []
    #s = remove_punc(s)
    # Vectorizes each word, excluding punctuation
    for sent in sent_tokenize(s):
        list_of_words = word_tokenize(remove_punc(sent.lower()))
        for word in list_of_words:
            try:
                vector = model.wv.key_to_index[word.lower()]
                word_vectors.append(vector)
            except KeyError:
                pass
                #print(word + " not found!")
                #print("it was in: " + s + "\n")
        
    return word_vectors
