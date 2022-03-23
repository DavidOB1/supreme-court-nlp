import string
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import pandas as pd

# Read in .csv file
cases = pd.read_csv("")  # insert path to .csv file

# Assumes facts column is called 'facts'
raw_text = ' '.join(cases['facts'])  # entire dataset text corpus

formatted_text = []  # will contain entirety of raw text in the required format for Word2Vec

# Iterates through all sentences in raw text
for sent in sent_tokenize(raw_text):
    token_sentence = []

    # tokenizes each sentence into list of words, excluding punctuation
    # required format for building Word2Vec model
    for word in word_tokenize(sent):
        lower_word = word.lower()
        if lower_word[0] not in string.punctuation:
            token_sentence.append(lower_word)
    formatted_text.append(token_sentence)

# Model created and trained on all the facts sections of each case
model = Word2Vec(formatted_text, min_count=1, window=5)



# Model created, run following on each facts section
# Loop through each case in cases, which represents all cases

# Loops through pandas dataframe rows
# May need to alter depending on formatting of data
for case in cases.itertuples():
    facts = case.facts  # insert facts of the case
    word_vectors = []

    # Vectorizes each word, excluding punctuation
    for word in word_tokenize(facts):
        if word not in string.punctuation:
            word_vectors.append(model.wv[word])

    # word_vectors now contains list of word vectors for this facts section
