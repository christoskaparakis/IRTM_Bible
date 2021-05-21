from os import path
import pandas as pd
<<<<<<< Updated upstream
# import spacy
# import matplotlib.pyplot as plt
# import seaborn as sns
# import nltk
#
#
# from textblob import TextBlob
# from collections import Counter, defaultdict
# import en_core_web_sm
# from tabulate import tabulate
# from spacytextblob.spacytextblob import SpacyTextBlob
# from spacy.lang.en.stop_words import STOP_WORDS
from preprocessing import preprocess
from visualisation import freq_barplot, unigrams, bigrams, trigrams, sentiment_hist
from barchartrace import bar_chart_race
=======
import spacy
import matplotlib.pyplot as plt

from spacytextblob.spacytextblob import SpacyTextBlob
>>>>>>> Stashed changes

pd.options.mode.chained_assignment = None  # default='warn'

path_a = 'data/t_kjv.csv'  # path to the bible csv
df = pd.read_csv(path_a)
df.head()
print("number of rows(verses): " + format(df.shape[0]))

path_b = 'data/key_english.csv'  # path to the book specification csv
df_b = pd.read_csv(path_b)
df_b.head()
print("number of rows(books): " + format(df_b.shape[0]))

# load or create the dataframe after preprocessing
if path.exists("out.csv"):
    df = pd.read_pickle("out.pkl")
else:
    df = preprocess(df, df_b)

# run the function calls below to get the visualisation

freq_barplot(df) # call function to create the frequencies barplot

unigrams(df)

bigrams(df)

trigrams(df)

bar_chart_race(df, testament='old')

sentiment_hist(df)