from os import path
import pandas as pd
from preprocessing import preprocess
from visualisation.visualisation import freq_barplot, unigrams, bigrams, trigrams, sentiment_hist
from ner_flair import ner

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
    df = pd.read_pickle("data\out.pkl")
else:
    df = preprocess(df, df_b)
    df = ner(df)

# run the function calls below to get the visualisation

freq_barplot(df) # call function to create the frequencies barplot

unigrams(df)

bigrams(df)

trigrams(df)

sentiment_hist(df)