import string
import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from spacytextblob.spacytextblob import SpacyTextBlob

pd.options.mode.chained_assignment = None  # default='warn'

path = 'data/t_kjv.csv'  # path to the bible csv
df = pd.read_csv(path)
df.head()
print("number of rows(verses): " + format(df.shape[0]))

path_b = 'data/key_english.csv'  # path to the book specification csv
df_b = pd.read_csv(path_b)
df_b.head()
print("number of rows(books): " + format(df_b.shape[0]))

nlp = spacy.load("en_core_web_sm")  # load the spacy pipeline
nlp.add_pipe("spacytextblob")  # add this for sentiment analysis

# create column without punctuations
for i in range(len(df)):
    df.at[i, 'np'] = df.loc[i]['t'].translate(str.maketrans('', '', string.punctuation))

# creating the target columns for our dataframe
df["lower"] = ""
df["tokens"] = ""
df["lemma"] = ""
df["POS"] = ""
df['sentiment_per_word'] = ""
df['sentiment_avg'] = np.nan
df['people'] = ""
df['locations'] = ""

# iterate through every verse (the no punctuation column)
count1 = 0
count2 = 0
for i in range(len(df)):
    doc = nlp(df.loc[i]['t'])
    lc = []
    tok = []
    lemma = []
    pos = []
    sent = []
    person = []
    locations = []

    for token in doc:
        # print(token.text, token.pos_, token.dep_)
        lc.append(token.lower_)
        tok.append(token.text)
        lemma.append(token.lemma_)
        pos.append(token.pos_)
        sent.append(token._.polarity)

    document = nlp(df.loc[i]['t'])
    for ent in document.ents:
        if ent.label_ == 'PERSON' or ent.label_ == 'ORG':
            person.append(ent.text)
            count1 += 1
        elif ent.label_ == 'LOC' or ent.label_ == 'GPE':
            locations.append(ent.text)
            count2 += 1

    df.at[i, 'lower'] = lc
    df.at[i, 'tokens'] = tok
    df.at[i, 'lemma'] = lemma
    df.at[i, 'POS'] = pos
    df.at[i, 'sentiment_per_word'] = sent
    df.at[i, 'sentiment_avg'] = doc._.polarity
    df.at[i, 'people'] = person
    df.at[i, 'locations'] = locations

plt.hist(df['sentiment_avg'], bins=50)  # histogram to check the sentiment throughout the Bible

df.to_csv('out.csv', index=False)
