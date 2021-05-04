import os.path
from os import path
import string
import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from collections import Counter, defaultdict
import en_core_web_sm
from tabulate import tabulate
from spacytextblob.spacytextblob import SpacyTextBlob

pd.options.mode.chained_assignment = None  # default='warn'

path_a = 'data/t_kjv.csv'  # path to the bible csv
df = pd.read_csv(path_a)
df.head()
print("number of rows(verses): " + format(df.shape[0]))

path_b = 'data/key_english.csv'  # path to the book specification csv
df_b = pd.read_csv(path_b)
df_b.head()
print("number of rows(books): " + format(df_b.shape[0]))

nlp = en_core_web_sm.load()  # load the spacy pipeline
nlp.add_pipe("spacytextblob")  # add this for sentiment analysis


def preprocess(data):
    # creating the target columns for our dataframe
    data["lower"] = np.empty((len(data), 0)).tolist()
    data["tokens"] = np.empty((len(data), 0)).tolist()
    data["lemma"] = np.empty((len(data), 0)).tolist()
    data["POS"] = np.empty((len(data), 0)).tolist()
    data['sentiment_per_word'] = np.empty((len(data), 0)).tolist()
    data['sentiment_avg'] = np.nan
    data['people'] = np.empty((len(data), 0)).tolist()
    data['locations'] = np.empty((len(data), 0)).tolist()

    # iterate through every verse (the no punctuation column)
    count1 = 0
    count2 = 0
    printcounter = 0
    for i in range(len(data)):
        # create column without punctuations
        data.at[i, 'np'] = data.loc[i]['t'].translate(str.maketrans('', '', string.punctuation))
        # create column with number of words per verse
        word_list = data.at[i, 'np'].split()
        data.at[i, 'word_count'] = len(word_list)

        doc = nlp(data.loc[i]['np'])  # create the lemma... etc without punctuations
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

        # entity extraction
        document = nlp(data.loc[i]['t'])
        for ent in document.ents:
            if ent.label_ == 'PERSON' or ent.label_ == 'ORG':
                person.append(ent.text)
                count1 += 1
            elif ent.label_ == 'LOC' or ent.label_ == 'GPE':
                locations.append(ent.text)
                count2 += 1

        data.at[i, 'lower'] = lc
        data.at[i, 'tokens'] = tok
        data.at[i, 'lemma'] = lemma
        data.at[i, 'POS'] = pos
        data.at[i, 'sentiment_per_word'] = sent
        data.at[i, 'sentiment_avg'] = doc._.polarity
        data.at[i, 'people'] = person
        data.at[i, 'locations'] = locations

        if printcounter % 1000 == 0:
            print('Progress report: ', round(printcounter / len(data) * 100), '% of verses processed.')
        printcounter += 1

    data.to_csv('out.csv', index=False)
    data.to_pickle('out.pkl')
    return data


def unigrams(data, plot=True):
    # UNIGRAMS
    NT_unigrams = defaultdict(int)  # New Testament
    OT_unigrams = defaultdict(int)  # Old Testament

    for x in range(len(data)):
        book = data.iloc[x]['b']
        testament = df_b[df_b['b'] == book].iloc[0]['t']
        if testament == 'OT':
            for word in data.loc[x]['lemma']:
                OT_unigrams[word.lower()] += 1  # use lowercase words
        elif testament == 'NT':
            for word in data.loc[x]['lemma']:
                NT_unigrams[word.lower()] += 1

    df_NT_unigrams = pd.DataFrame(sorted(NT_unigrams.items(), key=lambda x: x[1])[::-1])
    df_OT_unigrams = pd.DataFrame(sorted(OT_unigrams.items(), key=lambda x: x[1])[::-1])

    if plot:
        fig, axes = plt.subplots(ncols=2, figsize=(12, 10), dpi=80)
        plt.tight_layout()

        N = 25

        sns.barplot(y=df_NT_unigrams[0].values[:N], x=df_NT_unigrams[1].values[:N], ax=axes[0], color='red')
        sns.barplot(y=df_OT_unigrams[0].values[:N], x=df_OT_unigrams[1].values[:N], ax=axes[1], color='green')

        for i in range(2):
            axes[i].spines['right'].set_visible(False)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            axes[i].tick_params(axis='x', labelsize=10)
            axes[i].tick_params(axis='y', labelsize=10)

        axes[0].set_title(f'Top {N} most common unigrams in New Testament', fontsize=10)
        axes[1].set_title(f'Top {N} most common unigrams in Old Testament', fontsize=10)

        plt.show()


def bigrams(data):
    NT_bigrams = defaultdict(int)  # New Testament
    OT_bigrams = defaultdict(int)  # Old Testament

    for x in range(len(data)):
        book = data.iloc[x]['b']
        testament = df_b[df_b['b'] == book].iloc[0]['t']
        blob = TextBlob(' '.join(data.iloc[x]['lower']))
        if testament == 'OT':
            for bigram in blob.ngrams(2):
                OT_bigrams[' '.join(bigram)] += 1  # use lowercase words
        elif testament == 'NT':
            for bigram in blob.ngrams(2):
                NT_bigrams[' '.join(bigram)] += 1  # use lowercase words

    df_NT_bigrams = pd.DataFrame(sorted(NT_bigrams.items(), key=lambda x: x[1])[::-1])
    df_OT_bigrams = pd.DataFrame(sorted(OT_bigrams.items(), key=lambda x: x[1])[::-1])

    fig, axes = plt.subplots(ncols=2, figsize=(12, 10), dpi=80)
    plt.tight_layout()

    N = 25

    sns.barplot(y=df_NT_bigrams[0].values[:N], x=df_NT_bigrams[1].values[:N], ax=axes[0], color='red')
    sns.barplot(y=df_OT_bigrams[0].values[:N], x=df_OT_bigrams[1].values[:N], ax=axes[1], color='green')

    for i in range(2):
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].tick_params(axis='y', labelsize=10)

    axes[0].set_title(f'Top {N} most common bigrams in New Testament', fontsize=10)
    axes[1].set_title(f'Top {N} most common bigrams in Old Testament', fontsize=10)

    plt.show()


def trigrams(data):
    NT_trigrams = defaultdict(int)  # New Testament
    OT_trigrams = defaultdict(int)  # Old Testament

    for x in range(len(data)):
        book = data.iloc[x]['b']
        testament = df_b[df_b['b'] == book].iloc[0]['t']
        blob = TextBlob(' '.join(data.iloc[x]['lower']))
        if testament == 'OT':
            for trigram in blob.ngrams(3):
                OT_trigrams[' '.join(trigram)] += 1  # use lowercase words
        elif testament == 'NT':
            for trigram in blob.ngrams(3):
                NT_trigrams[' '.join(trigram)] += 1  # use lowercase words

    df_NT_trigrams = pd.DataFrame(sorted(NT_trigrams.items(), key=lambda x: x[1])[::-1])
    df_OT_trigrams = pd.DataFrame(sorted(OT_trigrams.items(), key=lambda x: x[1])[::-1])

    fig, axes = plt.subplots(ncols=2, figsize=(12, 10), dpi=80)
    plt.tight_layout()

    N = 25

    sns.barplot(y=df_NT_trigrams[0].values[:N], x=df_NT_trigrams[1].values[:N], ax=axes[0], color='red')
    sns.barplot(y=df_OT_trigrams[0].values[:N], x=df_OT_trigrams[1].values[:N], ax=axes[1], color='green')

    for i in range(2):
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].tick_params(axis='y', labelsize=10)

    axes[0].set_title(f'Top {N} most common trigrams in New Testament', fontsize=10)
    axes[1].set_title(f'Top {N} most common trigrams in Old Testament', fontsize=10)

    plt.show()


if path.exists("out.csv"):
    df = pd.read_pickle("out.pkl")
else:
    df = preprocess(df)


# plt.hist(df['sentiment_avg'], bins=50)  # histogram to check the sentiment throughout the Bible
