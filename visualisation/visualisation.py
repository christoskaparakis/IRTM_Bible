from collections import defaultdict

import matplotlib
import seaborn as sns
import pandas as pd
from textblob import TextBlob
from matplotlib import pyplot as plt
import numpy as np
from nltk.corpus import stopwords

sw = stopwords.words('english')
sw.extend(['from', 'upon', 'away', 'even', 'unto'])



def freq_barplot(df):
    color_1 = plt.cm.Blues(np.linspace(0.6, 1, 66))
    color_2 = plt.cm.Purples(np.linspace(0.6, 1, 66))

    words_verses = df.groupby('n', sort=False).agg({'b': 'count', 'word_count': 'sum'}).sort_values(by='b',
                                                                                                    ascending=False)
    true_sort = [s for s in df.n.unique() if s in words_verses.index.to_list()]
    words_verses = words_verses.loc[true_sort]

    data1 = words_verses['b']
    data2 = words_verses['word_count']

    plt.figure(figsize=(16, 8))
    x = np.arange(66)
    ax1 = plt.subplot(1, 1, 1)
    w = 0.3

    color = color_1
    plt.title('Number of words vs number of verses', fontsize=22)
    plt.xticks(x + w / 2, data1.index, rotation=-90)
    ax1.set_xlabel('Books of Bible')
    ax1.set_ylabel('Number of verses', fontsize=22)
    ax1.bar(x, data1.values, color=color_1, width=w, align='center')

    ax2 = ax1.twinx()

    color = color_2
    ax2.set_ylabel('Number of words', fontsize=22)
    ax2.bar(x + w, data2, color=color_2, width=w, align='center')

    plt.show()


def unigrams(data, plot=True):
    # UNIGRAMS
    NT_unigrams = defaultdict(int)  # New Testament
    OT_unigrams = defaultdict(int)  # Old Testament

    for x in range(len(data)):
        if data.loc[x]['Testament'] == 'Old':
            for word in data.loc[x]['tokens_wo_stop']:
                OT_unigrams[word.lower()] += 1  # use lowercase words
        elif data.loc[x]['Testament'] == 'New':
            for word in data.loc[x]['tokens_wo_stop']:
                NT_unigrams[word.lower()] += 1

    df_NT_unigrams = pd.DataFrame(sorted(NT_unigrams.items(), key=lambda x: x[1])[::-1])
    df_OT_unigrams = pd.DataFrame(sorted(OT_unigrams.items(), key=lambda x: x[1])[::-1])

    fig, axes = plt.subplots(ncols=2, figsize=(12, 8), dpi=80)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)

    N = 25

    sns.barplot(y=df_OT_unigrams[0].values[:N], x=df_OT_unigrams[1].values[:N], ax=axes[0], color='green')
    sns.barplot(y=df_NT_unigrams[0].values[:N], x=df_NT_unigrams[1].values[:N], ax=axes[1], color='red')

    for i in range(2):
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].tick_params(axis='y', labelsize=10)

    axes[0].set_title(f'Top {N} most common unigrams in Old Testament', fontsize=10)
    axes[1].set_title(f'Top {N} most common unigrams in New Testament', fontsize=10)

    plt.show()


def bigrams(data):
    NT_bigrams = defaultdict(int)  # New Testament
    OT_bigrams = defaultdict(int)  # Old Testament

    for x in range(len(data)):
        blob = TextBlob(data.iloc[x]['np']).lower() # use lowercase words
        if data.loc[x]['Testament'] == 'Old':
            for bigram in blob.ngrams(2):
                OT_bigrams[' '.join(bigram)] += 1
        elif data.loc[x]['Testament'] == 'New':
            for bigram in blob.ngrams(2):
                NT_bigrams[' '.join(bigram)] += 1

    df_NT_bigrams = pd.DataFrame(sorted(NT_bigrams.items(), key=lambda x: x[1])[::-1])
    df_OT_bigrams = pd.DataFrame(sorted(OT_bigrams.items(), key=lambda x: x[1])[::-1])

    # remove instance if both words are stop words
    drop_NT = []
    for x in range(len(df_NT_bigrams)):
        if any(elem in sw for elem in df_NT_bigrams[0][x].split()):
            drop_NT.append(x)
    df_NT_bigrams.drop(labels=drop_NT, axis=0, inplace=True)

    drop_OT = []
    for x in range(len(df_OT_bigrams)):
        if any(elem in sw for elem in df_OT_bigrams[0][x].split()):
            drop_OT.append(x)
    df_OT_bigrams.drop(labels=drop_OT, axis=0, inplace=True)

    fig, axes = plt.subplots(ncols=2, figsize=(12, 8), dpi=80)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    N = 25

    sns.barplot(y=df_NT_bigrams[0].values[:N], x=df_NT_bigrams[1].values[:N], ax=axes[1], color='red')
    sns.barplot(y=df_OT_bigrams[0].values[:N], x=df_OT_bigrams[1].values[:N], ax=axes[0], color='green')

    for i in range(2):
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].tick_params(axis='y', labelsize=10)

    axes[1].set_title(f'Top {N} most common bigrams in New Testament', fontsize=10)
    axes[0].set_title(f'Top {N} most common bigrams in Old Testament', fontsize=10)

    plt.show()


def trigrams(data):
    NT_trigrams = defaultdict(int)  # New Testament
    OT_trigrams = defaultdict(int)  # Old Testament

    for x in range(len(data)):
        blob = TextBlob(data.iloc[x]['np']).lower()
        if data.loc[x]['Testament'] == 'Old':
            for trigram in blob.ngrams(3):
                OT_trigrams[' '.join(trigram)] += 1  # use lowercase words
        elif data.loc[x]['Testament'] == 'New':
            for trigram in blob.ngrams(3):
                NT_trigrams[' '.join(trigram)] += 1  # use lowercase words

    df_NT_trigrams = pd.DataFrame(sorted(NT_trigrams.items(), key=lambda x: x[1])[::-1])
    df_OT_trigrams = pd.DataFrame(sorted(OT_trigrams.items(), key=lambda x: x[1])[::-1])

    drop_NT = []
    for x in range(len(df_NT_trigrams)):
        if any(elem in sw for elem in df_NT_trigrams[0][x].split()):
            drop_NT.append(x)
    df_NT_trigrams.drop(labels=drop_NT, axis=0, inplace=True)

    drop_OT = []
    for x in range(len(df_NT_trigrams)):
        if any(elem in sw for elem in df_OT_trigrams[0][x].split()):
            drop_OT.append(x)
    df_OT_trigrams.drop(labels=drop_OT, axis=0, inplace=True)

    fig, axes = plt.subplots(ncols=2, figsize=(13, 8), dpi=80)
    # plt.tight_layout()
    plt.subplots_adjust(left  = 0.2, wspace=0.4)

    N = 25

    sns.barplot(y=df_NT_trigrams[0].values[:N], x=df_NT_trigrams[1].values[:N], ax=axes[1], color='red')
    sns.barplot(y=df_OT_trigrams[0].values[:N], x=df_OT_trigrams[1].values[:N], ax=axes[0], color='green')

    for i in range(2):
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].tick_params(axis='y', labelsize=10)

    axes[1].set_title(f'Top {N} most common trigrams in New Testament', fontsize=10)
    axes[0].set_title(f'Top {N} most common trigrams in Old Testament', fontsize=10)

    plt.show()


def sentiment_hist(data):
    plt.hist(data[['sentiment_comp_score_vader']][data['sentiment_comp_score_vader'] != 0][data['Testament'] == 'Old'],
             bins=100,
             label='Old Testament')  # histogram to check the sentiment throughout the Bible
    plt.hist(data[['sentiment_comp_score_vader']][data['sentiment_comp_score_vader'] != 0][data['Testament'] == 'New'],
             bins=100,
             label='New Testament')
    plt.title('Sentiment scores for the Bible using VADER')
    plt.legend()
    plt.show()


def entity_flowchart(data, entity, testament):
    if entity == 'Person':
        filter = 'PER'
    elif entity == 'Location':
        filter = 'LOC'

    if testament == 'Old':
        ran = range(1, 40)
    else:
        ran = range(40, 67)

    book_ents = pd.DataFrame()
    for i in ran:
        entities = defaultdict(int)
        for verse_ents in data['ner'][data['b'] == i]:
            for ent in verse_ents:
                if ent.tag == filter:
                    entities[ent.text.lower()] += 1
        data_entities = pd.DataFrame(sorted(entities.items(), key=lambda x: x[1])[::-1]).iloc[:3]  # 3 top entities
        try:
            data_entities = data_entities.set_index(0)
            data_entities = data_entities.T
        except:
            data_entities = pd.DataFrame({book_ents.columns[0]: [0]})
        book_ents = book_ents.append(data_entities)
    book_ents = book_ents.reset_index(drop=True).T

    temp = book_ents.to_dict()
    new_d = {}
    for sub in temp.values():  # Python 3: use d.values()
        for key, value in sub.items():  # Python 3: use d.items()
            new_d.setdefault(key, []).append(value)

    import math
    for key in new_d.keys():
        for num, value in enumerate(new_d[key]):
            if math.isnan(value):
                new_d[key][num] = 0

    from matplotlib.pyplot import figure

    # plt.figure(figsize=(40, 20), dpi=100)
    #
    # plt.stackplot(ran, new_d.values(), labels=new_d.keys(), baseline='sym')
    # plt.legend(loc='lower right', prop={'size': 20}, ncol=6)
    # plt.show()

    wide = pd.DataFrame.from_dict(new_d).T
    wide.reset_index(inplace=True)
    # pd.wide_to_long(wide, ["A", "B"], i="index", j="year")
    melt = pd.melt(wide, id_vars=['index'])
    melt['variable'] = melt['variable'] + 1
    melt.to_csv('melt_' + testament + '_' + filter + '.csv')
