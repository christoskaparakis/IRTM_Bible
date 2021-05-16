from collections import defaultdict
import seaborn as sns
import pandas as pd
from textblob import TextBlob
from matplotlib import pyplot as plt
import numpy as np


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
    plt.title('Number of words vs number of verses')
    plt.xticks(x + w / 2, data1.index, rotation=-90)
    ax1.set_xlabel('Books of Bible')
    ax1.set_ylabel('Number of verses')
    ax1.bar(x, data1.values, color=color_1, width=w, align='center')

    ax2 = ax1.twinx()

    color = color_2
    ax2.set_ylabel('Number of words')
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
        blob = TextBlob(' '.join(data.iloc[x]['tokens_wo_stop']))
        if data.loc[x]['Testament'] == 'Old':
            for bigram in blob.ngrams(2):
                OT_bigrams[' '.join(bigram)] += 1  # use lowercase words
        elif data.loc[x]['Testament'] == 'New':
            for bigram in blob.ngrams(2):
                NT_bigrams[' '.join(bigram)] += 1  # use lowercase words

    df_NT_bigrams = pd.DataFrame(sorted(NT_bigrams.items(), key=lambda x: x[1])[::-1])
    df_OT_bigrams = pd.DataFrame(sorted(OT_bigrams.items(), key=lambda x: x[1])[::-1])

    fig, axes = plt.subplots(ncols=2, figsize=(12, 10), dpi=80)
    # plt.tight_layout()

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
        blob = TextBlob(' '.join(data.iloc[x]['tokens_wo_stop']))
        if data.loc[x]['Testament'] == 'Old':
            for trigram in blob.ngrams(3):
                OT_trigrams[' '.join(trigram)] += 1  # use lowercase words
        elif data.loc[x]['Testament'] == 'New':
            for trigram in blob.ngrams(3):
                NT_trigrams[' '.join(trigram)] += 1  # use lowercase words

    df_NT_trigrams = pd.DataFrame(sorted(NT_trigrams.items(), key=lambda x: x[1])[::-1])
    df_OT_trigrams = pd.DataFrame(sorted(OT_trigrams.items(), key=lambda x: x[1])[::-1])

    fig, axes = plt.subplots(ncols=2, figsize=(12, 10), dpi=80)
    # plt.tight_layout()

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


def sentiment_hist(data):
    plt.hist(data[['sentiment_avg']][data['sentiment_avg'] != 0][data['Testament'] == 'Old'], bins=100,
             label='Old Testament')  # histogram to check the sentiment throughout the Bible
    plt.hist(data[['sentiment_avg']][data['sentiment_avg'] != 0][data['Testament'] == 'New'], bins=100,
             label='New Testament')
    plt.title('Sentiment scores for the Bible')
    plt.legend()
    plt.show()
