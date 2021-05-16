import bar_chart_race as bcr
from collections import Counter
# need dataset with words and their frequencies per book
import numpy as np
import pandas as pd


def bar_chart_race(df, testament='both'):
    if testament == 'old':
        df = df.loc[:38]
    elif testament == 'new':
        df = df.loc[39:]

    aggregated = df.groupby('b').agg({'tokens_wo_stop': 'sum'})

    printcounter = 0
    word_freq = pd.DataFrame()
    for i in range(len(aggregated)):
        words = [y.lower() for y in aggregated.iloc[i, 0]]
        # Counter(words).keys() # equals to list(set(words))
        # Counter(words).values() # counts the elements' frequency
        x = Counter(words)
        x = x.most_common()

        for key in x[0:9]:
            if key[0] not in word_freq.columns:
                word_freq[key[0]] = np.nan
                word_freq.at[i, key[0]] = key[1]
            else:
                word_freq.at[i, key[0]] = key[1]

        printcounter += 1
        print('Progress report: ', round(printcounter / len(aggregated) * 100), '% of books processed.')

    cum_words = pd.DataFrame()
    for i in word_freq.columns:
        cum_words[i] = np.nancumsum(word_freq[i])

    bcr.bar_chart_race(
        df=cum_words,
        filename='bible_word_race.mp4',
        orientation='h',
        sort='desc',
        n_bars=6,
        fixed_order=False,
        fixed_max=True,
        steps_per_period=10,
        interpolate_period=False,
        label_bars=True,
        bar_size=.95,
        perpendicular_bar_func='median',
        period_length=500,
        figsize=(5, 3),
        dpi=144,
        cmap='dark12',
        title='The Bibles word frequencies',
        title_size='',
        bar_label_size=7,
        tick_label_size=7,
        shared_fontdict={'family': 'Helvetica', 'color': '.1'},
        scale='linear',
        writer=None,
        fig=None,
        bar_kwargs={'alpha': .7},
        filter_column_colors=False)
