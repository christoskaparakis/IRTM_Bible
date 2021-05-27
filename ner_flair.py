from flair.data import Sentence
from flair.models import SequenceTagger
import numpy as np


def ner(df):
    df['ner'] = np.empty((len(df), 0)).tolist()
    # load the NER tagger
    tagger = SequenceTagger.load('ner')

    printcounter = 0
    for i in range(len(df)):

        sentence = Sentence(df['t'].iloc[i])

        # run NER over sentence
        tagger.predict(sentence)

        ent_list = []
        # iterate over entities and print
        for entity in sentence.get_spans('ner'):
            ent_list.append(entity)

        if printcounter % 1000 == 0:
            print(sentence)
            print('The following NER tags are found:')
            for entity in sentence.get_spans('ner'):
                print(entity)
            print('Progress report: ', round(printcounter / len(df) * 100), '% of verses processed.')
        printcounter += 1

        df.at[i, 'ner'] = ent_list
