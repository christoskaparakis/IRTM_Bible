import string

import numpy as np
import en_core_web_sm
from nltk.corpus import stopwords

# run python -m nltk.downloader popular on command
sw = stopwords.words('english')
sw.extend(['from', 'upon', 'away', 'even', 'unto'])


def preprocess(data, data_b):
    # create Testament column
    data.loc[data['b'] <= 39, 'Testament'] = 'Old'
    data.loc[data['b'] > 39, 'Testament'] = 'New'

    # create book name column
    data = data.merge(data_b[['b', 'n']], on='b', how='left')

    nlp = en_core_web_sm.load()  # load the spacy pipeline
    nlp.add_pipe("spacytextblob")  # add this for sentiment analysis

    # creating the target columns for our dataframe
    data["lower"] = np.empty((len(data), 0)).tolist()
    data["tokens"] = np.empty((len(data), 0)).tolist()
    data["tokens_wo_stop"] = np.empty((len(data), 0)).tolist()
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
        tok_wo_stop = []
        lemma = []
        pos = []
        sent = []
        person = []
        locations = []

        for token in doc:
            # print(token.text, token.pos_, token.dep_)
            lc.append(token.lower_)
            tok.append(token.text)

            lexeme = nlp.vocab[token.lower_]
            if not lexeme in sw:
                tok_wo_stop.append(token.text)

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
        data.at[i, 'tokens_wo_stop'] = tok_wo_stop
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
