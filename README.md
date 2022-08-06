# Text Mining the Bible

## What's in this repository?

The code for the following project [Text Mining the Bible](https://github.com/ckaparakis/IRTM_Bible/blob/main/IRTM_Report_FrancescaBattipaglia_ChristosKaparakis.pdf)

## Authors

- [Christos Kaparakis](https://github.com/ckaparakis)
- [Francesca Battipaglia](https://github.com/FrancescaStud)

## Summary

The goal of this project is to analyze the text of the Bible’s books and answer some questions that arise for two topics.

- How does the text evolve through the years?
- Which are the differences between the Old and New Testament?

The above-mentioned questions led the whole development of the project. We used different techniques to try to answer these questions, and the most interesting ones turned out to be Classification, Sentiment Analysis and Named Entity Recognition.

## Results

### Classification using BERT

#### Classification of Old and New Testament

![alt text](https://github.com/ckaparakis/IRTM_Bible/blob/main/visualisation/output/classification_testaments.png)

#### Classification of Books

![alt text](https://github.com/ckaparakis/IRTM_Bible/blob/main/visualisation/output/classification_books.png)   

### Sentiment Analysis

![alt text](https://github.com/ckaparakis/IRTM_Bible/blob/main/visualisation/output/sentimentvader.png)

### Named Entity Recognition

![alt text](https://github.com/ckaparakis/IRTM_Bible/blob/main/visualisation/output/old_per_streamgraph.png)

![alt text](https://github.com/ckaparakis/IRTM_Bible/blob/main/visualisation/output/new_per_streamgraph.png)


## How do I run these scripts?

You can run this classifier code by installing the requirements

    pip install requirements.txt

and running:

- python main.py to perform Preprocessing, Sentiment Analysis, NER and produce some predefined plots.

- python clustering.py to perform SVD and plot the first two principal components.

- python classification_book.py and testament_class.py to train the BERT classifier for classifying verses to their respective books and testaments.

## Data
Source: https://www.kaggle.com/oswinrh/bible (Author: Oswin Rahadiyan Hartono)