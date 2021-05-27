import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data\out.csv')
df.word_count
#
df_old = df[df.Testament == 'Old']

df_new = df[df.Testament == 'New']

plt.figure(figsize=(16, 9))
g = sns.lineplot(x='b', y='word_count', hue='sent_categorical',
                 data=df_old,
                 markers=[".", "s", ">"], legend='brief')
g = (g.set_xticks(np.arange(1, 41)))
plt.title("Sentiment over books", fontsize=20)
plt.xlabel("Book", fontsize=15)
plt.ylabel("Number of words", fontsize=15)
plt.show(g)

plt.figure(figsize=(16, 9))
g = sns.lineplot(x='b', y='word_count', hue='sent_categorical',
                 data=df_new,
                 markers=[".", "s", ">"], legend='brief')
g = (g.set_xticks(np.arange(40, 67)))
plt.title("Sentiment over books", fontsize=20)
plt.xlabel("Book", fontsize=15)
plt.ylabel("Number of words", fontsize=15)
plt.show(g)

# In[ ]:
