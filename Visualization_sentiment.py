#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


df = pd.read_csv('/Users/francescabattipaglia/Desktop/IRTM_Bible/out.csv')


# In[7]:


df.word_count


# In[19]:


df_old =  df[df.Testament =='Old']


# In[20]:


df_new =  df[df.Testament == 'New']


# In[74]:


plt.figure(figsize = (16,9))
g = sns.lineplot(x = 'b' , y = 'word_count', hue ='sent_categorical', 
                 data = df_old,
                 markers = [".", "s", ">"],legend = 'brief')
g = (g.set_xticks(np.arange(1,41)))
plt.title("Sentiment over books", fontsize = 20)
plt.xlabel("Book", fontsize = 15)
plt.ylabel("Number of words", fontsize = 15)
plt.show(g)


# In[69]:


plt.figure(figsize = (16,9))
g = sns.lineplot(x = 'b' , y = 'word_count', hue ='sent_categorical', 
                 data = df_new,
                 markers = [".", "s", ">"],legend = 'brief')
g = (g.set_xticks(np.arange(40,67)))
plt.title("Sentiment over books", fontsize = 20)
plt.xlabel("Book", fontsize = 15)
plt.ylabel("Number of words", fontsize = 15)
plt.show(g)


# In[ ]:




