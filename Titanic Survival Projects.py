#!/usr/bin/env python
# coding: utf-8

# Author :- Ashutosh Kumar
# Batch :- April
# Domain :- Data Science
# Aim :- To Build a model that predicts whether a passenger on the Titanic survived or not.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[58]:


df = pd.read_csv('Titanic-Dataset.csv')
df.head(25)


# In[12]:


df.shape


# In[13]:


df.describe()


# In[16]:


df['Survived'].value_counts()


# In[17]:


#lets visualize the count of survivals wrt pclass
sns.countplot(x=df['Survived'], hue=df['Pclass'])


# In[18]:


df["Sex"]


# In[20]:


#let visualize the count of survival wrt Gender
sns.countplot(x=df['Sex'], hue=df['Survived'])


# In[21]:


#look at survival rate by sex 
df.groupby('Sex')[['Survived']].mean()


# In[26]:


df['Sex'].unique()
array['male', 'female'], dtype=object)


# In[25]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['Sex']= labelencoder.fit_transform(df['Sex'])
df.head()


# In[27]:


df['Sex'], df['Survived']


# In[28]:


sns.countplot(x=df['Sex'], hue=df["Survived"])


# In[29]:


df.isna().sum()


# In[30]:


#After dropping non required column
df=df.drop(['Age'], axis=1)


# In[31]:


df_final = df
df_final.head(10)


# In[45]:


X = df[['Pclass','Sex']]
Y = df['Survived']


# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[49]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state = 0)
log.fit(X_train, Y_train)


# In[51]:


pred = print(log.predict(X_test))


# In[52]:


print(Y_test)


# In[56]:


import warnings
warnings.filterwarnings("ignore")

res = log.predict([[2,1]])

if(res==0):
    print("So Sorry! Not Survived")
else:
    print("Survived")


# In[ ]:




