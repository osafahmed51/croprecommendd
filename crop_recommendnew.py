#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Crop_recommendation.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.size


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df['label'].unique()


# In[9]:


df.dtypes


# In[10]:


df['label'].value_counts()


# In[11]:


sns.heatmap(df.corr(),annot=True)


# In[12]:


features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
labels = df['label']


# In[13]:


# Initialzing empty lists to append all model's name and corresponding name
acc = []
model = []


# In[14]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


# In[15]:


from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)

print(classification_report(Ytest,predicted_values))


# In[16]:


from sklearn.model_selection import cross_val_score


# In[17]:


# Cross validation score (Decision Tree)
score = cross_val_score(DecisionTree, features, target,cv=5)


# In[18]:


score


# In[19]:


data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = DecisionTree.predict(data)
print(prediction)


# In[20]:


import pickle


# In[24]:


pickle.dump(DecisionTree,open('model9.pkl','wb'))


# In[ ]:




