#!/usr/bin/env python
# coding: utf-8

# In[ ]:


NAME :------- SWATHI K
The Sparks Foundation - Data Science & Business Analytics Internship
Task 06 : Prediction using Decision Tree Algorithm
The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.


# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets as datasets #for import dataset from sklearn library
import warnings
warnings.filterwarnings(action='ignore') #for ignore warnings


# In[7]:


#importing our iris datset
iris=datasets.load_iris() #loading the iris dataset

df= pd.DataFrame(iris.data, columns=iris.feature_names) # forming the iris dataset

df.head() # for display first 5 rows


# In[9]:


df.tail() #for display last 5 rows


# In[10]:


new = iris.target
new


# In[11]:


df['species']=iris['target']
df['species']=df['species'].apply(lambda x: iris['target_names'][x])
df.head()


# In[12]:


df.shape #shape of the dataset


# In[13]:


df.info() #over-all info about the dataset


# In[14]:


df.describe() #statistical overview of data and described all the featuresb

# petalwidth minimum value is 0.1 and maximum is 2.5 similarly you can check other


# In[15]:


# preprocessing of the dataset so turn to check null values

df.isnull().sum() # for check the sum of all num values in each columns

#data is clean


# In[16]:


#Pairplot
sns.pairplot(df) #It's a pairplot to gain more insights of the datab


# In[17]:


# now turn to compare with various features and relationship between columns

sns.pairplot(df, hue='species')


# In[18]:


plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), annot=True) 


# In[19]:


#preparation the data for training

# Ml is can not work with string values so we need to drop some columns. so we need to label encoding with 'y'

#x-y split 
x= df.iloc[:,:-1].values #feature matrix
y=df.iloc[:,-1].values #vector of predictions


# In[20]:


from sklearn.preprocessing import LabelEncoder # import library for label encoding


# In[21]:


y


# In[22]:


lab = LabelEncoder()
y= lab.fit_transform(y)
y


# In[23]:


#spliting into training and test set

from sklearn.model_selection import train_test_split as tts

x_train, x_test, y_train, y_test = tts(x, y, train_size=0.8, random_state=1)

x_train.shape, x_test.shape, y_train.shape, y_test.shape # shape of spliting set


# In[24]:


#training the model

from sklearn.tree import DecisionTreeClassifier as dt

classifier= dt(class_weight='balanced')
classifier.fit(x_train, y_train)

y_pred= classifier.predict(x_test)


# In[25]:


y_pred #predicted value


# In[26]:


y_test  #actual value


# In[27]:


#compare with actual and predicted value

data = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
data.head()


# In[28]:


#Confusion Matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm= confusion_matrix(y_test,y_pred)
cm


# In[29]:


#accuracy score

accuracy_score(y_test,y_pred)


# In[30]:


# Install required libraries
# !pip install pydotplus
#!pip install graphviz


# In[31]:


fn=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']


# In[32]:


# Visualize the graph or decision tree
from sklearn.tree import plot_tree

plt.figure(figsize=(20,15))
plot_tree(classifier, feature_names=fn, class_names=cn, filled=True, rounded=True)
plt.show()


# In[33]:


print('Thank you, We have completed task. Done by Swathi K')


# In[ ]:




