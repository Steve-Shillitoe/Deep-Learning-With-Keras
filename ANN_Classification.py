"""
In this module a classification task is performed with TensorFlow.

This inclueds identifying and dealing with overfitting through Early Stopping Callbacks 
and Dropout Layers.

Early Stopping - Keras can stop training based on a loss condition on the validation data
passed during the model.fit() call.

Dropout Layers - Dropout layer can be added to layers to 'turn off' neurons to prevent
overfitting.  Each Dropout Layer will turn off a user-defined percentage of neurons in the
previous layer with each batch of data. 

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
            mean_absolute_percentage_error, explained_variance_score)
from tensorflow.keras.models import Sequential

df = pd.read_csv('cancer_classification.csv')

########################################################
# Data Exploration
########################################################
print(df.info())
print('\n There are no null values.')
print(df.describe().transpose())

#Is the label class well balanced?
sns.countplot(x='benign_0__mal_1', data=df)
plt.show()

#Let's see if the label class is highly correlated with its features
print('\n Lets see if the label class is highly correlated with its features')
print(df.corr()['benign_0__mal_1'].sort_values())
#Lets look at a plot of this data. Exclude the last classification column
plt.figure(figsize=(12,6))
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')
plt.show()

#############################################################
# Data Preparation
#############################################################
#Get numpy arrays of data
X = df.drop('benign_0__mal_1', axes=1).values  #Drop label column from training data
y = df['benign_0__mal_1'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)





