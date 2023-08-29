"""
Using Keras to solve a regression problem
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('fake_reg.csv')

sns.pairplot(df)
plt.show()

X = df[['feature1', 'feature2']].values
y = df['price'].values #labels or categories

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Normalise and scale feature data 
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



