from pickletools import optimize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('kc_house_data.csv')  #real data from a house price website

############################################
# 1. Explore the data
###########################################
print(df.describe().transpose())
#Is there any missing data?
print('\n Is there any missing data? A non-zero value in the right-hand column indicates missing data.\n')
print(df.isnull().sum())

#plot histogram of price values, to identify outliers, which perhaps should be ignored
#plt.figure(figsize=(10,6))
#sns.histplot(df['price'])
#plt.show()
#This plot shows there's very little data for houses costing more than $2000,000
# plt.figure(figsize=(5,15))
# sns.countplot(df['bedrooms'])
# plt.show()
#print('\n What correlates with price \n')
#print(df.corr()['price'].sort_values)
#
#Explore highly correlated values
# plt.figure(figsize=(10,5))
# sns.scatterplot(x='price', y='sqft_living', data=df)
# plt.show()

# plt.figure(figsize=(10,6))
# sns.boxplot(x='bedrooms', y='price',data=df)
# plt.show()


#Explore effect of location on house price
#plt.figure(figsize=(12,8))
#sns.scatterplot(x='price', y='long', data=df)
#plt.show()

# plt.figure(figsize=(12,8))
# sns.scatterplot(x='price', y='lat', data=df)
# plt.show()

# plt.figure(figsize=(12,8))
# sns.scatterplot(x='long', y='lat', data=df, hue='price')
# plt.show()

#The top 1% of houses, the most expensive, are skewing the data, so remove them
non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]

# plt.figure(figsize=(12,8))
# sns.scatterplot(x='long', y='lat', data=non_top_1_perc,
#                edgecolor=None, alpha=0.2, palette='RdYlGn', hue='price')
# plt.show()

###################################################################
# Feature Engineering
###################################################################
# A useful mindset is 'should I make this continuous function categorical
print(df.head())
#Drop the id column as it has no influence on house price
print('\nDrop the id column as it has no influence on house price\n')
df = df.drop('id', axis=1)
print(df.head)

#convert date string to datetime object
df['date'] = pd.to_datetime(df['date'])
#create a year & month column
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)
print(df.head())

#df.groupby('month').mean()['price'].plot()
#plt.show()

#drop date & zipcode columns
df = df.drop('date', axis=1)
df = df.drop('zipcode', axis=1)
print(df.columns)

#################################################################
# Data Processing
################################################################
#Separate features from the label
X = df.drop('price', axis=1).values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
scaler = MinMaxScaler(feature_range=(0,1) )
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  #only transform test data

###################################################################
# Build the model
###################################################################
model = Sequential()

model.add(Dense(units=19, activation='relu'))
model.add(Dense(units=19, activation='relu'))
model.add(Dense(units=19, activation='relu'))
model.add(Dense(units=19, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=128, epochs=400)






