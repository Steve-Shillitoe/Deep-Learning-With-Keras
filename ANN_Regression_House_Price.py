"""
In this module an ANN is used to predict the price of a house based on its features.

The model is trained using existing house features and price data.

An data analysis is performed in order to identify data points that should be
excluded from the training.
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

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

#Make predictions with X_test
predictions = model.predict(X_test)
print("Predictions = {}".format(predictions))

#Compare predictions to what we are the correct values
print("\n mean_absolute_percentage_error  = ", mean_absolute_percentage_error(y_test, predictions)*100)
print("\n explained_variance_score = ", explained_variance_score(y_test, predictions))

#pred_df = pd.DataFrame(y_test, columns=['Actual House Price'])
#pred_df = pd.concat([pred_df, predictions], axis=1) # axis=1, join along columns
#pred_df.columns = ['Actual House Price', 'Model Predictions']
plt.figure(figsize=(12,6))
sns.scatterplot(data=predictions)
plt.plot(y_test, y_test, 'r')
plt.show()
print('The plot shows that the model is thrown out by the expensive house outliers. So we should ignore them & only train with the bottom 99% of houses')

#Let's do a test prediction with the first house in the data frame
single_house = df.drop('price', axis=1).iloc[0]
#Convert this data to a numpy array and rescale it
single_house = scaler.transform(single_house.values.reshape(-1,19))
print('scaled single house data = {}'.format(single_house))
print('single house  = predicted price = {}'.format(model.predict(single_house)))
