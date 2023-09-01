import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('kc_house_data.csv')  #real data from a house price website

############################################
# 1. Explore the data
###########################################
print(df.describe().transpose())
#Is there any missing data?
print('\n Is there any missing data? A non-zero value in the right-hand column indicates missing data.\n')
print(df.isnull().sum())

#plot histogram of price values, to identify outliers, which perhaps should be ignored
plt.figure(figsize=(10,6))
sns.histplot(df['price'])
plt.show()
#This plot shows there's very little data for houses costing more than $2000,000
# plt.figure(figsize=(5,15))
# sns.countplot(df['bedrooms'])
# plt.show()
#print('\n What correlates with price \n')
#print(df.corr()['price'].sort_values)
#
#Explore highly correlated values
plt.figure(figsize=(10,5))
sns.scatterplot(x='price', y='sqft_living', data=df)
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='bedrooms', y='price',data=df)
plt.show()


#Explore effect of location on house price
plt.figure(figsize=(12,8))
sns.scatterplot(x='price', y='long', data=df)
plt.show()

# plt.figure(figsize=(12,8))
# sns.scatterplot(x='price', y='lat', data=df)
# plt.show()

# plt.figure(figsize=(12,8))
# sns.scatterplot(x='long', y='lat', data=df, hue='price')
# plt.show()

#The top 1% of houses, the most expensive, are skewing the data, so remove them
non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]

plt.figure(figsize=(12,8))
sns.scatterplot(x='long', y='lat', data=non_top_1_perc,
               edgecolor=None, alpha=0.2, palette='RdYlGn', hue='price')
plt.show()

###################################################################
# Feature Engineering
###################################################################
