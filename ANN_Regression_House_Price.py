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




