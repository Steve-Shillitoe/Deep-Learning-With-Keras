import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

######################################################
# Set up data
######################################################
df = pd.read_csv('RSCCASN.csv', parse_dates=True, index_col='DATE')
print(df)

df.columns=['Sales']

df.plot(figsize=(12,8))
plt.show()

test_size = 18 #18 months of data, more than one annual cycle
test_index = len(df) - test_size

training_data = df.iloc[:test_index]
test_data = df.iloc[test_index:]

scaler = MinMaxScaler()
scaler.fit(training_data)
scaled_training_data = scaler.transform(training_data)
scaled_test_data = scaler.transform(test_data)

length = 12 #12 months
generator = TimeseriesGenerator(scaled_training_data, scaled_training_data, 
                                length=length, batch_size=1)


