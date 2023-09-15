import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
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

#################################################
## Build the model
#################################################
n_features = 1
model = Sequential()
model.add(LSTM(units=100, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=2)
validation_generator = TimeseriesGenerator(scaled_test_data, scaled_test_data, 
                                           length=length, batch_size=1)

model.fit(generator, epochs=20, 
          validation_data=validation_generator,
          callbacks=[early_stop])


