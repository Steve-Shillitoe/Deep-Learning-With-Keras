from pickletools import optimize
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

#df.plot(figsize=(12,8))
#plt.show()

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

#losses = pd.DataFrame(model.history.history)
#losses.plot(figsize=(12,8))
#plt.show()

test_predictions = []

first_eval_batch = scaled_training_data[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test_data)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    
    #To predict one step into the future, move current batch one step forward
    #current_batch[:,1:,:] gets rid of first item
    #and replaces it with [[current_pred]]
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)
    
true_predictions = scaler.inverse_transform(test_predictions)

#Now compare predictions with test data
#Add Predictions to pandas test DataFrame
test_data['Predictions'] = true_predictions
print('test data frame = {}'.format(test_data))
#plot data
#test_data.plot(figsize=(12,8))
#plt.show()

##################################################
## Forecasting into the future
##################################################
full_scaler = MinMaxScaler()
scaled_full_data = scaler.fit_transform(df)

length = 12
generator = TimeseriesGenerator(scaled_full_data, scaled_full_data,
                                length=length, batch_size=1)# scaled_full_data twice as source of both X & y

model = Sequential()
model.add(LSTM(units=100, activation='relu', input_shape=(length, n_features)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')
model.fit(generator, epochs=8)

forecast = []
periods = 12  #forecast period into the future
first_eval_batch = scaled_training_data[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(periods):
    current_pred = model.predict(current_batch)[0]
    forecast.append(current_pred)
    
    #To predict one step into the future, move current batch one step forward
    #current_batch[:,1:,:] gets rid of first item
    #and replaces it with [[current_pred]]
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)
    
forecast = scaler.inverse_transform(forecast)
forecast_index = pd.date_range(start='2019-11-01', periods=periods, freq='MS' ) #freq='MS' == month start
forecast_df = pd.DataFrame(data=forecast, index=forecast_index, columns=['Forecast'])
print(forecast_df)

ax=df.plot()
forecast_df.plot(ax=ax)
plt.xlim('2018-01-01', '2020-12-01')
plt.show()

