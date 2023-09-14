"""
In this module a simple Recurrent Neural Network is created that predicts
values in a sine curve time sequence.

TimeseriesGenerator is used to generate time series data samples.

TimeseriesGenerator is a utility class provided by TensorFlow's Keras API (tf.keras)
for generating time series data samples from a given sequence of data. 
It's particularly useful when working with time series forecasting 
and sequence prediction tasks. 
The primary function of TimeseriesGenerator is to create batches of 
input-output pairs from a given sequence of data, allowing you to 
feed this data into a machine learning model for training or evaluation.

TimeseriesGenerator is particularly useful when you have a large 
time series dataset and want to efficiently generate batches of data 
for training recurrent neural networks (RNNs), convolutional neural networks (CNNs),
or any other type of sequence-based model. 
It simplifies the process of handling time series data and ensures 
that your model receives input data in the appropriate format.

"""
from gc import callbacks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

#####################################################
# Create sine wave data
#####################################################
x = np.linspace(0, 50, 501)  #Returns 501 evenly spaced numbers over range 0->50
y = np.sin(x)

#plt.plot(x,y)
#plt.show()  #Shows a sine wave from 0 to 50

df = pd.DataFrame(data=y, index=x, columns=['Sine'])  #column in sine of index value, x

test_percent = 0.1 #10% of values at the end of the data is the test set
test_point = np.round(len(df) * test_percent)
test_index = int(len(df) - test_point) #test data are beyond this index

#iloc = index location slicing
train = df.iloc[:test_index] #slice data from start of the array upto test_index 
test = df.iloc[test_index:]  #slice data from test_index to end of array

# scale the data. Need to scale the y, label data as it is  
# fed back into the neurons.
scaler = MinMaxScaler()
scaler.fit(train)  #fit to training data
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test) #Only transform test data, do not fit

########################################################
# TimeseriesGenerator
########################################################
length = 50  #Feed model 50 data items which will cover one cycle of the time series
batch_size = 1
generator = TimeseriesGenerator(scaled_train, scaled_train,
                                length=length, batch_size=batch_size)

########################################################
# Create the model
########################################################
n_features = 1
model = Sequential()
#input layer
model.add(SimpleRNN(units=50, input_shape=(length, n_features)))  #units=50 as batch size is 50
#output layer
model.add(Dense(units=1))  #only 1 prediction
model.compile(optimizer='adam', loss='mse') #loss='mse' as values are continuous
print(model.summary())

#######################################################
# Train the model
#######################################################
model.fit(generator, epochs=5)

######################################################
# Evaluate the model
#####################################################
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

#print('predicted {}, test {}'.format(model.predict(first_eval_batch), scaled_test[0]))

test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    
    #To predict one step into the future, move current batch one step forward
    #current_batch[:,1:,:] gets rid of first item
    #and replaces it with [[current_pred]]
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)
    
true_predictions = scaler.inverse_transform(test_predictions)

#Now compare predictions with test data
#Add Predictions to pandas test DataFrame
test['Predictions'] = true_predictions
print('test data frame = {}'.format(test))
# #plot data
# test.plot(figsize=(12,8))
# plt.show()
    

###############################################################
### Add early stopping & use a LSTM for performance comparison
###############################################################
early_stop = EarlyStopping(monitor='val_loss', patience=2)

#take 49 data points and predict the 50th point.
#scaled_test has 50 data points, so length must be 50 - 1 = 49
length = 49
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)
validation_generator = TimeseriesGenerator(scaled_test, scaled_test, length=length, batch_size=1)

model = Sequential()
#input layer
model.add(LSTM(units=50, input_shape=(length, n_features)))  #units=50 as batch size is 50
#output layer
model.add(Dense(units=1))  #only 1 prediction
model.compile(optimizer='adam', loss='mse') #loss='mse' as values are continuous
print(model.summary())
model.fit(generator, epochs=20, 
          validation_data=validation_generator,
          callbacks=[early_stop])

test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    
    #To predict one step into the future, move current batch one step forward
    #current_batch[:,1:,:] gets rid of first item
    #and replaces it with [[current_pred]]
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)
    
true_predictions = scaler.inverse_transform(test_predictions)

#Now compare predictions with test data
#Add Predictions to pandas test DataFrame
test['LSTM_Predictions'] = true_predictions
print('test data frame = {}'.format(test))
#plot data
test.plot(figsize=(12,8))
plt.show()

##############################################
### Forecast into the future
#############################################
full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)
generator = TimeseriesGenerator(scaled_full_data, scaled_full_data, length=length, batch_size=1 )

model = Sequential()
#input layer
model.add(LSTM(units=50, input_shape=(length, n_features)))  #units=50 as batch size is 50
#output layer
model.add(Dense(units=1))  #only 1 prediction
model.compile(optimizer='adam', loss='mse') #loss='mse' as values are continuous
print(model.summary())
model.fit(generator, epochs=6)

forecast = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(25):  #choice of 25 is arbitrary
    current_pred = model.predict(current_batch)[0]
    forecast.append(current_pred)
    
    #To predict one step into the future, move current batch one step forward
    #current_batch[:,1:,:] gets rid of first item
    #and replaces it with [[current_pred]]
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)
    
forecast = scaler.inverse_transform(forecast)

forecast_index = np.arange(50.1, 52.6, step=0.1)

plt.plot(df.index, df['Sine'])
plt.plot(forecast_index, forecast)
plt.show()


    