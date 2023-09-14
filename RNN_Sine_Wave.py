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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
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
model.fit_generator(generator, epochs=5)

######################################################
# Evaluate the model
#####################################################
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

first_eval_batch = scaled_train[-length:]
first_eval_batch = scaled_train.reshape((1, length, n_features))
print('predicted {}, test {}'.format(model.predict(first_eval_batch), scaled_test[0]))

test_predictions = []


    