import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

#####################################################
# Create sine wave data
#####################################################
x = np.linspace(0, 50, 501)  #Returns 501 evenly spaced numbers over range 0->50
y = np.sin(x)

plt.plot(x,y)
plt.show()  #Shows a sine wave from 0 to 50

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




    