"""
In this module a classification task is performed with TensorFlow.  
Based on the values of 30 features, does someone has a cancer tumour or not?

This inclueds identifying and dealing with overfitting through Early Stopping Callbacks 
and Dropout Layers.

Early Stopping - Keras can stop training based on a loss condition on the validation data
passed during the model.fit() call.

Dropout Layers - Dropout layer can be added to layers to 'turn off' neurons to prevent
overfitting.  Each Dropout Layer will turn off a user-defined percentage of neurons in the
previous layer with each batch of data. 

In TensorFlow, the model.compile function is used to configure the training process for a neural network model. 
It takes several arguments, including the loss and optimizer properties, 
which are essential for training a machine learning model. 
Here's a brief explanation of each:

Loss Function (loss):
The loss function (also known as the objective or cost function) quantifies how well your model's predictions match the actual target values during training.
It calculates a single scalar value that represents the error or discrepancy between the predicted output and the true target.
The goal during training is to minimize this loss, as a lower loss indicates better model performance.
The choice of the loss function depends on the type of machine learning task. For example, Mean Squared Error (MSE) is commonly used for regression tasks, while Categorical Crossentropy is used for classification tasks.

Optimizer (optimizer):
The optimizer is responsible for updating the model's weights and biases during training to minimize the loss function.
It implements an optimization algorithm, such as Stochastic Gradient Descent (SGD), Adam, RMSprop, etc., to adjust the model's parameters iteratively.
Different optimizers have their own update rules and hyperparameters that influence the training process, convergence speed, and generalization performance of the model.
The choice of optimizer and its hyperparameters can significantly affect the training process, and it's often a matter of experimentation to find the best combination for a specific problem.
Here's a basic example of how you would use these properties in the model.compile function:


"""

#from pickletools import optimize
from calendar import EPOCH
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
            mean_absolute_percentage_error, explained_variance_score, classification_report, confusion_matrix)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('cancer_classification.csv')

########################################################
# Data Exploration
########################################################
print(df.info())
print('\n There are no null values.')
print(df.describe().transpose())

#Is the label class well balanced?
sns.countplot(x='benign_0__mal_1', data=df)
#plt.show()

#Let's see if the label class is highly correlated with its features
print('\n Lets see if the label class is highly correlated with its features')
print(df.corr()['benign_0__mal_1'].sort_values())
#Lets look at a plot of this data. Exclude the last classification column
plt.figure(figsize=(12,6))
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')
#plt.show()

#############################################################
# Data Preparation
#############################################################
#Get numpy arrays of data
X = df.drop('benign_0__mal_1', axis=1).values  #Drop label column from training data
y = df['benign_0__mal_1'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

###############################################################
# Build the model
###############################################################
#X_train.shape = (436, 30)  436 rows of 30 features
# model = Sequential()

# model.add(Dense(units=30, activation='relu'))

# model.add(Dense(units=15, activation='relu'))
# #Output layer
# model.add(Dense(units=1, activation='sigmoid'))  #Binary Classification, sigmoid outputs 0 to 1

# model.compile(loss='binary_crossentropy', optimizer='adam')

# #train the model
# model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test))

# losses = pd.DataFrame(model.history.history)
# losses.plot()
# plt.show()
#Above plot shows overfitting due to large number of epochs

#New model, configured to avoid overfitting
model = Sequential()

model.add(Dense(units=30, activation='relu'))
model.add(Dropout(rate=0.5)) #To prevent overfitting
model.add(Dense(units=15, activation='relu'))
model.add(Dropout(rate=0.5)) #To prevent overfitting
#Output layer
model.add(Dense(units=1, activation='sigmoid'))  #Binary Classification, sigmoid outputs 0 to 1

model.compile(loss='binary_crossentropy', optimizer='adam')

##o prevent overfitting
#we wish to minimise validation loss, loss = 0 is a perfect fit
#patience =25, we will wait 25 epochs after a stopping point is detected
early_stop = EarlyStopping(monitor='val_loss', mode='min',
                           verbose=1, patience=25)

#train the model
model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test),
                callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)
losses.plot()
#plt.show()
#In the plot, we that training loss & validation loss are decreasing together and flattening out together

threshold = 0.5  # Adjust this threshold as needed
predicted_labels = (model.predict(X_test) > threshold).astype(int)

# Get the predicted class labels
#predicted_classes = tf.argmax(predictions, axis=-1).numpy()
#print(predicted_classes)

print(classification_report(y_test, predicted_labels))
print('**********************************************************************')
print(confusion_matrix(y_test, predicted_labels))
#THe confusion matrix is [[TP  FP]
#                           FN TN]]
# So we have 1 FP & 5 FN

