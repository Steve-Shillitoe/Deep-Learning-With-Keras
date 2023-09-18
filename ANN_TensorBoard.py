"""
In this module the built in data visualization capabilities of Tensorboard are explored
when a classification task is performed with TensorFlow.  
Based on the values of 30 features, does someone have a cancer tumour or not?

To run Tensorboard, open a command prompt in the root of this project and enter
    tensorboard --logdir logs\fit

Tensorboard Arguments:
    log_dir: the path of the directory where to save the log files to be
      parsed by TensorBoard.
    histogram_freq: frequency (in epochs) at which to compute activation and
      weight histograms for the layers of the model. If set to 0, histograms
      won't be computed. Validation data (or split) must be specified for
      histogram visualizations.
    write_graph: whether to visualize the graph in TensorBoard. The log file
      can become quite large when write_graph is set to True.
    write_images: whether to write model weights to visualize as image in
      TensorBoard.
    update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
      writes the losses and metrics to TensorBoard after each batch. The same
      applies for `'epoch'`. If using an integer, let's say `1000`, the
      callback will write the metrics and losses to TensorBoard every 1000
      samples. Note that writing too frequently to TensorBoard can slow down
      your training.
    profile_batch: Profile the batch to sample compute characteristics. By
      default, it will profile the second batch. Set profile_batch=0 to
      disable profiling. Must run in TensorFlow eager mode.
    embeddings_freq: frequency (in epochs) at which embedding layers will
      be visualized. If set to 0, embeddings won't be visualized.

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard

##########################################################
# Pre-process the data
##########################################################
df = pd.read_csv('cancer_classification.csv')

X = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values  #labels

# Extract training and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)

# scale the data
scaler = MinMaxScaler()
scaler.fit(X_train)  # scales to the range 0-1
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

###########################################################
# Build the model
###########################################################
#create EarlyStopping callback
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

#create Tensorboard callback
timestamp = datetime.now().strftime("%Y-%m-%d--%H%M")
log_directory = 'logs\\fit\\' + timestamp  # note \\ as we are using Windows
board = TensorBoard(log_dir=log_directory,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=1)

#create the model layers
model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

##################################################
# Train the model
#################################################
model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop, board])



