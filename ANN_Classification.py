"""
In this module a classification task is performed with TensorFlow.

This inclueds identifying and dealing with overfitting through Early Stopping Callbacks 
and Dropout Layers.

Early Stopping - Keras can stop training based on a loss condition on the validation data
passed during the model.fit() call.

Dropout Layers - Dropout layer can be added to layers to 'turn off' neurons to prevent
overfitting.  Each Dropout Layer will turn off a user-defined percentage of neurons in the
previous layer with each batch of data. 

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
            mean_absolute_percentage_error, explained_variance_score)
from tensorflow.keras.models import Sequential

df = pd.read_csv('cancer_classification.csv')


