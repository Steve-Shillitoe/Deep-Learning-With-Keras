import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

iris = pd.read_csv('iris.csv')
print(iris.head())

#remove label column
X = iris.drop('species', axis=1)
print(X)

y = iris['species']
print(y.unique())

#one-hot encode species labels
encoder = LabelBinarizer()
y = encoder.fit_transform(y)
print('One hot encoded y =', y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scaler = MinMaxScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(units=4, activation='relu', input_shape=[4,]))
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(patience=10)

model.fit(x=scaled_X_train, y=y_train, validation_data=(scaled_X_test, y_test), 
          epochs=300, callbacks=[early_stop])

metrics = pd.DataFrame(model.history.history)
print(metrics)

metrics[['loss', 'val_loss']].plot()
plt.show()
metrics[['accuracy', 'val_accuracy']].plot()
plt.show()

print(model.evaluate(scaled_X_test, y_test, verbose=0))









