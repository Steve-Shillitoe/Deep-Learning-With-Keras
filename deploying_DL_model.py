import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


#####################################################
# Prepare the data
####################################################
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

##################################################
# Build the model
#################################################
# model = Sequential()
# model.add(Dense(units=4, activation='relu', input_shape=[4,]))
# model.add(Dense(units=3, activation='softmax'))

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# early_stop = EarlyStopping(patience=10)

# model.fit(x=scaled_X_train, y=y_train, validation_data=(scaled_X_test, y_test), 
#           epochs=300, callbacks=[early_stop])

####################################################################
# Evaluate the model
###################################################################
# metrics = pd.DataFrame(model.history.history)
# print(metrics)

# metrics[['loss', 'val_loss']].plot()
# plt.show()
# metrics[['accuracy', 'val_accuracy']].plot()
# plt.show()

# print(model.evaluate(scaled_X_test, y_test, verbose=0))

#######################################################
# Prepare for deployment
######################################################
epochs = 300
scaled_X = scaler.fit_transform(X)

#Build the model
model = Sequential()
model.add(Dense(units=4, activation='relu', input_shape=[4,]))
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#early_stop = EarlyStopping(patience=10)

model.fit(x=scaled_X, y=y, epochs=epochs)
model.save('final_iris_model.h5')
joblib.dump(scaler, 'iris_scaler.pkl')

flower_model = load_model('final_iris_model.h5')
flower_scaler = joblib.load('iris_scaler.pkl')


def return_prediction(model, scaler, sample_json):
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']
    
    flower = [[s_len, s_wid, p_len, p_wid]]
    flower = scaler.transform(flower)
    
    prediction = model.predict(flower)  
    classes_index = np.argmax(prediction, axis=-1)

    classes = np.array(['setosa', 'versicolor', 'virginica'])
    return classes[classes_index]
    








