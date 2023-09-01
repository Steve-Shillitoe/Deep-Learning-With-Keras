"""
Using Keras to solve a regression problem. The prediction of the price of an item for sale
based on 2 of its features

Choosing an optimizer and loss
   Keep in mind what kind of problem you are trying to solve:
   
   For a multi-class classification problem,
        model.compile(optimizer='rmsprop',,
                      loss='categorical_crossentropy',,
                      metrics=['accuracy']),
   
  For a binary classification problem,
        model.compile(optimizer='rmsprop',,
                      loss='binary_crossentropy',,
                      metrics=['accuracy']),
    
    For a mean squared error regression problem,
        model.compile(optimizer='rmsprop',,
                      loss='mse')
  
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense


########################
# Preparate Data
#######################
df = pd.read_csv('fake_reg.csv')

#sns.pairplot(df)
#plt.show()

X = df[['feature1', 'feature2']].values
y = df['price'].values #labels or categories

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Normalise and scale feature data 
# All data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well. 
scaler = MinMaxScaler(feature_range=(0, 1))
#Compute the max and min values to be used in later scaling
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#############################################
#  Build Model
############################################
model = Sequential()

model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=1))  #Output layer with one neuron, as we only wish to output 'price'

model.compile(optimizer='rmsprop', loss='mse')

model.fit(x=X_train, y=y_train, epochs=250)

#plot the loss value at each epoch
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
plt.show()

#############################################
#Evaluating the model
############################################
print('Loss for test data = ', model.evaluate(X_test, y_test, verbose=0))
print('Loss for training data = ', model.evaluate(X_train, y_train, verbose=0))

test_predictions = model.predict(X_test)
test_predictions = pd.Series(test_predictions.reshape(300,))
print(test_predictions)
print('***********************************************')

#print a table of actual prices against the corresponding model prediction
pred_df = pd.DataFrame(y_test, columns=['Test True Y'])
pred_df = pd.concat([pred_df, test_predictions], axis=1) # axis=1, join along columns
pred_df.columns = ['Test True Y', 'Model Predictions']
print(pred_df)

#plot model prediction against actual price
sns.scatterplot(data=pred_df, x='Test True Y', y='Model Predictions')
plt.show()

print('Mean Absolute Error = ', mean_absolute_error(pred_df['Test True Y'], pred_df['Model Predictions']))
print('Mean Squared Error = ', mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions']))

#######################################################
# Use the model to make a prediction
######################################################
new_item = [[998, 1000]]
new_item = scaler.transform(new_item)
print('Predicted price of the new item = ', model.predict(new_item))

#####################################
# Save the model
####################################
model.save('my_new_item_model.keras')

new_model = load_model('my_new_item_model.keras')
print('Predicted price of the new item using the saved model = ', new_model.predict(new_item))
