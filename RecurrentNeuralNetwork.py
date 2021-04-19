import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset_train = pd.read_csv('###.csv')

sns.pairplot(dataset_train)
sns.heatmap(dataset_train.isnull(),cbar=False,yticklabels=False,cmap='viridis')
plt.hist(dataset_train['Open'],bins=35)

training_set = dataset_train.iloc[:, 1:2].values

# For RNN don't generally use Standardisation instead use Normalisation(MIN_MAX_SCALER)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 timesteps and 1 outputs

X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train.shape
y_train.shape

# Dimension that we need to convert it into should be found online on keras documentation

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1 ))

# End of Data Preprocessing

# Part -2 Building The Recurrent Neural Network
import tensorflow as tf
tf.__version__
from keras.models import Sequential
from keras.layers import Dense, LSTM , Dropout

regressor = Sequential()
# First LSTM Layer and some Dropout to Prevent Overfitting

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1 )))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))

# Output Layer

regressor.add(Dense(1))

regressor.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

regressor.fit(X_train , y_train, epochs=100, batch_size=32)

regressor.summary()

# Now Visualising the Data with the Test set and Predicted Set and finding out the accuracy

# Getting Real Stock price of the year 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# Getting the predicted Stock price of the year 2017 from our model
# Concatenation of the pandas dataframe instead of using .values so that we can normalize it

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0 )
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs) # Dont use fit_transform because we want the same scaling to be fitted as the original training
                              # so by using transform the inputs are same standard as the training set

X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

# Reshaping into 3d dimension of keras documentation

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1 ))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)  # To get real values and not the Normalized Values

# Visualising
plt.plot(real_stock_price, color='blue',ls='--', label='Real Stock Price')
plt.plot(predicted_stock_price, color='red',ls='-.',label='Predicted Stock Price')
plt.title("Google Stock Price predictions")
plt.xlabel('Year')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
