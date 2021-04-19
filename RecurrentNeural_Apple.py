import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns


df = pd.read_csv('AAPL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
sns.heatmap(df.corr())
plt.hist(df['Open'], bins=40)
sns.rugplot(df['Open'])
sns.pairplot(df)

training_set = df[:'2016'].iloc[:, 1:2].values
test_set = df['2017':].iloc[:, 1:2].values

df['High'][:'2016'].plot(figsize=(16,9), legend=True)
df['High']['2017':].plot(figsize=(16,9), legend=True)
plt.legend(['Training Set (Before 2016)', ' Test Set (After 2017)'])
plt.title("Apple Stock Price")
plt.tight_layout()
plt.show()


from sklearn.preprocessing import MinMaxScaler
sc_x = MinMaxScaler(feature_range=(0,1))
scaled_training_set = sc_x.fit_transform(training_set)


X_train = []
y_train = []

for i in range(60, 2768):
    X_train.append(scaled_training_set[i-60:i, 0])
    y_train.append(scaled_training_set[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)


X_train.shape
y_train.shape

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1 ))


from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential



model = Sequential()
# First LSTM layer with Dropout regularisation
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
# Second LSTM layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Third LSTM layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Fourth LSTM layer
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# For Output layer

model.add(Dense(1))

model.compile(optimizer='rmsprop',loss='mean_squared_error', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, batch_size=32, epochs=50)

model.save_weights('RecurrentNeural_Apple.hdf5')

dataset_total = pd.concat((df['High'][:2016],df['High'][2017:]),axis=0 )
inputs = dataset_total[len(dataset_total)-len(test_set)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc_x.transform(inputs)

X_test = []

for i in range(60, 311):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1 ))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc_x.inverse_transform(predicted_stock_price)

plt.plot(test_set, color='blue', label='Real Stock Price')
plt.plot(predicted_stock_price, color='red',label='Predicted Stock Price')
plt.title("Apple Stock Price predictions")
plt.xlabel('Year')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()



