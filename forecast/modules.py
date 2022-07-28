import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
%matplotlib inline

# For reading stock data from yahoo
!pip install yfinance

from pandas_datareader.data import DataReader
import yfinance as yf

# For time stamps
from datetime import datetime


# The tech stocks we'll use for this analysis
tech_list = ['AAPL', 'TSLA', 'AMZN']

# Set up End and Start times for data grab
tech_list = ['AAPL', 'TSLA', 'AMZN']

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)


company_list = [AAPL, TSLA, AMZN]
company_name = ["Apple", "Tesla", "Amazon"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
    
df = pd.concat(company_list, axis=0)
df.tail(10) ##changed by me to head change back to tail if error


# Summary Stats
AAPL.describe()

# General info
AAPL.info()

plt.figure(figsize=(15, 6))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {tech_list[i - 1]}")
    
plt.tight_layout()


# Now let's plot the total volume of stock being traded each day
plt.figure(figsize=(15, 7))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Volume'].plot()
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"Sales Volume for {tech_list[i - 1]}")
    
plt.tight_layout()



# We'll use pct_change to find the percent change for each day
for company in company_list:
    company['Daily Return'] = company['Adj Close'].pct_change()

# Then we'll plot the daily return percentage
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(8)
fig.set_figwidth(15)

AAPL['Daily Return'].plot(ax=axes[0,0], legend=True, linestyle='--', marker='o')
axes[0,0].set_title('APPLE')

TSLA['Daily Return'].plot(ax=axes[1,0], legend=True, linestyle='--', marker='o')
axes[1,0].set_title('TESLA')

AMZN['Daily Return'].plot(ax=axes[1,1], legend=True, linestyle='--', marker='o')
axes[1,1].set_title('AMAZON')

fig.tight_layout()


#daily return checker

plt.figure(figsize=(12, 7))

for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Daily Return'].hist(bins=50)
    plt.ylabel('Daily Return')
    plt.title(f'{company_name[i - 1]}')
    
plt.tight_layout()


AAPL = yf.download('AAPL')

AMZN = yf.download('AMZN')

TSLA = yf.download('TSLA')


# Set up End and Start times for data grab

df = pd.concat([AAPL, AMZN, TSLA])
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

for stock in tech_list:
   # globals()[stock] = yf.download(stock, start = '2015-01-01', end)
    data = yf.download(stock, start="2015-01-01", end="2022-07-21")
    print('data fields downloaded:', set(data.columns.get_level_values(0)))


data



import pandas as pd


plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(AAPL.iloc[(AAPL.shape[0]-1700):(AAPL.shape[0]), :]['Close'], label="APPLE")
plt.plot(AMZN.iloc[(AMZN.shape[0]-1700):(AMZN.shape[0]), :]['Close'], label="AMAZON")
plt.plot(TSLA.iloc[(TSLA.shape[0]-1700):(TSLA.shape[0]), :]['Close'], label="TESLA")

plt.legend(loc="upper left", fontsize=18)

plt.xlabel('Years', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()


# Create a new dataframe with only the 'Close column 
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))

training_data_len


# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


train_data = scaled_data[0:int(training_data_len), :]

x_train = []
y_train = []

for i in range(60,len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i, 0])
  if i <= 61:
    print(x_train)
    print(y_train)
    print()


x_train , y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
#model.add(LSTM(128, return_sequences=True, input_shape = (x_train.shape, [1], 1)))

model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences  = False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(x_train, y_train, batch_size = 128, epochs = 3)



test_data = scaled_data[training_data_len - 60: ,:]
x_test=[]

y_test = dataset[training_data_len:,:]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)

x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rsme = np.sqrt(np.mean(((predictions - y_test)**2)))

rsme


#accuracy
import numpy as np
from sklearn.metrics import balanced_accuracy_score

#define array of actual classes
actual = np.repeat([1, 0], repeats=[20, 380])

#define array of predicted classes
pred = np.repeat([1, 0, 1, 0], repeats=[15, 5, 5, 375])

#calculate balanced accuracy score
balanced_accuracy_score(actual, pred)


# Plot
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions


# Visualize 
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
#plt.plot(train.iloc[(train.shape[0]-600) : (train.shape[0]), :]['Close'])
plt.plot(valid.iloc[(valid.shape[0]-600) : (valid.shape[0]), :][['Close', 'Predictions']])
plt.legend(['Val', 'Predictions'], loc='lower right')
plt.show()


