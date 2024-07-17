import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l1_l2,l2
from tensorflow.keras.models import load_model
import schedule 
import datetime as dt



def extract_data_table(url):
    try:
        # Send a GET request to the PHP site
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')

            # Use BeautifulSoup to find the HTML table containing data
            # This depends on the specific structure of the PHP-generated HTML.
            data_table = soup.find('table', class_='table table-striped table-bordered datatable')

            # Extract the text content of the table
            table_data = []
            if data_table:
                for row in data_table.find_all('tr'):
                    columns = [col.text.strip() for col in row.find_all(['th', 'td'])]
                    table_data.append(columns)

                return table_data
            else:
                print("Data table not found on the page.")
                return None

        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Replace this URL with the actual URL of the PHP site
php_site_url = "https://ideabeam.com/finance/rates/goldprice.php"
table_data = extract_data_table(php_site_url)

if table_data:
    # Convert the table data to a Pandas DataFrame
    data = pd.DataFrame(table_data[1:], columns=table_data[0])

    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

    print(data)

train_dates = data['Date'][:-1]
data.set_index('Date', inplace=True)

data['24 Carat 1 Gram'] = pd.to_numeric(data['24 Carat 1 Gram'].str.replace('Rs. ', '').str.replace(',', ''), errors='coerce')
data = data[:-1]
data['24 Carat 1 Gram'] = data['24 Carat 1 Gram'].astype(int)
temp = data[['24 Carat 1 Gram']]
temp.to_csv('existing_data.csv')

existing_data = pd.read_csv('existing_data.csv',index_col = 'Date')
existing_data.update(temp['24 Carat 1 Gram'])
existing_data.to_csv('existing_data.csv')

temp = pd.read_csv('existing_data.csv',index_col = 'Date')

print(temp.dtypes)

scaler = MinMaxScaler()
temp_normalized = scaler.fit_transform(temp.values.reshape(-1, 1))

def df_to_X_y(df, window_size):
    X = []
    y = []
    for i in range(len(df)-window_size):
        row = [[a] for a in df[i:i+window_size]]
        X.append(row)
        label = df[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)

WINDOW_SIZE = 6
X1, y1 = df_to_X_y(temp_normalized, WINDOW_SIZE)
print(X1.shape, y1.shape)

X_train1, y_train1 = X1[:30], y1[:30]
X_val1, y_val1 = X1[30:40], y1[30:40]
X_test1, y_test1 = X1[40:], y1[40:]
print(X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape)


model1 = Sequential()
model1.add(InputLayer((WINDOW_SIZE, 1)))
model1.add(LSTM(30))
model1.add(Dense(60, activation='relu'))
model1.add(Dropout(0.3))
model1.add(Dense(120, activation='relu'))
model1.add(Dense(30, activation='relu'))
model1.add(Dense(1,activation='linear'))


cp1 = ModelCheckpoint('model1/model_checkpoint.keras', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])

model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=30,batch_size = 10, callbacks=[cp1])

model1 = load_model('model1/model_checkpoint.keras')

train_predictions = scaler.inverse_transform(model1.predict(X_train1).reshape(-1,1)).flatten()
y_train1 = scaler.inverse_transform(y_train1.reshape(-1,1)).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
print(train_results)

n_future_predict = 10
forecasting_dates = pd.date_range(list(train_dates)[1],periods=n_future_predict,freq='1d').tolist()

forecasts = model1.predict(X_train1[-n_future_predict:])
true_forecasts = scaler.inverse_transform(forecasts.reshape(-1,1)).flatten()

print(pd.DataFrame(true_forecasts))

email_data = train_results



def job():
    print(f"predictions for upcoming dates are {true_forecasts}")


schedule.every().day.at("13:00").do(job)


# Plotting the results
plt.figure(figsize=(14, 7))

# Plot actual values
plt.plot(train_dates, temp['24 Carat 1 Gram'], label='Actuals')

# Plot train predictions
plt.plot(train_dates[:len(train_predictions)], train_predictions, label='Train Predictions')

# Plot forecasted values
forecast_dates = pd.date_range(train_dates.iloc[1], periods=n_future_predict)
plt.plot(forecast_dates, true_forecasts, label='Forecasts', linestyle='dashed')

plt.xlabel('Date')
plt.ylabel('24 Carat 1 Gram Price (Rs)')
plt.title('Gold Price Predictions')
plt.legend()
plt.show()