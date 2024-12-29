import pandas as pd
import datetime
import numpy as np
from keras import Sequential
from matplotlib import pyplot as plt

from tensorflow.keras.models import *
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------------------------------------------------
# 1) Read CSV and Parse Dates
# -------------------------------------------------------------------
df = pd.read_csv('A.csv')

# If your CSV dates are in "MM/DD/YYYY" format (like "4/1/2019"):
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# If your CSV is already in ISO format "YYYY-MM-DD" or can be parsed automatically:
# df['Date'] = pd.to_datetime(df['Date'])

# Set the Date column as the index (DatetimeIndex)
df.set_index('Date', inplace=True)

# Keep only the columns we need
df = df[['Close']]


# -------------------------------------------------------------------
# 2) df_to_windowed_df Function
# -------------------------------------------------------------------
def df_to_windowed_df(dataframe, first_date_str, last_date_str, n):
    """
    dataframe: DataFrame with a DatetimeIndex
    first_date_str, last_date_str: strings like '2019-04-01'
    n: number of past observations to use in each window
    """
    first_date = pd.to_datetime(first_date_str)
    last_date = pd.to_datetime(last_date_str)

    target_date = first_date
    dates = []
    X, Y = [], []

    while target_date <= last_date:
        # Make sure the target_date is actually in the DataFrame index
        # or find the closest valid date before/at target_date
        if target_date not in dataframe.index:
            # Option A: skip this date if it's not in the data
            # target_date += datetime.timedelta(days=1)
            # continue

            # Option B: use "asof" approach to find the closest previous date
            # to target_date that actually exists in the index:
            valid_index = dataframe.index.asof(target_date)
            if pd.isna(valid_index):
                # No valid prior date -> break or continue
                target_date += datetime.timedelta(days=1)
                continue
            else:
                target_date = valid_index

        # Now slice up to target_date and take last (n+1) rows
        df_subset = dataframe.loc[:target_date].tail(n + 1)

        # If not enough data, break
        if len(df_subset) < n + 1:
            break

        # Convert the 'Close' column to numpy
        values = df_subset["Close"].to_numpy()

        # X = first n values, Y = the last value (the "target date" close)
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        # Move to the next date in the DataFrame index
        # (the day after the current target_date)
        # We'll find the exact location of target_date in the index, then go +1
        try:
            current_loc = dataframe.index.get_loc(target_date)
            next_loc = current_loc + 1
            # If next_loc is out of range, break
            if next_loc >= len(dataframe.index):
                break
            next_date = dataframe.index[next_loc]
            target_date = next_date
        except KeyError:
            # If we can't find target_date in the index, break or handle differently
            break

    # Create the resulting DataFrame
    ret_df = pd.DataFrame()
    ret_df["Target Date"] = dates

    # Convert X to a numpy array for easy column extraction
    X = np.array(X)

    # For n=3, we want columns: Target-3, Target-2, Target-1
    # Generally: Target-(n), Target-(n-1), ...
    for i in range(n):
        ret_df[f"Target-{n - i}"] = X[:, i]

    ret_df["Target"] = Y
    return ret_df


# -------------------------------------------------------------------
# 3) Create the Windowed DataFrame
# -------------------------------------------------------------------
# Example date range:
window_df = df_to_windowed_df(df, '2019-04-01', '2020-04-1', n=3)

# -------------------------------------------------------------------
# 4) Add Additional Columns (Return, RSI, Bollinger, etc.)
# -------------------------------------------------------------------
# We'll merge the window_df on "Target Date" as needed,
# or just keep going with window_df if it's all we need.
# For demonstration, let's continue with window_df alone.

window_df['Return'] = (window_df['Target'] - window_df['Target-1']) / window_df['Target-1']


def calculate_rsi(window_df, column='Return', window=14):
    delta = window_df[column]
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    window_df['RSI'] = rsi
    return window_df


def calculate_bollinger_bands(window_df, column='Target', window=14, num_std_dev=3):
    window_df['Middle Band'] = window_df[column].rolling(window=window).mean()
    rolling_std = window_df[column].rolling(window=window).std()

    window_df['Upper Band'] = window_df['Middle Band'] + (rolling_std * num_std_dev)
    window_df['Lower Band'] = window_df['Middle Band'] - (rolling_std * num_std_dev)
    return window_df


window_df = calculate_rsi(window_df, column='Return', window=14)
window_df = calculate_bollinger_bands(window_df, column='Target', window=14, num_std_dev=3)

# Drop any rows that have NaN due to rolling calculations
window_df.dropna(inplace=True)

# -------------------------------------------------------------------
# 5) Prepare Data for LSTM
# -------------------------------------------------------------------
feature_columns = [
    'Target-3', 'Target-2', 'Target-1',
    'Return', 'RSI', 'Middle Band', 'Upper Band', 'Lower Band'
]

X = window_df[feature_columns].values
y = window_df['Target'].values

# Standard scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y = y.reshape(-1, 1)
y_scaled = scaler_y.fit_transform(y)


# Create sequences
def create_sequences(x, y, lookback=5):
    xs, ys = [], []
    for i in range(len(x) - lookback):
        xs.append(x[i:(i + lookback), :])
        ys.append(y[i + lookback])
    return np.array(xs), np.array(ys)


LOOKBACK = 5
X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback=LOOKBACK)

test_size = 0.2
num_samples = X_seq.shape[0]
train_samples = int((1 - test_size) * num_samples)

X_train = X_seq[:train_samples]
y_train = y_seq[:train_samples]
X_test = X_seq[train_samples:]
y_test = y_seq[train_samples:]

# -------------------------------------------------------------------
# 6) Build & Train LSTM Model
# -------------------------------------------------------------------
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(LOOKBACK, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

# -------------------------------------------------------------------
# 7) Evaluate
# -------------------------------------------------------------------
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

print(f"Test MSE:  {mse}")
print(f"Test RMSE: {rmse}")
print(f"Test MAE:  {mae}")


from copy import deepcopy

recursive_dates = np.concatenate([y_pred, y_true])

# Suppose you want to do 10 future predictions:
n_future = 1

# 1) Take the last window from your test set (shape: (lookback, num_features))
last_sequence = deepcopy(X_test[-1])

# 2) We'll store the predictions (in scaled form initially, then invert scale)
recursive_predictions_scaled = []

# 3) Loop n_future times
for i in range(n_future):
    # LSTM expects shape (1, lookback, num_features) for a single prediction
    next_pred_scaled = model.predict(last_sequence[np.newaxis, :, :])[0, 0]
    # [0, 0] extracts the single float from shape (1,1)

    recursive_predictions_scaled.append(next_pred_scaled)

    # SHIFT the window by 1 to the left:
    # Move rows up (drop the oldest time step)
    last_sequence[:-1, :] = last_sequence[1:, :]

    # The last row in `last_sequence` is now empty and should incorporate the new predicted target
    # If your target is the LAST column in your features:
    # Copy forward or estimate the other features, or set them to 0 if not known.
    # For a simple approach, let's just replicate the second-to-last row's features
    # except for the target column:
    last_sequence[-1, :-1] = last_sequence[-2, :-1]

    # Finally, set the newly predicted value in the last column:
    last_sequence[-1, -1] = next_pred_scaled

# Now you have `recursive_predictions_scaled` of length n_future
# Convert to a NumPy array for convenience
recursive_predictions_scaled = np.array(recursive_predictions_scaled).reshape(-1, 1)

# 4) Invert scale if your target was scaled. Suppose you have `scaler_y` for the target:
recursive_predictions = scaler_y.inverse_transform(recursive_predictions_scaled).flatten()


#
# for i in range(0,49):
#     recursive_predictions = np.insert(recursive_predictions, 0, 0., axis=0)
# print("Recursive multi-step predictions:", recursive_predictions)

print(y_true)
print("Recursive multi-step predictions:", recursive_predictions)
plt.figure(figsize=(10,6))
plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Predicted')
# plt.plot(recursive_predictions, label='Recursive predictions')
plt.title("Stock Price Prediction")
plt.xlabel("Time (Test Set Index)")
plt.ylabel("Price")
plt.legend()
plt.show()
