import pandas as pd
import numpy as np
import datetime
from keras import Sequential
from matplotlib import pyplot as plt
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor


def df_to_windowed_df(dataframe, first_date_str, last_date_str, n):
    """
    Create a windowed DataFrame for LSTM modeling.
    dataframe: DataFrame with a DatetimeIndex
    first_date_str, last_date_str: strings like '2019-04-01'
    n: number of past observations to use in each window
    """
    first_date = pd.to_datetime(first_date_str)
    last_date = pd.to_datetime(last_date_str)

    dates, X, Y = [], [], []
    target_date = first_date

    while target_date <= last_date:
        if target_date not in dataframe.index:
            target_date = dataframe.index.asof(target_date)
            if pd.isna(target_date):
                target_date += datetime.timedelta(days=1)
                continue

        df_subset = dataframe.loc[:target_date].tail(n + 1)

        if len(df_subset) < n + 1:
            break

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_loc = dataframe.index.get_loc(target_date) + 1
        if next_loc >= len(dataframe.index):
            break

        target_date = dataframe.index[next_loc]

    ret_df = pd.DataFrame()
    ret_df['Target Date'] = dates
    X = np.array(X)

    for i in range(n):
        ret_df[f"Target-{n - i}"] = X[:, i]

    ret_df['Target'] = Y
    return ret_df


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


def calculate_bollinger_bands(window_df, column='Target', window=14, num_std_dev=2):
    window_df['Middle Band'] = window_df[column].rolling(window=window).mean()
    rolling_std = window_df[column].rolling(window=window).std()

    window_df['Upper Band'] = window_df['Middle Band'] + (rolling_std * num_std_dev)
    window_df['Lower Band'] = window_df['Middle Band'] - (rolling_std * num_std_dev)
    return window_df


def create_sequences(x, y, lookback=5):
    xs, ys = [], []
    for i in range(len(x) - lookback):
        xs.append(x[i:(i + lookback), :])
        ys.append(y[i + lookback])
    return np.array(xs), np.array(ys)


df = pd.read_csv('A.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.set_index('Date', inplace=True)


window_df = df_to_windowed_df(df, '2019-04-01', '2020-04-01', n=3)

window_df['Return'] = (window_df['Target'] - window_df['Target-1']) / window_df['Target-1']


window_df = calculate_rsi(window_df, column='Return', window=14)
window_df = calculate_bollinger_bands(window_df, column='Target', window=14, num_std_dev=2)
window_df.dropna(inplace=True)


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

# Create sequences for LSTM

LOOKBACK = 5
X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback=LOOKBACK)

# Split into training and testing
test_size = 0.2
num_samples = X_seq.shape[0]
train_samples = int((1 - test_size) * num_samples)


# Prepare data for KNN
X_knn = X_scaled
y_knn = y_scaled.flatten()

X_train_knn = X_knn[:train_samples]
y_train_knn = y_knn[:train_samples]
X_test_knn = X_knn[train_samples:]
y_test_knn = y_knn[train_samples:]


knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_knn, y_train_knn)

# Predict with KNN
y_pred_knn_scaled = knn.predict(X_test_knn)
y_pred_knn = scaler_y.inverse_transform(y_pred_knn_scaled.reshape(-1, 1)).flatten()
y_true_knn = scaler_y.inverse_transform(y_test_knn.reshape(-1, 1)).flatten()

# Evaluate KNN
mse_knn = mean_squared_error(y_true_knn, y_pred_knn)
r2_knn = r2_score(y_true_knn, y_pred_knn)
mape_knn = np.mean(np.abs((y_true_knn - y_pred_knn) / y_true_knn)) * 100
threshold = 0.05
accuracy_knn = np.mean(np.abs((y_true_knn - y_pred_knn) / y_true_knn) <= threshold) * 100

print(f"KNN Test MSE:  {mse_knn}")
print(f"KNN R² Score: {r2_knn}")
print(f"KNN MAPE: {mape_knn}%")
print(f"KNN Accuracy within {threshold * 100}%: {accuracy_knn}%")

plt.figure(figsize=(10, 6))
plt.plot(y_true_knn, label='Actual')
plt.plot(y_pred_knn, label='Predicted (KNN)')
plt.title("KNN Stock Price Prediction")
plt.xlabel("Time (Test Set Index)")
plt.ylabel("Price")
plt.legend()
plt.show()
