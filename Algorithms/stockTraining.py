from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def classify_action(row):

    if (row['RSI'] < 30 and
            row['Close'] <= row['Lower Band'] and
            row['Return'] < 0 and
            row['Volume'] > row['Volume_Rolling_Mean']):
        return 1  # Buy

    elif (row['RSI'] > 70 and
          row['Close'] >= row['Upper Band'] and
          row['Return'] > 0 and
          row['Volume'] < row['Volume_Rolling_Mean']):
        return -1  # Sell


    else:
        return 0  # Hold

def calculate_bollinger_bands(window_df, column='Target', window=14, num_std_dev=2):
    window_df['Middle Band'] = window_df[column].rolling(window=window).mean()
    rolling_std = window_df[column].rolling(window=window).std()

    window_df['Upper Band'] = window_df['Middle Band'] + (rolling_std * num_std_dev)
    window_df['Lower Band'] = window_df['Middle Band'] - (rolling_std * num_std_dev)
    return window_df


def find_matching_date(self, first_date):

    current_date = datetime.strptime(first_date, "%Y-%m-%d")

    if not isinstance(self.df.index, pd.DatetimeIndex):
        self.df.index = pd.to_datetime(self.df.index, errors='coerce')

    available_dates = set(self.df.index[self.df.index > current_date].date)

    if not available_dates:
        raise ValueError("No available dates after the specified first_date.")

    while current_date.date() not in available_dates:
        current_date += timedelta(days=1)

    return current_date.strftime("%Y-%m-%d")


def create_sequences(x, y, lookback=5):
    xs, ys = [], []
    for i in range(len(x) - lookback):
        xs.append(x[i:(i + lookback), :])
        ys.append(y[i + lookback])
    return np.array(xs), np.array(ys)

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n):

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
    Y = np.array(Y)

    for i in range(n):
        ret_df[f"Target-{n - i}"] = X[:, i]

    ret_df['Target'] = Y
    return ret_df

class Analysis:
    def __init__(self, stock_name):
        self.stock_name = stock_name  # Store the stock name as an attribute
        self.stock_file = f'DataSets/{stock_name}.csv'  # Construct file path
        self.df = self.load_data()  # Load the DataFrame as an attribute
        self.df['Volume_Rolling_Mean'] = self.df['Volume'].rolling(window=20).mean()
        self.df['Return'] = (self.df['Close'] - self.df['Close'].shift(1)) / self.df['Close'].shift(1)

    def load_data(self):

        try:
            df = pd.read_csv(self.stock_file)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="mixed")
            df.set_index('Date', inplace=True)  # Set Date as the index
            return df
        except FileNotFoundError:
            print(f"File {self.stock_file} not found.")
            return None

    def get_close_plot(self):

        if self.df is None:
            print("DataFrame is not loaded. Cannot generate plot.")
            return None

        plt.figure(figsize=(10, 6))
        plt.plot(self.df.index, self.df['Close'], label='Close Price', linewidth=2)


        plt.title(f'Close Price Over Time for {self.stock_name}', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Close Price', fontsize=12)
        plt.grid(True)


        plt.legend()


        plot_path = f"static/Images/{self.stock_name}_close_plot.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def get_rsi_plot(self):
        self.df['RSI'] = calculate_rsi(self.df['Close'])
        self.df = self.df.dropna()

        last_date = self.df.index.max()


        cutoff_date = last_date - timedelta(days=365)


        part_df = self.df[self.df.index > cutoff_date]

        plt.figure(figsize=(12, 6))
        plt.plot(part_df.index, part_df['RSI'], label='RSI', color='blue', linewidth=1)
        plt.axhline(70, color='red', linestyle='--', label='Overbought')
        plt.axhline(30, color='green', linestyle='--', label='Oversold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('RSI Value', fontsize=12)
        plt.legend()
        plt.title('Relative Strength Index (RSI)')
        plot_path = f"static/Images/{self.stock_name}_rsi_plot.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path


    def get_bollinger_bands_plot(self):
        self.df = calculate_bollinger_bands(self.df, column='Close', window=14, num_std_dev=2)
        self.df = self.df.dropna()
        last_date = self.df.index.max()
        self.df['Action'] = self.df.apply(classify_action, axis=1)


        cutoff_date = last_date - timedelta(days=365)


        part_df = self.df[self.df.index > cutoff_date]
        plt.figure(figsize=(12, 6))
        plt.plot(part_df['Close'], label='Close Price', color='blue')
        plt.plot(part_df['Middle Band'], label='Middle Band (SMA)', color='orange')
        plt.plot(part_df['Upper Band'], label='Upper Band', color='green', linestyle='--')
        plt.plot(part_df['Lower Band'], label='Lower Band', color='red', linestyle='--')
        plt.fill_between(part_df.index, part_df['Upper Band'], part_df['Lower Band'], color='gray', alpha=0.1)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Close prices', fontsize=12)
        plt.title('Bollinger Bands')
        plt.legend()
        plot_path = f"static/Images/{self.stock_name}_bollinger_band_plot.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def classify_stock(self, row):
        self.df['Action'] = self.df.apply(row.classify_action, axis=1)
        return self.df['Action']


    def train_model_with_df(self, test_size):
        feature_columns = ['Close', 'Return', 'RSI', 'Upper Band', 'Lower Band', 'Middle Band', 'Volume']
        X = self.df[feature_columns]
        y = self.df['Action']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)


        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)


        plt.figure(figsize=(12, 8))
        plot_tree(clf, feature_names=feature_columns, class_names=["Hold", "Sell", "Buy"], filled=True)
        plt.title("Decision Tree for Stock Trading")
        plot_path = f"static/Images/{self.stock_name}_decision_tree.png"
        plt.savefig(plot_path)
        plt.close()
        return classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred), plot_path


    def train_model_with_KNN(self, test_size, n, lookback):
        first_date = '2019-04-01'
        matching_date = find_matching_date(self, first_date)
        window_df = df_to_windowed_df(self.df, matching_date, '2020-04-01', n=n)
        window_df['Return'] = (window_df['Target'] - window_df['Target-1']) / window_df['Target-1']
        window_df['RSI'] = calculate_rsi(window_df['Target'])
        window_df = calculate_bollinger_bands(window_df, column='Target', window=14, num_std_dev=2)
        window_df.dropna(inplace=True)
        feature_columns = [
            'Target-3', 'Target-2', 'Target-1',
            'Return', 'RSI', 'Middle Band', 'Upper Band', 'Lower Band'
        ]

        X = window_df[feature_columns].values
        y = window_df['Target'].values


        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y = y.reshape(-1, 1)
        y_scaled = scaler_y.fit_transform(y)

        LOOKBACK = lookback
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback=LOOKBACK)

        # Split into training and testing
        test_size = test_size
        num_samples = X_seq.shape[0]
        train_samples = int((1 - test_size) * num_samples)


        X_knn = X_scaled
        y_knn = y_scaled.flatten()

        X_train_knn = X_knn[:train_samples]
        y_train_knn = y_knn[:train_samples]
        X_test_knn = X_knn[train_samples:]
        y_test_knn = y_knn[train_samples:]
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train_knn, y_train_knn)


        y_pred_knn_scaled = knn.predict(X_test_knn)
        y_pred_knn = scaler_y.inverse_transform(y_pred_knn_scaled.reshape(-1, 1)).flatten()
        y_true_knn = scaler_y.inverse_transform(y_test_knn.reshape(-1, 1)).flatten()


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
        plot_path = f"static/Images/{self.stock_name}_knn.png"
        plt.savefig(plot_path)
        plt.close()
        return mse_knn, r2_knn, mape_knn, accuracy_knn, plot_path

