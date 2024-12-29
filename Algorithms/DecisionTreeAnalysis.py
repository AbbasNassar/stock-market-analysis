import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dropout, Dense
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read the stock data
df = pd.read_csv('A.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.set_index('Date', inplace=True)

df['Return'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
df['Middle Band'] = df['Close'].rolling(window=20).mean()
rolling_std = df['Close'].rolling(window=20).std()
df['Upper Band'] = df['Middle Band'] + (2 * rolling_std)
df['Lower Band'] = df['Middle Band'] - (2 * rolling_std)
df['Volume_Rolling_Mean'] = df['Volume'].rolling(window=20).mean()
df

# RSI function (as per earlier code)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df['RSI'] = calculate_rsi(df['Close'])

df

df = df.dropna()

df

# Calculate rolling mean for Volume (outside the apply function)
df['Volume_Rolling_Mean'] = df['Volume'].rolling(window=20).mean()

# Define the decision labels with precomputed rolling mean
def classify_action(row):
    # Buy conditions
    if (row['RSI'] < 30 and
        row['Close'] <= row['Lower Band'] and
        row['Return'] < 0 and
        row['Volume'] > row['Volume_Rolling_Mean']):  # High volume
        return 1  # Buy

    # Sell conditions
    elif (row['RSI'] > 50 and
          row['Close'] >= row['Upper Band'] and
          row['Return'] > 0 and
          row['Volume'] < row['Volume_Rolling_Mean']):  # Low volume
          return -1  # Sell

    # Hold conditions
    elif (row['RSI'] >= 30 and row['RSI'] <= 50 and
          row['Close'] > row['Lower Band'] and row['Close'] < row['Upper Band']):
        return 0  # Hold

    # Default to Hold if none of the above conditions match
    else:
        return 0  # Hold

# Apply the classification function
df['Action'] = df.apply(classify_action, axis=1)

# Drop rows with NaN values introduced by rolling calculations
df.dropna(inplace=True)

df

# Select features and target variable
feature_columns = ['Close','Return', 'RSI', 'Upper Band', 'Lower Band', 'Middle Band', 'Volume']
X = df[feature_columns]
y = df['Action']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=feature_columns, class_names=["Hold", "Sell", "Buy"], filled=True)
plt.title("Decision Tree for Stock Trading")
plt.show()