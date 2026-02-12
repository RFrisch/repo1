import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Config: moving average windows (easy to tweak)
SHORT_WINDOW = 40
LONG_WINDOW = 100

# Define a function to generate signals based on moving average crossover
def generate_signals(data, short_window=SHORT_WINDOW, long_window=LONG_WINDOW):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    
    # Create short-term simple moving average
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    
    # Create long-term simple moving average
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    
    # Generate signals
    signals.loc[signals.index[short_window:], 'signal'] = np.where(
        signals['short_mavg'].iloc[short_window:] > signals['long_mavg'].iloc[short_window:], 1.0, 0.0)
    
    # Generate trading orders
    signals['positions'] = signals['signal'].diff()
    
    return signals

# Load historical price data (replace this with your own data)
# For the sake of demonstration, let's use some random price data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2024-01-01')
prices = np.random.normal(loc=100, scale=5, size=len(dates))
data = pd.DataFrame({'Close': prices}, index=dates)

# Generate trading signals
signals = generate_signals(data)

# Simple strategy stats: buy-and-hold when signal is 1
n_buys = (signals['positions'] == 1.0).sum()
n_sells = (signals['positions'] == -1.0).sum()
returns = data['Close'].pct_change()
strategy_returns = returns * signals['signal'].shift(1)
cumulative_return = (1 + strategy_returns).prod() - 1
print(f"MA Crossover ({SHORT_WINDOW}/{LONG_WINDOW}): {n_buys} buys, {n_sells} sells | Cumulative return: {cumulative_return:.2%}")

# Plotting the closing price and moving averages
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.plot(signals['short_mavg'], label=f'{SHORT_WINDOW}-Day Moving Average')
plt.plot(signals['long_mavg'], label=f'{LONG_WINDOW}-Day Moving Average')

# Plot buy signals
plt.plot(signals.loc[signals.positions == 1.0].index, 
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='g', lw=0, label='Buy Signal')

# Plot sell signals
plt.plot(signals.loc[signals.positions == -1.0].index, 
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title(f'Moving Average Crossover Strategy ({SHORT_WINDOW}/{LONG_WINDOW})')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


