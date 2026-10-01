import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/prices.csv")
print(df)
print(df.dtypes)

df["Date"] = pd.to_datetime(df["Date"])
print(df.dtypes)
df = df.sort_values("Date")
df = df.set_index("Date")
print(df)
df["Daily_Return"] = df["Close"].pct_change()
print(df[["Close", "Daily_Return"]])
df = df.dropna(subset=["Daily_Return"])
print(df[["Close", "Daily_Return"]])
df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1
print(df[["Close", "Daily_Return", "Cumulative_Return"]])
daily_volatility = df["Daily_Return"].std()
print("Daily volatility:", daily_volatility)
annualized_volatility = daily_volatility * np.sqrt(252)
print("Annualized volatility:", annualized_volatility)
df["Peak"] = df["Close"].cummax()
print(df[["Close", "Peak"]])
df["Drawdown"] = df["Close"] / df["Peak"] - 1
print(df[["Close", "Peak", "Drawdown"]])
max_drawdown = df["Drawdown"].min()
print("Maximum drawdown:", max_drawdown)

df["Close"].plot(title="Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

df["Rolling_Mean_3"] = df["Close"].rolling(window=3).mean()
print(df[["Close", "Rolling_Mean_3"]])
df["Rolling_Vol_3"] = df["Daily_Return"].rolling(window=3).std()
print(df[["Daily_Return", "Rolling_Vol_3"]])

df["Close"].plot(label="Close")
df["Rolling_Mean_3"].plot(label="3-Day Rolling Mean")
plt.title("Closing Price and 3-Day Rolling Mean")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()