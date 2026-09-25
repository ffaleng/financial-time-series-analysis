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