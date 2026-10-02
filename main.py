import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("data/prices.csv")

# 2. Prepare time-series data
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")
df = df.set_index("Date")

# 3. Calculate returns and risk metrics
df["Daily_Return"] = df["Close"].pct_change()

df = df.dropna(subset=["Daily_Return"])

df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1

daily_volatility = df["Daily_Return"].std()

annualized_volatility = daily_volatility * np.sqrt(252)

df["Peak"] = df["Close"].cummax()

df["Drawdown"] = df["Close"] / df["Peak"] - 1

max_drawdown = df["Drawdown"].min()


# 4. Calculate rolling statistics
df["Rolling_Mean_3"] = df["Close"].rolling(window=3).mean()

df["Rolling_Vol_3"] = df["Daily_Return"].rolling(window=3).std()

# 5. Create visualizations

df["Close"].plot(title="Closing Price")

plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()

plt.savefig("figures/closing price.png")
plt.show()

df["Close"].plot(label="Close")
df["Rolling_Mean_3"].plot(label="3-Day Rolling Mean")

plt.title("Closing Price and 3-Day Rolling Mean")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()

plt.savefig("figures/close_and_rolling_mean.png")
plt.show()

total_return = df["Cumulative_Return"].iloc[-1]


summary = {
    "Total Return": total_return,
    "Annualized Volatility": annualized_volatility,
    "Max Drawdown": max_drawdown,
}


summary_df = pd.DataFrame([summary])
print(summary_df.to_string(index=False))

summary_df.to_csv("outputs/summary.csv", index=False)
df.to_csv("outputs/processed_prices.csv")
