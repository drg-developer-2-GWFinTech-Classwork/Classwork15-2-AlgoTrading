import os
import ccxt
import asyncio
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt

global new_df  # = pd.DataFrame()


def initialize(cash=None):
    """Initialize the dashboard, data storage, and account balances."""
    print("Initializing Account and DataFrame")

    # @TODO: Update to build the plot
    # Initialize Account
    account = {"balance": cash, "shares": 0}

    # Initialize DataFrame
    df = fetch_data()

    # Initialize the plot

    return account, df


def build_plot(df):
    """Build the plot."""

    plt = df.hvplot.line()
    return plt


def build_dashboard():
    """Build the dashboard."""

    price_df = signals_df[["close", "SMA50", "SMA100"]]
    price_chart = price_df.hvplot.line()
    price_chart.opts(xaxis=None)

    portfolio_evaluation_df.reset_index(inplace=True)
    portfolio_evaluation_table = portfolio_evaluation_df.hvplot.table()

    trade_evaluation_table = trade_evaluation_df.hvplot.table()

    # Assemble dashboard visualization
    return price_chart + portfolio_evaluation_table + trade_evaluation_table


def update_plot():
    return None


def fetch_data():
    """Fetches the latest prices."""
    print("Fetching data...")
    load_dotenv()
    kraken_public_key = os.getenv("KRAKEN_PUBLIC_KEY")
    kraken_secret_key = os.getenv("KRAKEN_SECRET_KEY")
    kraken = ccxt.kraken({"apiKey": kraken_public_key, "secret": kraken_secret_key})

    close = kraken.fetch_ticker("BTC/USD")["close"]
    datetime = kraken.fetch_ticker("BTC/USD")["datetime"]
    df = pd.DataFrame({"close": [close]})
    df.index = pd.to_datetime([datetime])

    new_df = df

    return df


def generate_signals(df):
    """Generates trading signals for a given dataset."""
    print("-----> Generating trading signals <-----")
    # Set window
    short_window = 10

    signals = df.copy()
    signals["signal"] = 0.0

    # Generate the short and long moving averages
    signals["sma10"] = signals["close"].rolling(window=10).mean()
    signals["sma20"] = signals["close"].rolling(window=20).mean()

    # Generate the trading signal 0 or 1,
    signals["signal"][short_window:] = np.where(
        signals["sma10"][short_window:] > signals["sma20"][short_window:], 1.0, 0.0
    )

    # Calculate the points in time at which a position should be taken, 1 or -1
    signals["entry/exit"] = signals["signal"].diff()
    print("-----> Trading signals generated  <-----")

    return signals


def execute_trade_strategy(signals, account):
    """Makes a buy/sell/hold decision."""

    print("**Executing Trading Strategy**")

    if signals["entry/exit"].iloc[-1] == 1.0:
        print("Buy")
        number_to_buy = round(account["balance"] / signals["close"].iloc[-1], 0) * 0.001
        account["balance"] -= number_to_buy * signals["close"].iloc[-1]
        account["shares"] += number_to_buy
    elif signals["entry/exit"].iloc[-1] == -1.0:
        print("Sell")
        account["balance"] += signals["close"].iloc[-1] * account["shares"]
        account["shares"] = 0
    else:
        print("Hold")

    print(f"Account balance: ${account['balance']}")
    print(f"Account shares : {account['shares']}")
    print("**Trading Strategy Executed**")

    return account


print("Initializing account and DataFrame")
account, df = initialize(10000)
print(df)


def main():

    while True:
        global account
        global df
        global new_df

        # Fetch and save new data
        # new_df = fetch_data()
        df = df.append(new_df, ignore_index=True)
        min_window = 22
        if df.shape[0] >= min_window:
            signals = generate_signals(df)
            print(signals)
            account = execute_trade_strategy(signals, account)
        time.sleep(1)


initialize(cash=100e3)
loop = asyncio.get_event_loop()
fetch_data_task = loop.create_task(fetch_data())

loop.execute(main())
