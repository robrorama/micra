import pandas as pd
import os
from datetime import datetime

def generate_summary_output(ticker, buy_signals, sell_signals, sequence_stars, wick_touches, fib_wick_touches, ma_touches, output_directory):
    summary = []

    for date, price in buy_signals:
        summary.append({"Date": date, "Signal": "Buy", "Price": price})
    for date, price in sell_signals:
        summary.append({"Date": date, "Signal": "Sell", "Price": price})
    for date, price, size, color in sequence_stars:
        summary.append({"Date": date, "Signal": f"{color.capitalize()} Star", "Price": price, "Size": size})
    for date, (level, price) in wick_touches:
        summary.append({"Date": date, "Signal": "Wick Touch", "Level": level, "Price": price})
    for date, (level, price) in fib_wick_touches:
        summary.append({"Date": date, "Signal": "Fib Wick Touch", "Level": level, "Price": price})
    for date, (ma, price) in ma_touches:
        summary.append({"Date": date, "Signal": "MA Touch", "MA": ma, "Price": price})

    summary_df = pd.DataFrame(summary)

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    filename = os.path.join(output_directory, f"{ticker}_detailed_signal_summary.csv")
    summary_df.to_csv(filename, index=False)
    print(f"Detailed summary saved to {filename}")

def generate_summary_outputOLD(ticker, buy_signals, sell_signals, sequence_stars, wick_touches, fib_wick_touches, ma_touches, output_directory):
    summary = []
    
    # Compile summary data for each type of signal
    for date, price in buy_signals:
        summary.append({"Date": date, "Type": "Buy Signal", "Price": price})
    for date, price in sell_signals:
        summary.append({"Date": date, "Type": "Sell Signal", "Price": price})
    for date, price, size, color in sequence_stars:
        summary.append({"Date": date, "Type": f"{color.capitalize()} Star", "Price": price})
    for date, (level, price) in wick_touches:
        summary.append({"Date": date, "Type": "Wick Touch", "Level": level, "Price": price})
    for date, (level, price) in fib_wick_touches:
        summary.append({"Date": date, "Type": "Fib Wick Touch", "Level": level, "Price": price})
    for date, (ma, price) in ma_touches:
        summary.append({"Date": date, "Type": "MA Touch", "MA": ma, "Price": price})

    # Convert to DataFrame and save as CSV
    summary_df = pd.DataFrame(summary)
    #filename = f"{ticker}_detailed_signal_summary.csv"
    filename = os.path.join(output_directory, f"{ticker}_detailed_signals_summary.csv")
    summary_df.to_csv(filename, index=False)
    print(f"Detailed summary saved to {filename}")

