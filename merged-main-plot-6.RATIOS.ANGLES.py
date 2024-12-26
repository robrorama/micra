from data_retrieval import get_stock_data, add_moving_averages
from plot_helpers import (add_open_shape_indicator, plot_signals_with_candlestick_refactored, 
                         plot_intersection_marker, add_anchored_volume_profile)
from signals import (detect_signals, calculate_fibonacci_levels, detect_wick_touches, 
                    detect_fib_wick_touches, detect_body_ma_touches, detect_consecutive_days,
                    calculate_bollinger_bands, add_bollinger_band_markers, detect_volume_price_spikes)
from geometry import (find_two_peaks, find_two_high_peaks, find_two_low_troughs,
                     calculate_intersection, plot_projection_line, calculate_linear_regression_and_deviations)
from summary import generate_summary_output
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math

def create_output_directory(ticker):
    today = datetime.now().strftime('%Y-%m-%d')
    directory = os.path.join('data', today, ticker)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_ratio_dataframe(ticker1, ticker2, date_range_input):
    """Calculate ratio data between two tickers preserving OHLC structure."""
    if ',' in date_range_input:
        start_date, end_date = [d.strip() for d in date_range_input.split(',')]
        df1 = yf.download(ticker1, start=start_date, end=end_date)
        df2 = yf.download(ticker2, start=start_date, end=end_date)
    else:
        df1 = yf.download(ticker1, period=date_range_input.strip())
        df2 = yf.download(ticker2, period=date_range_input.strip())

    ratio_df = pd.DataFrame()
    # Calculate true OHLC ratios
    ratio_df['Open'] = df1['Open'] / df2['Open']
    ratio_df['High'] = df1['High'] / df2['High']
    ratio_df['Low'] = df1['Low'] / df2['Low']
    ratio_df['Close'] = df1['Close'] / df2['Close']
    ratio_df['Volume'] = df1['Volume'] / df2['Volume']
    ratio_df.index = df1.index

    return ratio_df

def main():
    # User input and initialization
    tickers = input("Enter one or two stock tickers (comma-separated if two): ").split(',')
    tickers = [t.strip().upper() for t in tickers]
    output_directory = create_output_directory(tickers[0])
    date_range_input = input("Enter the time period (e.g., '1y') or date range (e.g., '2020-01-02,2021-06-22'): ")

    # Data retrieval
    if len(tickers) == 2:
        print(f"Calculating ratio analysis for {tickers[0]}/{tickers[1]}...")
        df = get_ratio_dataframe(tickers[0], tickers[1], date_range_input)
    else:
        if ',' in date_range_input:
            start_date, end_date = [d.strip() for d in date_range_input.split(',')]
            df = yf.download(tickers[0], start=start_date, end=end_date)
        else:
            df = yf.download(tickers[0], period=date_range_input.strip())

    # Add all technical indicators
    df = add_moving_averages(df)
    df = calculate_bollinger_bands(df)

    # Calculate standard deviation bands
    df['9DMA'] = df['Close'].rolling(window=9).mean()
    df['20DMA'] = df['Close'].rolling(window=20).mean()
    df['50DMA'] = df['Close'].rolling(window=50).mean()
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper_1std'] = df['BB_Middle'] + (df['Close'].rolling(window=20).std() * 1)
    df['BB_Upper_2std'] = df['BB_Middle'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower_1std'] = df['BB_Middle'] - (df['Close'].rolling(window=20).std() * 1)
    df['BB_Lower_2std'] = df['BB_Middle'] - (df['Close'].rolling(window=20).std() * 2)

    # Calculate all signals
    latest_date = df.index[-1].strftime("%Y-%m-%d")
    current_price = df['Close'].iloc[-1]
    buy_signals, sell_signals = detect_signals(df)
    fib_levels, high_price, low_price = calculate_fibonacci_levels(df)
    
    # Calculate regression and deviation bands
    len_regression = 144
    slope, intercept, std_dev, deviations = calculate_linear_regression_and_deviations(df, len_regression)
    wick_touches, touched_devs = detect_wick_touches(df, deviations, len_regression)
    fib_wick_touches, touched_fibs = detect_fib_wick_touches(df, fib_levels)
    ma_touches, touched_mas = detect_body_ma_touches(df)
    sequence_stars = detect_consecutive_days(df)
    spike_days = detect_volume_price_spikes(df)
    
    # Generate signal summary
    generate_summary_output(tickers[0], buy_signals, sell_signals, sequence_stars, 
                          wick_touches, fib_wick_touches, ma_touches, output_directory)

    # Calculate geometry patterns
    high_peaks = find_two_high_peaks(df)
    low_troughs = find_two_low_troughs(df)

    # Create main plot with updated parameters
    fig = plot_signals_with_candlestick_refactored(
        df, buy_signals, sell_signals, fib_levels, wick_touches, fib_wick_touches,
        ma_touches, sequence_stars, slope, intercept, std_dev, tickers[0], deviations, 
        touched_devs, spike_days
    )

    # Add Bollinger Bands
    for band, color, style in [
        ('BB_Middle', 'gray', 'dash'),
        ('BB_Upper_1std', 'green', 'dot'),
        ('BB_Upper_2std', 'lightgreen', 'dot'),
        ('BB_Lower_1std', 'orange', 'dot'),
        ('BB_Lower_2std', 'pink', 'dot')
    ]:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[band],
            mode='lines',
            name=band,
            line=dict(color=color, dash=style)
        ))

    ############
    # Add markers for points beyond 2σ bands
    high_beyond_2std = df[df['High'] > df['BB_Upper_2std']]
    low_below_2std = df[df['Low'] < df['BB_Lower_2std']]

    if not high_beyond_2std.empty:
        fig.add_trace(go.Scatter(
            x=high_beyond_2std.index,
            y=high_beyond_2std['High'],
            mode='markers',
            name='Pierces +2 Sigma',
            marker=dict(symbol='square', size=10, color='lawngreen')))

    if not low_below_2std.empty:
        fig.add_trace(go.Scatter(
            x=low_below_2std.index,
            y=low_below_2std['Low'],
            mode='markers',
            name='Pierces -2 Sigma',
            marker=dict(symbol='square', size=10, color='red')))

    # Add 6-month high/low lines
    six_months_ago = df.index[-1] - pd.DateOffset(months=6)
    last_six_months = df[df.index > six_months_ago]

    highest_above = last_six_months['High'][last_six_months['High'] > last_six_months['BB_Upper_2std']].max()
    lowest_below = last_six_months['Low'][last_six_months['Low'] < last_six_months['BB_Lower_2std']].min()

    if pd.notna(highest_above):
        fig.add_trace(go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[highest_above, highest_above],
            mode='lines',
            name='Highest Above 2σ (6M)',
            line=dict(color='lawngreen', dash='dash', width=1)
        ))

    if pd.notna(lowest_below):
        fig.add_trace(go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[lowest_below, lowest_below],
            mode='lines',
            name='Lowest Below 2σ (6M)',
            line=dict(color='red', dash='dash', width=1)
        ))
    ############

    # Add remaining indicators
    add_open_shape_indicator(fig, spike_days)
    fig = add_bollinger_band_markers(fig, df)

    # Plot projection lines
    slope_high, intercept_high = plot_projection_line(df, fig, high_peaks['High'], 
                                                     color='green', line_name='High Peak Line')
    slope_low, intercept_low = plot_projection_line(df, fig, low_troughs['Low'], 
                                                   color='red', line_name='Low Trough Line')

    # Calculate angles
    high_angle = round(math.degrees(math.atan(slope_high)), 2)
    low_angle = round(math.degrees(math.atan(slope_low)), 2)

    # Calculate and plot intersections
    intersection_date = None
    try:
        date_intersect, y_intersect = calculate_intersection(slope_high, intercept_high, 
                                                           slope_low, intercept_low)
        plot_intersection_marker(fig, date_intersect, y_intersect)
        intersection_date = date_intersect.strftime("%Y-%m-%d")
    except ValueError as e:
        print("Skipping intersection marker:", e)

    # Update title with angles and intersection date
    title_suffix = f" | High Angle: {high_angle}° | Low Angle: {low_angle}°"
    if intersection_date:
        title_suffix += f" | Intersection: {intersection_date}"
    fig.update_layout(title=f"{tickers[0]} Analysis{title_suffix}")

    # Add watermark
    fig.add_annotation(
        text=f"{tickers[0].upper()} \n- {latest_date}\n- ${current_price:.2f}",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=100, color="green"),
        opacity=0.1
    )

    # Show and save plot
    fig.show()
    
    fig_for_saving = fig.full_figure_for_development(warn=False)
    fig_for_saving.update_layout(showlegend=False, xaxis_rangeslider_visible=False)
    
    plot_filename = os.path.join(output_directory, f"{tickers[0]}_{date_range_input.replace(',', '_')}_plot.png")
    pio.write_image(fig_for_saving, plot_filename, format="png")
    print(f"Plot saved as {plot_filename}")

if __name__ == "__main__":
    main()