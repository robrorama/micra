import plotly.graph_objects as go
import pandas as pd

def plot_bollinger_bands(ax, data):
    ax.plot(data.index, data['Upper Band'], label='Upper Bollinger Band', linestyle='--', alpha=0.7)
    ax.plot(data.index, data['Lower Band'], label='Lower Bollinger Band', linestyle='--', alpha=0.7)

def add_bollinger_band_pierce_markers(fig, df):
    """Add single group of markers for each band pierce type."""
    # Create lists to store pierce points
    upper_pierce_x = []
    upper_pierce_y = []
    lower_pierce_x = []
    lower_pierce_y = []
    
    for i in range(len(df)):
        if df['High'].iloc[i] > df['BB_Upper_2std'].iloc[i]:
            upper_pierce_x.append(df.index[i])
            upper_pierce_y.append(df['High'].iloc[i])
        elif df['Low'].iloc[i] < df['BB_Lower_2std'].iloc[i]:
            lower_pierce_x.append(df.index[i])
            lower_pierce_y.append(df['Low'].iloc[i])

    # Add all pierces as single traces
    if upper_pierce_x:
        fig.add_trace(go.Scatter(
            x=upper_pierce_x,
            y=upper_pierce_y,
            mode='markers',
            marker=dict(symbol='square-open', size=10, color='black'),
            name='Upper Band Pierce'
        ))

    if lower_pierce_x:
        fig.add_trace(go.Scatter(
            x=lower_pierce_x,
            y=lower_pierce_y,
            mode='markers',
            marker=dict(symbol='square-open', size=10, color='black'),
            name='Lower Band Pierce'
        ))
    
    return fig



def add_bollinger_bands_and_markers(fig, df):
    # Add Bollinger Bands
    df['20DMA'] = df['Close'].rolling(window=20).mean()
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper_1std'] = df['BB_Middle'] + (df['Close'].rolling(window=20).std() * 1)
    df['BB_Upper_2std'] = df['BB_Middle'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower_1std'] = df['BB_Middle'] - (df['Close'].rolling(window=20).std() * 1)
    df['BB_Lower_2std'] = df['BB_Middle'] - (df['Close'].rolling(window=20).std() * 2)

    # Add all Bollinger Band traces
    band_configs = [
        ('BB_Middle', 'gray', 'dash', 'BB Middle'),
        ('BB_Upper_1std', 'green', 'dot', 'BB Upper 1σ'),
        ('BB_Upper_2std', 'lightgreen', 'dot', 'BB Upper 2σ'),
        ('BB_Lower_1std', 'orange', 'dot', 'BB Lower 1σ'),
        ('BB_Lower_2std', 'pink', 'dot', 'BB Lower 2σ')
    ]

    for column, color, dash, name in band_configs:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[column],
            mode='lines',
            name=name,
            line=dict(color=color, dash=dash)
        ))

    # Add markers for points beyond 2σ bands
    high_beyond_2std = df[df['High'] > df['BB_Upper_2std']]
    low_below_2std = df[df['Low'] < df['BB_Lower_2std']]

    if not high_beyond_2std.empty:
        fig.add_trace(go.Scatter(
            x=high_beyond_2std.index,
            y=high_beyond_2std['High'],
            mode='markers',
            name='High > 2σ Upper',
            marker=dict(color='lawngreen', size=8, line=dict(color='lawngreen', width=2))
        ))

    if not low_below_2std.empty:
        fig.add_trace(go.Scatter(
            x=low_below_2std.index,
            y=low_below_2std['Low'],
            mode='markers',
            name='Low < 2σ Lower',
            marker=dict(color='red', size=8, line=dict(color='red', width=2))
        ))

    # Add 6-month highest/lowest lines
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

    return fig, df


def add_anchored_volume_profile(fig, df, anchor_price, period=20):
    # Calculate volume profile
    df['PriceBin'] = pd.cut(df['Close'], bins=20)  # Adjust bins as needed
    volume_profile = df.groupby('PriceBin')['Volume'].sum()

    # Create bar trace for volume profile
    fig.add_trace(go.Bar(
        x=volume_profile.index.astype(str),  # Convert intervals to strings for plotting
        y=volume_profile.values,
        yaxis='y2',  # Use a secondary y-axis
        name='Volume Profile',
        opacity=0.5,
        marker=dict(color='lightblue')
    ))

    # Add a vertical line at the anchor price
    fig.add_shape(go.layout.Shape(
        type="line",
        x0=anchor_price,
        x1=anchor_price,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="red", width=2)
    ))

    # Update layout for secondary y-axis
    fig.update_layout(
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        )
    )

    return fig

def add_open_shape_indicator(fig, spike_days):
    spike_dates, spike_prices = zip(*spike_days) if spike_days else ([], [])
    fig.add_trace(go.Scatter(
        x=spike_dates, y=spike_prices, mode='markers',
        marker=dict(symbol='circle-open', size=15, color='black', line=dict(width=2)),  # Customize as needed
        name='Volume & Price Spike'
    ))
    return fig


# plot_helpers.py - add_sequence_stars function
def add_sequence_stars(fig, sequence_stars):
    for color in set([star[3] for star in sequence_stars]):
        star_x = [date for date, price, size, c in sequence_stars if c == color]
        star_y = [price for date, price, size, c in sequence_stars if c == color]
        star_sizes = [size for date, price, size, c in sequence_stars if c == color]  # Extract sizes
        fig.add_trace(go.Scatter(
            x=star_x, y=star_y, mode='markers',
            marker=dict(symbol='star', size=star_sizes, color=color),  # Use star_sizes here
            name=f'{color.capitalize()} Stars'
        ))
    return fig


def plot_intersection_marker(fig, date_intersect, y_intersect):
    """
    Plots concentric circles at the intersection point.
    """
    for i, radius in enumerate([10, 15, 20], start=1):
        fig.add_trace(go.Scatter(
            x=[date_intersect], y=[y_intersect],
            mode="markers", visible="legendonly",
            marker=dict(symbol="circle-open", size=radius, color="red", line=dict()),  # Corrected line
            name=f"Intersection Circle {i}",
        ))

def create_candlestick_chart(df):
    fig = go.Figure()

    # Add OHLC dots first
    for price_type, color, size in [
        ('Open', 'cyan', 2),
        ('Close', 'white', 2),
        ('High', 'green', 2),
        ('Low', 'yellow', 2),
        ('Midpoint', 'orange', 2)  # (High + Low)/2
    ]:
        if price_type == 'Midpoint':
            y_values = (df['High'] + df['Low']) / 2
        else:
            y_values = df[price_type]
            
        fig.add_trace(go.Scatter(
            x=df.index,
            y=y_values,
            mode='markers',
            name=price_type,
            marker=dict(color=color, size=size)
        ))

    # Add candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlesticks'
    ))

    # Fill between 20DMA and 50DMA without creating dots
    for i in range(1, len(df)):
        if pd.notna(df['20DMA'].iloc[i]) and pd.notna(df['50DMA'].iloc[i]):
            color = 'rgba(0, 255, 0, 0.5)' if df['20DMA'].iloc[i] > df['50DMA'].iloc[i] else 'rgba(255, 0, 0, 0.5)'
            fig.add_trace(go.Scatter(
                x=[df.index[i-1], df.index[i], df.index[i], df.index[i-1]],
                y=[df['50DMA'].iloc[i-1], df['50DMA'].iloc[i], df['20DMA'].iloc[i], df['20DMA'].iloc[i-1]],
                fill='toself',
                fillcolor=color,
                mode='none',  # This removes the dots
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Add volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker=dict(
            color=df['Close'] > df['Open'],
            colorscale=[[0, 'red'], [1, 'green']],
            opacity=0.2
        ),
        yaxis='y2'
    ))

    fig.update_layout(
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        barmode='overlay'
    )

    return fig


def create_candlestick_chartAlmostMan(df):
    fig = go.Figure()

    # Add OHLC dots
    for price_type, color, size in [
        ('Open', 'cyan', 2),
        ('Close', 'white', 2),
        ('High', 'green', 2),
        ('Low', 'yellow', 2),
        ('Midpoint', 'orange', 2)  # (High + Low)/2
    ]:
        if price_type == 'Midpoint':
            y_values = (df['High'] + df['Low']) / 2
        else:
            y_values = df[price_type]
            
        fig.add_trace(go.Scatter(
            x=df.index,
            y=y_values,
            mode='markers',
            name=price_type,
            marker=dict(color=color, size=size)
        ))

    # Add candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlesticks'
    ))

    # Fill between 20DMA and 50DMA
    for i in range(1, len(df)):
        if pd.notna(df['20DMA'].iloc[i]) and pd.notna(df['50DMA'].iloc[i]):
            color = 'rgba(0, 255, 0, 0.5)' if df['20DMA'].iloc[i] > df['50DMA'].iloc[i] else 'rgba(255, 0, 0, 0.5)'
            fig.add_trace(go.Scatter(
                x=[df.index[i-1], df.index[i], df.index[i], df.index[i-1]],
                y=[df['50DMA'].iloc[i-1], df['50DMA'].iloc[i], df['20DMA'].iloc[i], df['20DMA'].iloc[i-1]],
                fill='toself',
                fillcolor=color,
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Add volume bars with conditional coloring
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker=dict(
            color=df['Close'] > df['Open'],
            colorscale=[[0, 'red'], [1, 'green']],
            opacity=0.2
        ),
        yaxis='y2'
    ))

    fig.update_layout(
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        barmode='overlay'
    )

    return fig

def create_candlestick_chartOrig(df):
    fig = go.Figure()

    # Add volume trace as a bar chart with conditional coloring
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker=dict(
            color=df['Close'] > df['Open'],  # Condition for red/green
            colorscale=[[0, 'red'], [1, 'green']],  # Colors for False/True
            opacity=0.2  # Adjust opacity for transparency
        ),
        yaxis='y2'
    ))

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    ))

    # Update layout for secondary y-axis
    fig.update_layout(
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False,
        ),
        barmode='overlay'  # Ensure bars are overlaid on the candlestick chart
    )

    return fig


def add_moving_averages(fig, df):
    for ma in ['SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200']:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], mode='lines', name=ma))
    return fig

def add_buy_signals(fig, buy_signals):
    buy_dates, buy_prices = zip(*buy_signals) if buy_signals else ([], [])
    fig.add_trace(go.Scatter(
        x=buy_dates, y=buy_prices, mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
        name='Buy Signals'
    ))
    return fig

def add_sell_signals(fig, sell_signals):
    sell_dates, sell_prices = zip(*sell_signals) if sell_signals else ([], [])
    fig.add_trace(go.Scatter(
        x=sell_dates, y=sell_prices, mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
        name='Sell Signals'
    ))
    return fig

def add_fibonacci_levels(fig, fib_levels, df):
    # Define the typical Fibonacci percentage levels and their colors
    fib_percentages = [23.6, 38.2, 50, 61.8, 78.6]
    fib_colors = ['purple', 'green', 'blue', 'red', 'orange']

    fib_x = [df.index[0], df.index[-1]]  # Start and end dates for plotting horizontal lines

    for i, (level, percentage) in enumerate(zip(fib_levels['Fibonacci Levels'], fib_percentages)):
        rounded_price = round(level)  # Round the price to the nearest integer
        color = fib_colors[i % len(fib_colors)]
        
        # Update the legend text to include the Fibonacci percentage and rounded price
        fig.add_trace(go.Scatter(
            x=fib_x, y=[level, level], mode="lines",
            line=dict(dash="dash", color=color),
            name=f"{percentage}% @ {rounded_price}"  # Legend text with percentage and rounded price
        ))

    return fig



def add_fibonacci_levelsOLDNOPERCENTS(fig, fib_levels, df):  # Add df as an argument
    fib_colors = ['purple', 'green', 'blue', 'red', 'orange', 'cyan', 'magenta']
    fib_x = [df.index[0], df.index[-1]]
    for i, level in enumerate(fib_levels['Fibonacci Levels']):
        color = fib_colors[i % len(fib_colors)]
        fig.add_trace(go.Scatter(
            x=fib_x * 2, y=[level, level] * 2, mode="lines",
            line=dict(dash="dash", color=color), name=f'Fibonacci Level {level}'
        ))
    return fig



def add_linear_regression(fig, slope, intercept, df):
    """Add main regression line to plot."""
    x0 = df.index[0].toordinal()
    x1 = df.index[-1].toordinal()
    y0 = slope * x0 + intercept
    y1 = slope * x1 + intercept
    
    fig.add_trace(go.Scatter(
        x=[df.index[0], df.index[-1]],
        y=[y0, y1],
        mode='lines',
        line=dict(color='black', dash='solid', width=2),
        name='Regression Line'
    ))
    return fig


def add_regression_bands(fig, slope, intercept, std_dev, df):
    """Add parallel deviation bands to plot."""
    x0 = df.index[0].toordinal()
    x1 = df.index[-1].toordinal()
    
    for dev in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
        # Upper band
        y0_upper = slope * x0 + intercept + (dev * std_dev)
        y1_upper = slope * x1 + intercept + (dev * std_dev)
        
        fig.add_trace(go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[y0_upper, y1_upper],
            mode='lines',
            line=dict(color='blue', dash='dot', width=1),
            name=f'Upper {dev}σ Band'
        ))
        
        # Lower band
        y0_lower = slope * x0 + intercept - (dev * std_dev)
        y1_lower = slope * x1 + intercept - (dev * std_dev)
        
        fig.add_trace(go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[y0_lower, y1_lower],
            mode='lines',
            line=dict(color='orange', dash='dot', width=1),
            name=f'Lower {dev}σ Band'
        ))
    return fig



def add_linear_regressionORIG(fig, slope, intercept, df):
    fig.add_trace(go.Scatter(
        x=[df.index[0], df.index[-1]],
        y=[slope * 0 + intercept, slope * (len(df) - 1) + intercept],
        mode='lines', line=dict(color='black', dash='solid'), name='Regression Line'
    ))
    return fig


def add_regression_bandsOrig(fig, slope, intercept, deviation, df):
    # Upper band
    fig.add_trace(go.Scatter(
        x=[df.index[0], df.index[-1]],
        y=[slope * 0 + intercept + deviation, slope * (len(df) - 1) + intercept + deviation],
        mode='lines', line=dict(color='blue', dash='dot'), name='Upper Regression Band'
    ))
    # Lower band
    fig.add_trace(go.Scatter(
        x=[df.index[0], df.index[-1]],
        y=[slope * 0 + intercept - deviation, slope * (len(df) - 1) + intercept - deviation],
        mode='lines', line=dict(color='orange', dash='dot'), name='Lower Regression Band'
    ))
    return fig

def add_deviation_bands(fig, deviations, df, touched_devs):
    for dev_name, dev_prices in deviations.items():
        if dev_name not in touched_devs:
            continue
            
        color = 'blue' if 'upper' in dev_name else 'orange'
        dev_value = float(dev_name.split('_')[1])
        line_name = f"{'Upper' if 'upper' in dev_name else 'Lower'} {dev_value}σ"
        
        fig.add_trace(go.Scatter(
            x=[df.index[-len(dev_prices)], df.index[-1]], 
            y=dev_prices,
            mode='lines',
            line=dict(color=color, dash='dot', width=1),
            name=line_name
        ))
    return fig

def add_deviation_bandsOriginal(fig, deviations, df, touched_devs):
    # Loop through each deviation level and plot only if it's touched by a wick
    for dev_name, dev_prices in deviations.items():
        if dev_name not in touched_devs:
            continue  # Skip bands that were not touched by a wick

        # Determine color and label for the band
        if 'upper' in dev_name:
            color = 'blue'
            line_name = f"Upper {dev_name.split('_')[1]} Sigma"
        elif 'lower' in dev_name:
            color = 'orange'
            line_name = f"Lower {dev_name.split('_')[1]} Sigma"
        else:
            continue

        # Plot each touched deviation band independently
        fig.add_trace(go.Scatter(
            x=df.index[-len(dev_prices):], y=dev_prices, mode='lines',
            line=dict(color=color, dash='dot'), name=line_name
        ))


def add_deviation_bandsSecondCandidate(fig, deviations, df):
    # Loop through each deviation level and plot it as an independent line
    for dev_name, dev_prices in deviations.items():
        if 'upper' in dev_name:
            color = 'blue'
            line_name = f"Upper {dev_name.split('_')[1]} Sigma"
        elif 'lower' in dev_name:
            color = 'orange'
            line_name = f"Lower {dev_name.split('_')[1]} Sigma"
        else:
            continue

        # Plot each deviation level independently for better clarity
        fig.add_trace(go.Scatter(
            x=df.index[-len(dev_prices):], y=dev_prices, mode='lines',
            line=dict(color=color, dash='dot'), name=line_name
        ))



def add_deviation_bandsOriginal(fig, deviations, df):
    upper_x, upper_y = [], []
    lower_x, lower_y = [], []
    for dev_name, dev_prices in deviations.items():
        if 'upper' in dev_name:
            upper_x.extend(df.index[-len(dev_prices):])
            upper_y.extend(dev_prices)
        elif 'lower' in dev_name:
            lower_x.extend(df.index[-len(dev_prices):])
            lower_y.extend(dev_prices)

    fig.add_trace(go.Scatter(
        x=upper_x, y=upper_y, mode='lines', line=dict(color='blue', dash='dot'), name='Upper Deviation Bands'
    ))
    fig.add_trace(go.Scatter(
        x=lower_x, y=lower_y, mode='lines', line=dict(color='orange', dash='dot'), name='Lower Deviation Bands'
    ))
    return fig

def add_wick_touches(fig, wick_touches):
    wick_x = [date for date, (level, price) in wick_touches]
    wick_y = [price for date, (level, price) in wick_touches]
    fig.add_trace(go.Scatter(
        x=wick_x, y=wick_y, mode='markers',
        marker=dict(symbol='x', color='blue', size=8),
        name='Wick Touches'
    ))
    return fig

def add_fib_wick_touches(fig, fib_wick_touches):
    fib_wick_x = [date for date, (level, price) in fib_wick_touches]
    fib_wick_y = [price for date, (level, price) in fib_wick_touches]
    fig.add_trace(go.Scatter(
        x=fib_wick_x, y=fib_wick_y, mode='markers',
        marker=dict(symbol='circle', color='purple', size=6),
        name='Fib Wick Touches'
    ))
    return fig

def add_ma_touches(fig, ma_touches):
    ma_x = [date for date, (ma, price) in ma_touches]
    ma_y = [price for date, (ma, price) in ma_touches]
    fig.add_trace(go.Scatter(
        x=ma_x, y=ma_y, mode='markers',
        marker=dict(symbol='cross', color='black', size=8),
        name='MA Touches'
    ))
    return fig
def finalize_layout(fig, ticker):
    fig.update_layout(
        title=f'{ticker} Candlestick Chart with Interactive Signal Toggles',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(itemsizing='constant'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        spikedistance=1000,
        xaxis=dict(
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            showline=True,
            showgrid=True,
            spikethickness=1,
            spikedash='solid',
            spikecolor='black'
        ),
        yaxis=dict(
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            showline=True,
            showgrid=True,
            spikethickness=1,
            spikedash='solid',
            spikecolor='black',
            side='right'
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family="Courier New"
        )
    )
    return fig



def finalize_layoutORIG(fig, ticker):
    fig.update_layout(
        title=f'{ticker} Candlestick Chart with Interactive Signal Toggles',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(itemsizing='constant'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig


def plot_signals_with_candlestick_refactored(
    df, buy_signals, sell_signals, fib_levels, wick_touches, 
    fib_wick_touches, ma_touches, sequence_stars, slope, 
    intercept, std_dev, ticker, deviations, touched_devs, spike_days
):
    """Create complete chart with all indicators and signals."""
    fig = create_candlestick_chart(df)
    add_moving_averages(fig, df)
    add_buy_signals(fig, buy_signals)
    add_sell_signals(fig, sell_signals)
    add_fibonacci_levels(fig, fib_levels, df)
    add_sequence_stars(fig, sequence_stars)
    add_linear_regression(fig, slope, intercept, df)
    add_regression_bands(fig, slope, intercept, std_dev, df)
    add_open_shape_indicator(fig, spike_days)
    add_wick_touches(fig, wick_touches)
    add_fib_wick_touches(fig, fib_wick_touches)
    add_ma_touches(fig, ma_touches)
    finalize_layout(fig, ticker)
    return fig


def plot_signals_with_candlestick_refactoredOrig(
    df, buy_signals, sell_signals, fib_levels,
    wick_touches, fib_wick_touches, ma_touches,
    sequence_stars, slope, intercept, ticker, deviations, touched_devs,
    spike_days
):
    fig = create_candlestick_chart(df)
    add_moving_averages(fig, df)
    add_buy_signals(fig, buy_signals)
    add_sell_signals(fig, sell_signals)
    add_fibonacci_levels(fig, fib_levels, df)
    add_sequence_stars(fig, sequence_stars)
    add_linear_regression(fig, slope, intercept, df)
    add_open_shape_indicator(fig, spike_days)

    # Ensure touched_devs is passed to add_deviation_bands
    add_deviation_bands(fig, deviations, df, touched_devs)

    add_wick_touches(fig, wick_touches)
    add_fib_wick_touches(fig, fib_wick_touches)
    add_ma_touches(fig, ma_touches)
    finalize_layout(fig, ticker)
    return fig


def plot_signals_with_candlestick_refactoredBroken(
    df, buy_signals, sell_signals, fib_levels,
    wick_touches, fib_wick_touches, ma_touches,
    sequence_stars, slope, intercept, ticker, deviations
):
    fig = create_candlestick_chart(df)
    add_moving_averages(fig, df)
    add_buy_signals(fig, buy_signals)
    add_sell_signals(fig, sell_signals)
    add_fibonacci_levels(fig, fib_levels, df)
    add_sequence_stars(fig, sequence_stars)
    add_linear_regression(fig, slope, intercept, df)

    # Ensure touched_devs is passed to add_deviation_bands
    add_deviation_bands(fig, deviations, df, touched_devs)  # <-- Add touched_devs here

    add_wick_touches(fig, wick_touches)
    add_fib_wick_touches(fig, fib_wick_touches)
    add_ma_touches(fig, ma_touches)
    finalize_layout(fig, ticker)
    return fig



def plot_signals_with_candlestick_refactoredOrig(
    df, buy_signals, sell_signals, fib_levels,
    wick_touches, fib_wick_touches, ma_touches,
    sequence_stars, slope, intercept, ticker, deviations
):
    fig = create_candlestick_chart(df)
    add_moving_averages(fig, df)
    add_buy_signals(fig, buy_signals)
    add_sell_signals(fig, sell_signals)
    add_fibonacci_levels(fig, fib_levels,df)
    add_sequence_stars(fig, sequence_stars)
    add_linear_regression(fig, slope, intercept, df)
    add_deviation_bands(fig, deviations, df)  # Assuming 'deviations' is defined
    add_wick_touches(fig, wick_touches)
    add_fib_wick_touches(fig, fib_wick_touches)
    add_ma_touches(fig, ma_touches)
    finalize_layout(fig, ticker)
    return fig
