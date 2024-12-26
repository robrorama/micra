import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta
from scipy.signal import argrelextrema
from scipy.stats import linregress

def find_two_peaks(df):
    """
    Finds the two highest peaks in the Close price using a rolling window.
    """
    local_maxima = argrelextrema(df['Close'].values, np.greater, order=10)[0]
    peaks = df.iloc[local_maxima].nlargest(2, 'Close')

    # Ensure we have exactly two peaks
    if len(peaks) < 2:
        raise ValueError("Not enough peaks detected in the data.")
    
    return peaks

def find_two_high_peaks(df):
    """
    Finds the two highest peaks based on the High prices.
    """
    local_maxima = argrelextrema(df['High'].values, np.greater, order=10)[0]
    peaks = df.iloc[local_maxima].nlargest(2, 'High')

    # Ensure we have exactly two high peaks
    if len(peaks) < 2:
        raise ValueError("Not enough high peaks detected in the data.")
    
    return peaks

def find_two_low_troughs(df):
    """
    Finds the two lowest troughs based on the Low prices over different time frames.
    """
    six_months_ago = df.index[-1] - timedelta(days=182)
    one_month_ago = df.index[-1] - timedelta(days=30)

    # Filter the data for each date range
    last_six_months_data = df[df.index >= six_months_ago]
    last_month_data = df[df.index >= one_month_ago]

    # Find the lowest point in the last six months
    first_trough = last_six_months_data.loc[last_six_months_data['Low'].idxmin()]

    # Find the lowest point in the last month
    second_trough = last_month_data.loc[last_month_data['Low'].idxmin()]

    # Combine into a DataFrame
    troughs = pd.DataFrame([first_trough, second_trough], index=[first_trough.name, second_trough.name])

    return troughs


def calculate_intersection(slope1, intercept1, slope2, intercept2):
    # Check if slopes are the same (parallel lines)
    if slope1 == slope2:
        raise ValueError("The lines are parallel and do not intersect.")
    
    # Calculate the x-coordinate of the intersection
    x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
    date_intersect = pd.Timestamp.fromordinal(int(x_intersect))
    y_intersect = slope1 * x_intersect + intercept1
    
    return date_intersect, y_intersect


def plot_projection_line(df, fig, points, color='black', line_name='Projection Line', project_until=None):
    """
    Plots a projection line between two points with an optional extension.
    """
    # Extract x and y coordinates of the two points (peaks or troughs)
    point_1, point_2 = points.index[0], points.index[1]
    price_1, price_2 = points.iloc[0], points.iloc[1]

    # Calculate slope and intercept of the line between the two points
    slope = (price_2 - price_1) / (point_2 - point_1).days
    intercept = price_1 - slope * (point_1.toordinal())

    # Generate x values to directly connect the two points
    x_values = [point_1, point_2]
    y_values = [price_1, price_2]

    # Plot initial line segment between the two points
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines',
                             line=dict(color=color, width=2, dash='solid'),
                             name=f'{line_name} (Initial Segment)'))

    # Set the end date for projection based on intersection or one month after the last date
    if project_until is None:
        end_date = df.index[-1] + timedelta(days=30)
    else:
        end_date = project_until

    # Ensure that end_date and point_2 have the same timezone
    if point_2.tzinfo is not None and end_date.tzinfo is None:
        end_date = end_date.tz_localize(point_2.tzinfo)
    elif point_2.tzinfo is None and end_date.tzinfo is not None:
        end_date = end_date.tz_convert(None)

    # Extend the line beyond the second point for projection
    x_proj_values = pd.date_range(start=point_2, end=end_date, freq='D')
    y_proj_values = slope * x_proj_values.map(lambda d: d.toordinal()) + intercept

    # Plot the projection line starting from the second point
    fig.add_trace(go.Scatter(x=[point_2] + list(x_proj_values), y=[price_2] + list(y_proj_values),
                             mode='lines', line=dict(color=color, width=2, dash='dash'),
                             name=f'{line_name} (Projection)'))

    # Plot markers on the two points with larger markers for visibility
    fig.add_trace(go.Scatter(x=points.index, y=points, mode='markers',
                             marker=dict(symbol='circle', size=15, color=color, line=dict(width=2, color='black')),
                             name=f'{line_name} Points'))

    return slope, intercept  # Return slope and intercept for intersection calculation

import numpy as np

import numpy as np

# Modifying the `calculate_linear_regression_and_deviations` function to use 0.25 sigma increments.

# this one calculates them parallel to the linear regression fit 

def calculate_linear_regression_and_deviations(df, length):
    """
    Calculate linear regression and parallel deviation bands using date ordinals.
    
    Args:
        df (pd.DataFrame): DataFrame with datetime index and OHLC data
        length (int): Number of periods to use for regression
        
    Returns:
        tuple: (slope, intercept, std_dev, deviations)
    """
    # Convert dates to ordinals for proper regression
    x = np.array([d.toordinal() for d in df.index[-length:]])
    y = df['Close'].values[-length:]
    
    # Calculate regression
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Calculate regression line and residuals
    base_line = slope * x + intercept
    residuals = y - base_line
    std_dev = np.std(residuals)
    
    # Calculate parallel deviation bands
    deviations = {}
    for dev in [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
        deviations[f'upper_{dev}'] = base_line + dev * std_dev
        deviations[f'lower_{dev}'] = base_line - dev * std_dev

    return slope, intercept, std_dev, deviations



def calculate_linear_regression_and_deviationsOLDER(df, length):
    x = np.arange(len(df))
    y = df['Close'].values
    
    x_recent = x[-length:]
    y_recent = y[-length:]
    
    A = np.vstack([x_recent, np.ones(length)]).T
    slope, intercept = np.linalg.lstsq(A, y_recent, rcond=None)[0]
    
    # Calculate base regression line
    base_line = slope * x_recent + intercept
    residuals = y_recent - base_line
    std_dev = np.std(residuals)
    
    # Create parallel bands by keeping the same slope
    deviations = {}
    for dev in [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
        deviations[f'upper_{dev}'] = slope * x_recent + (intercept + dev * std_dev)
        deviations[f'lower_{dev}'] = slope * x_recent + (intercept - dev * std_dev)

    return slope, intercept, base_line[0], base_line[-1], deviations

def calculate_linear_regression_and_deviationsSmoothButHorizontal(df, length):
    x = np.arange(len(df))
    y = df['Close'].values
    
    x_recent = x[-length:]
    y_recent = y[-length:]
    
    A = np.vstack([x_recent, np.ones(length)]).T
    slope, intercept = np.linalg.lstsq(A, y_recent, rcond=None)[0]
    
    base_line = slope * x_recent + intercept
    residuals = y_recent - base_line
    std_dev = np.std(residuals)
    
    deviations = {}
    for dev in [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
        deviations[f'upper_{dev}'] = base_line + dev * std_dev
        deviations[f'lower_{dev}'] = base_line - dev * std_dev

    return slope, intercept, base_line[0], base_line[-1], deviations

# reverting back broken ...
def calculate_linear_regression_and_deviationsOLD(df, length):
    x = np.arange(len(df))
    y = df['Close'].values
    
    # Recent data slice for regression
    x_recent = x[-length:]
    y_recent = y[-length:]
    
    # Matrix-based regression
    A = np.vstack([x_recent, np.ones(length)]).T
    slope, intercept = np.linalg.lstsq(A, y_recent, rcond=None)[0]
    
    # Generate start and end points only
    x_points = np.array([x_recent[0], x_recent[-1]])
    base_line = slope * x_points + intercept
    
    # Calculate residuals and std for bands
    residuals = y_recent - (slope * x_recent + intercept)
    std_dev = np.std(residuals)
    
    # Create two-point deviation bands
    deviations = {}
    for dev in [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
        deviations[f'upper_{dev}'] = base_line + dev * std_dev
        deviations[f'lower_{dev}'] = base_line - dev * std_dev

    return slope, intercept, base_line[0], base_line[-1], deviations




# this calculates the bands at each step ... 
def calculate_linear_regression_and_deviationsB(df, length):
    x = np.arange(len(df))
    y = df['Close'].values  # Directly using Close prices without log scale

    # Perform linear regression on price
    A = np.vstack([x[-length:], np.ones(length)]).T
    slope, intercept = np.linalg.lstsq(A, y[-length:], rcond=None)[0]

    # Calculate regression line (start and end points) in price scale
    start_price = slope * x[-length] + intercept
    end_price = slope * x[-1] + intercept

    # Calculate standard deviation of residuals
    residuals = y[-length:] - (slope * x[-length:] + intercept)
    std_dev = np.std(residuals)

    # Create deviation bands at 0.25 sigma intervals
    deviations = {}
    for dev in [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
        deviations[f'upper_{dev}'] = (slope * x[-length:] + intercept + dev * std_dev)
        deviations[f'lower_{dev}'] = (slope * x[-length:] + intercept - dev * std_dev)

    return slope, intercept, start_price, end_price, deviations

# Replace the original function with this modified version in the relevant part of the script.
# This would ensure all standard deviation bands are calculated as offsets from the regression line at 0.25 sigma intervals.
#calculate_linear_regression_and_deviations_modified


def calculate_linear_regression_and_deviationsOrig(df, length):
    x = np.arange(len(df))
    y = df['Close'].values  # Directly using Close prices without log scale

    # Perform linear regression on price
    A = np.vstack([x[-length:], np.ones(length)]).T
    slope, intercept = np.linalg.lstsq(A, y[-length:], rcond=None)[0]

    # Calculate regression line (start and end points) in price scale
    start_price = slope * x[-length] + intercept
    end_price = slope * x[-1] + intercept

    # Calculate standard deviation of residuals
    residuals = y[-length:] - (slope * x[-length:] + intercept)
    std_dev = np.std(residuals)

    # Create deviation bands
    deviations = {}
    for dev in [0, 0.5, 1, 1.5, 2]:
        deviations[f'upper_{dev}'] = (slope * x[-length:] + intercept + dev * std_dev)
        deviations[f'lower_{dev}'] = (slope * x[-length:] + intercept - dev * std_dev)

    return slope, intercept, start_price, end_price, deviations




