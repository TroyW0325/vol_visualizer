from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
from datetime import datetime

app = Flask(__name__)

def black_scholes_call(S, K, T, r, sigma):
    """Calculate the Black–Scholes price for a call option."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """Calculate the Black–Scholes price for a put option."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility_call(price, S, K, T, r):
    """Invert the Black–Scholes call formula using Brent's method to compute implied volatility."""
    try:
        iv = brentq(lambda sigma: black_scholes_call(S, K, T, r, sigma) - price,
                      1e-6, 5.0, maxiter=500)
        return iv
    except Exception:
        return None

def implied_volatility_put(price, S, K, T, r):
    """Invert the Black–Scholes put formula using Brent's method to compute implied volatility."""
    try:
        iv = brentq(lambda sigma: black_scholes_put(S, K, T, r, sigma) - price,
                      1e-6, 5.0, maxiter=500)
        return iv
    except Exception:
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/vol_surface', methods=['POST'])
def vol_surface():
    ticker = request.form.get('ticker')
    if not ticker:
        return jsonify({'error': 'Ticker symbol is required'}), 400

    # Retrieve ticker data using yfinance
    stock = yf.Ticker(ticker.upper())
    hist = stock.history(period='1d')
    if hist.empty:
        return jsonify({'error': 'Ticker not found or no data available'}), 404

    S = hist['Close'].iloc[-1]
    r = 0.01  # risk-free rate

    # Separate arrays for calls and puts
    calls_strikes, calls_times, calls_ivs = [], [], []
    puts_strikes, puts_times, puts_ivs = [], [], []

    # Loop through available expiration dates
    for exp in stock.options:
        expiration_date = pd.to_datetime(exp)
        # Compute days to expiration (minimum 1 day)
        days = max((expiration_date - pd.Timestamp.now()).days, 1)
        T_years = days / 365.0  # for pricing calculations

        try:
            chain = stock.option_chain(exp)
            calls = chain.calls
            puts = chain.puts
        except Exception:
            continue

        # Process calls
        for _, row in calls.iterrows():
            K = row['strike']
            price = row['lastPrice']
            if price <= 0:
                continue
            iv = implied_volatility_call(price, S, K, T_years, r)
            if iv is not None:
                calls_strikes.append(K)
                calls_times.append(days)
                calls_ivs.append(iv * 100)  # convert to percentage

        # Process puts
        for _, row in puts.iterrows():
            K = row['strike']
            price = row['lastPrice']
            if price <= 0:
                continue
            iv = implied_volatility_put(price, S, K, T_years, r)
            if iv is not None:
                puts_strikes.append(K)
                puts_times.append(days)
                puts_ivs.append(iv * 100)

    if not (calls_strikes or puts_strikes):
        return jsonify({'error': 'Insufficient data to generate surface.'}), 400

    # Use the union of calls and puts for a common grid
    all_strikes = (calls_strikes if calls_strikes else []) + (puts_strikes if puts_strikes else [])
    all_times = (calls_times if calls_times else []) + (puts_times if puts_times else [])
    grid_x = np.linspace(min(all_strikes), max(all_strikes), num=50)
    grid_y = np.linspace(min(all_times), max(all_times), num=50)
    X, Y = np.meshgrid(grid_x, grid_y)

    # Interpolate the smooth IV surface for calls and clip negative values
    if calls_strikes:
        calls_Z = griddata((calls_strikes, calls_times), calls_ivs, (X, Y), method='cubic')
        nan_mask = np.isnan(calls_Z)
        if np.any(nan_mask):
            calls_Z[nan_mask] = griddata((calls_strikes, calls_times), calls_ivs, (X, Y), method='nearest')[nan_mask]
        calls_Z = np.maximum(calls_Z, 0)  # remove negative values
        calls_Z_list = calls_Z.tolist()
    else:
        calls_Z_list = None

    # Interpolate the smooth IV surface for puts and clip negative values
    if puts_strikes:
        puts_Z = griddata((puts_strikes, puts_times), puts_ivs, (X, Y), method='cubic')
        nan_mask = np.isnan(puts_Z)
        if np.any(nan_mask):
            puts_Z[nan_mask] = griddata((puts_strikes, puts_times), puts_ivs, (X, Y), method='nearest')[nan_mask]
        puts_Z = np.maximum(puts_Z, 0)
        puts_Z_list = puts_Z.tolist()
    else:
        puts_Z_list = None

    return jsonify({
        'grid_x': grid_x.tolist(),
        'grid_y': grid_y.tolist(),
        'calls_Z': calls_Z_list,
        'puts_Z': puts_Z_list,
        'calls_strike': calls_strikes,
        'calls_time': calls_times,
        'calls_iv': calls_ivs,
        'puts_strike': puts_strikes,
        'puts_time': puts_times,
        'puts_iv': puts_ivs
    })

if __name__ == '__main__':
    app.run()
