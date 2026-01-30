import io
import os
from flask import Flask, render_template, make_response
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
CSV_PATH = os.path.join(os.path.dirname(__file__), 'parking_data.csv')


def read_data():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH)
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            pass
    if 'occupancy_percent' in df.columns:
        df['occupancy_percent'] = pd.to_numeric(df['occupancy_percent'], errors='coerce')
    return df


def fig_to_png_response(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    response = make_response(buf.read())
    response.headers.set('Content-Type', 'image/png')
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/plot/occupancy.png')
def plot_occupancy():
    df = read_data()
    fig, ax = plt.subplots(figsize=(10, 4))
    if df.empty or 'occupancy_percent' not in df.columns:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
    else:
        x = df['timestamp'] if 'timestamp' in df.columns else np.arange(len(df))
        y = df['occupancy_percent']
        ax.plot(x, y, marker='.', linewidth=1)
        ax.set_title('Occupancy % vs Time')
        ax.set_ylabel('Occupancy (%)')
        if 'timestamp' in df.columns:
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            fig.autofmt_xdate()
        ax.grid(alpha=0.3)
    resp = fig_to_png_response(fig)
    plt.close(fig)
    return resp


@app.route('/plot/bar.png')
def plot_bar():
    df = read_data()
    fig, ax = plt.subplots(figsize=(6, 4))
    if df.empty or 'free_slots' not in df.columns:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
    else:
        last = df.iloc[-1]
        free = last['free_slots']
        occ = last['occupied_slots']
        bars = ax.bar(['Free', 'Occupied'], [free, occ], color=['green', 'red'])
        ax.set_title('Current Free vs Occupied')
        for bar in bars:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, y, f'{y:.0f}', ha='center', va='bottom')
    resp = fig_to_png_response(fig)
    plt.close(fig)
    return resp


@app.route('/plot/moving.png')
def plot_moving():
    df = read_data()
    fig, ax = plt.subplots(figsize=(10, 4))
    if df.empty or 'occupancy_percent' not in df.columns:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
    else:
        series = df['occupancy_percent'].astype(float).reset_index(drop=True)
        ma = series.rolling(window=min(30, len(series)), min_periods=1).mean()
        idx = np.arange(len(series))
        ax.plot(idx, series, label='raw', alpha=0.4)
        ax.plot(idx, ma, label='MA (30)', linewidth=2)
        n_fit = min(200, len(series))
        if n_fit >= 2:
            x_fit = np.arange(n_fit)
            y_fit = series.iloc[-n_fit:].values
            coeffs = np.polyfit(x_fit, y_fit, 1)
            poly = np.poly1d(coeffs)
            x_fore = np.arange(len(series), len(series) + 30)
            x_fore_fit = x_fit[-1] + (x_fore - len(series) + 1)
            y_fore = poly(x_fore_fit)
            ax.plot(x_fore, y_fore, '--', color='orange', label='forecast')
        ax.set_title('Moving Average + Forecast')
        ax.set_xlabel('Frame index')
        ax.set_ylabel('Occupancy (%)')
        ax.legend()
        ax.grid(alpha=0.3)
    resp = fig_to_png_response(fig)
    plt.close(fig)
    return resp


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print('Starting web dashboard on http://127.0.0.1:%d' % port)
    app.run(host='0.0.0.0', port=port, debug=False)
