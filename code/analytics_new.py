import os
import sys

# Attempt to import required libraries
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.dates import DateFormatter
except Exception as e:
    print(f"Missing Python packages for analytics: {e}")
    print("Install with: python -m pip install pandas matplotlib numpy")
    sys.exit(1)


def ensure_plots_dir(path="plots"):
    out = os.path.join(os.path.dirname(__file__), path)
    os.makedirs(out, exist_ok=True)
    return out


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Try to parse timestamp column
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            pass
    # Ensure occupancy_percent is numeric
    if 'occupancy_percent' in df.columns:
        df['occupancy_percent'] = pd.to_numeric(df['occupancy_percent'], errors='coerce')
    return df


def plot_occupancy_time(df, out_dir):
    plt.figure(figsize=(12, 5))
    x = df['timestamp'] if 'timestamp' in df.columns else pd.RangeIndex(len(df))
    y = df['occupancy_percent']
    plt.plot(x, y, marker='.', linewidth=1)
    plt.title('Occupancy % vs Time')
    plt.ylabel('Occupancy (%)')
    plt.xlabel('Time')
    if 'timestamp' in df.columns:
        plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate()
    plt.grid(alpha=0.3)
    out_path = os.path.join(out_dir, 'occupancy_vs_time.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print('Saved', out_path)


def plot_free_vs_occupied(df, out_dir):
    # Use mean counts across dataset as summary
    free_mean = df['free_slots'].mean()
    occ_mean = df['occupied_slots'].mean()
    labels = ['Free (avg)', 'Occupied (avg)']
    values = [free_mean, occ_mean]
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, color=['green', 'red'])
    plt.title('Average Free vs Occupied Slots')
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, y, f'{y:.1f}', ha='center', va='bottom')
    out_path = os.path.join(out_dir, 'free_vs_occupied_avg.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print('Saved', out_path)


def plot_moving_average_and_forecast(df, out_dir, window=30, forecast_steps=30):
    series = df['occupancy_percent'].astype(float).reset_index(drop=True)
    if series.isna().all():
        print('No numeric occupancy_percent data available for moving average.')
        return
    ma = series.rolling(window=window, min_periods=1).mean()

    # Simple linear regression forecast on last N points
    n_fit = min(200, len(series))
    x = np.arange(n_fit)
    y = series.iloc[-n_fit:].values
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    # Predict next forecast_steps
    x_forecast = np.arange(n_fit, n_fit + forecast_steps)
    y_forecast = poly(x_forecast)

    plt.figure(figsize=(12, 5))
    idx = np.arange(len(series))
    plt.plot(idx, series, label='Occupancy % (raw)', alpha=0.4)
    plt.plot(idx, ma, label=f'Moving Average (window={window})', linewidth=2)
    # plot forecast appended after existing index
    plt.plot(x_forecast, y_forecast, label='Linear Forecast', color='orange', linestyle='--')
    plt.title('Moving Average and Simple Forecast of Occupancy %')
    plt.xlabel('Frame index')
    plt.ylabel('Occupancy (%)')
    plt.legend()
    plt.grid(alpha=0.3)
    out_path = os.path.join(out_dir, 'moving_average_forecast.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print('Saved', out_path)


def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'parking_data.csv')
    if not os.path.exists(csv_path):
        print('parking_data.csv not found at', csv_path)
        sys.exit(1)
    df = load_data(csv_path)
    out_dir = ensure_plots_dir('plots')
    # Basic checks
    if 'occupancy_percent' not in df.columns or df['occupancy_percent'].dropna().empty:
        print('No occupancy_percent data to plot.')
        sys.exit(1)

    plot_occupancy_time(df, out_dir)
    plot_free_vs_occupied(df, out_dir)
    plot_moving_average_and_forecast(df, out_dir)


if __name__ == '__main__':
    main()
