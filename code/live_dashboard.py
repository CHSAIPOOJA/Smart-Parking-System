import os
import sys
import argparse
import time

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.dates import DateFormatter
    from matplotlib.animation import FuncAnimation
except Exception as e:
    print(f"Missing packages: {e}")
    print("Install with: python -m pip install pandas matplotlib numpy")
    sys.exit(1)


def read_data(csv_path):
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            pass
    if 'occupancy_percent' in df.columns:
        df['occupancy_percent'] = pd.to_numeric(df['occupancy_percent'], errors='coerce')
    return df


def ensure_plots_dir():
    out = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(out, exist_ok=True)
    return out


class LiveDashboard:
    def __init__(self, csv_path, interval=1000, window=300):
        self.csv_path = csv_path
        self.interval = interval
        self.window = window
        self.plots_dir = ensure_plots_dir()

        # Setup figure with 3 subplots
        self.fig, (self.ax_time, self.ax_bar, self.ax_ma) = plt.subplots(3, 1, figsize=(10, 10))
        plt.tight_layout()

        # Lines/placeholders
        self.line_time, = self.ax_time.plot([], [], marker='.', linestyle='-')
        self.bar_rects = None
        self.line_ma, = self.ax_ma.plot([], [], linewidth=2)
        self.line_raw, = self.ax_ma.plot([], [], alpha=0.4)
        self.ax_time.set_title('Occupancy % vs Time')
        self.ax_time.set_ylabel('Occupancy (%)')
        self.ax_ma.set_title('Moving Average + Forecast')
        self.ax_ma.set_xlabel('Frame Index')

    def update(self, frame=None):
        df = read_data(self.csv_path)
        if df.empty:
            return

        # Occupancy vs time
        x = df['timestamp'] if 'timestamp' in df.columns else np.arange(len(df))
        y = df['occupancy_percent']
        self.ax_time.cla()
        self.ax_time.plot(x, y, marker='.', linewidth=1)
        self.ax_time.set_title('Occupancy % vs Time')
        self.ax_time.set_ylabel('Occupancy (%)')
        if 'timestamp' in df.columns:
            self.ax_time.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            self.fig.autofmt_xdate()
        self.ax_time.grid(alpha=0.3)

        # Bar chart: latest free vs occupied (use last row)
        last = df.iloc[-1]
        free = last['free_slots'] if 'free_slots' in df.columns else np.nan
        occupied = last['occupied_slots'] if 'occupied_slots' in df.columns else np.nan
        self.ax_bar.cla()
        self.ax_bar.bar(['Free', 'Occupied'], [free, occupied], color=['green', 'red'])
        self.ax_bar.set_title('Current Free vs Occupied')
        for i, v in enumerate([free, occupied]):
            self.ax_bar.text(i, v, f'{v:.0f}', ha='center', va='bottom')

        # Moving average and simple forecast
        series = df['occupancy_percent'].astype(float).reset_index(drop=True)
        ma = series.rolling(window=min(self.window, len(series)), min_periods=1).mean()
        self.ax_ma.cla()
        idx = np.arange(len(series))
        self.ax_ma.plot(idx, series, label='raw', alpha=0.4)
        self.ax_ma.plot(idx, ma, label=f'MA (window={min(self.window, len(series))})')

        # Linear forecast from last up-to-200 samples
        n_fit = min(200, len(series))
        if n_fit >= 2:
            x_fit = np.arange(n_fit)
            y_fit = series.iloc[-n_fit:].values
            coeffs = np.polyfit(x_fit, y_fit, 1)
            poly = np.poly1d(coeffs)
            x_fore = np.arange(len(series), len(series) + int(self.interval/1000 * 5))
            # scale x_fore to fit linear fit domain
            x_fore_fit = x_fit[-1] + (x_fore - len(series) + 1)
            y_fore = poly(x_fore_fit)
            self.ax_ma.plot(x_fore, y_fore, '--', color='orange', label='forecast')

        self.ax_ma.set_title('Moving Average + Forecast')
        self.ax_ma.set_xlabel('Frame Index')
        self.ax_ma.set_ylabel('Occupancy (%)')
        self.ax_ma.legend()
        self.ax_ma.grid(alpha=0.3)

        # Save snapshot of plots (optional)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.fig.savefig(os.path.join(self.plots_dir, f'live_snapshot_{timestamp}.png'))

    def run(self):
        anim = FuncAnimation(self.fig, self.update, interval=self.interval)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Live parking analytics dashboard')
    parser.add_argument('--csv', default=os.path.join(os.path.dirname(__file__), 'parking_data.csv'))
    parser.add_argument('--interval', type=int, default=1000, help='Update interval in ms')
    parser.add_argument('--window', type=int, default=30, help='Moving average window (frames)')
    parser.add_argument('--test', action='store_true', help='Run one update and exit (save snapshot)')
    args = parser.parse_args()

    dash = LiveDashboard(args.csv, interval=args.interval, window=args.window)
    if args.test:
        dash.update()
        print('Saved test snapshot(s) to plots/')
        return
    dash.run()


if __name__ == '__main__':
    main()
