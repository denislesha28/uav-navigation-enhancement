import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import signal, stats
from scipy.stats import normaltest
from statsmodels.tsa.stattools import acf, adfuller

def analyze_sensor_correlations(aligned_dict):
    """
    Analyze correlations between IMU and Gyro
    """
    imu_df = aligned_dict['IMU_df']
    gyro_df = aligned_dict['RawGyro_df']
    correlations = {
        'IMU_Gyro': {},
        'raw_values': {}
    }

    if imu_df.empty or gyro_df.empty:
        return correlations

    for axis in ['x', 'y', 'z']:
        if axis not in imu_df.columns or axis not in gyro_df.columns:
            continue

        valid_mask = (~imu_df[axis].isna()) & (~gyro_df[axis].isna())
        imu_data = imu_df[axis][valid_mask].values
        gyro_data = gyro_df[axis][valid_mask].values

        if len(imu_data) == 0 or len(gyro_data) == 0:
            continue

        corr = signal.correlate(imu_data, gyro_data, mode='full')
        lags = signal.correlation_lags(len(imu_data), len(gyro_data), mode='full')

        correlations['raw_values'][axis] = {
            'correlation': corr,
            'lags': lags,
            'peak_correlation': np.max(np.abs(corr)),
            'peak_lag': lags[np.argmax(np.abs(corr))]
        }

        if np.std(imu_data) > 0 and np.std(gyro_data) > 0:
            correlations['IMU_Gyro'][axis] = np.corrcoef(imu_data, gyro_data)[0, 1]
        else:
            correlations['IMU_Gyro'][axis] = np.nan

    return correlations


def plot_correlations(correlations):
    """
    Plot correlation results between IMU and Gyro.
    """
    if 'raw_values' not in correlations:
        print("Error: No correlation data to plot")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    for idx, axis in enumerate(['x', 'y', 'z']):
        corr_data = correlations['raw_values'].get(axis)
        if corr_data is None:
            axes[idx].set_title(f"No data for {axis} axis")
            continue

        lags = corr_data['lags']
        corr = corr_data['correlation']
        peak_corr = corr_data['peak_correlation']
        peak_lag = corr_data['peak_lag']

        axes[idx].plot(lags, corr)
        axes[idx].axvline(x=peak_lag, color='r', linestyle='--')
        axes[idx].set_title(
            f'{axis}-axis correlation between IMU and Gyro \nPeak={peak_corr:.3f} at lag={peak_lag}'
        )
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()


def compare_x_axis_movement(aligned_dict):
    imu_df = aligned_dict['IMU_df']
    gyro_df = aligned_dict['RawGyro_df']

    window = slice(0, 100)

    plt.figure(figsize=(15, 5))
    plt.plot(imu_df['x'][window], label='IMU x-axis', marker='o')
    plt.plot(gyro_df['x'][window], label='Gyro x-axis', marker='x')
    plt.legend()
    plt.grid(True)
    plt.title('X-axis Movement Comparison')
    plt.show()


def analyze_distributions(aligned_dict):
    """Analyze sensor data distributions"""
    for sensor_name, df in aligned_dict.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if sensor_name == 'Board_gps_df':
            df_columns = ['vel_n_m_s', 'vel_e_m_s', 'vel_e_m_s']
        else:
            df_columns = ['x', 'y', 'z']

        for i, col in enumerate(df_columns):
            if col in df.columns:
                # Histogram
                axes[i].hist(df[col], bins=50, density=True, alpha=0.7)
                # Add KDE plot
                sns.kdeplot(data=df[col], ax=axes[i], color='red')
                axes[i].set_title(f'{sensor_name} {col}-axis distribution')

                # Add statistical metrics in bottom left
                stats_text = f'Mean: {df[col].mean():.3f}\n'
                stats_text += f'Std: {df[col].std():.3f}\n'
                stats_text += f'Skew: {df[col].skew():.3f}'
                axes[i].text(0.05, 0.05, stats_text, transform=axes[i].transAxes,
                             bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

def analyze_noise(aligned_dict):
    """Analyze sensor noise patterns with GPS handling"""
    for sensor_name, df in aligned_dict.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Select appropriate column based on sensor type
        if sensor_name == 'Board_gps_df':
            analysis_col = 'vel_n_m_s'  # Use north velocity component
        else:
            analysis_col = 'x'

        # Time domain noise
        axes[0, 0].plot(df.index[:1000], df[analysis_col][:1000])
        axes[0, 0].set_title(f'{sensor_name} Time Domain Noise ({analysis_col})')

        # Power spectral density
        freqs, psd = signal.welch(df[analysis_col].dropna())
        axes[0, 1].semilogy(freqs, psd)
        axes[0, 1].set_title('Power Spectral Density')

        # Autocorrelation
        lag_acf = acf(df[analysis_col].dropna(), nlags=100)
        axes[1, 0].plot(lag_acf)
        axes[1, 0].set_title('Autocorrelation')

        # Q-Q plot
        stats.probplot(df[analysis_col].dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')

        plt.suptitle(f'{sensor_name} Noise Analysis', y=1.02)
        plt.tight_layout()
        plt.show()


def validate_statistics(aligned_dict):
    """Comprehensive statistical validation with proper GPS handling"""
    stats_report = {}

    for sensor_name, df in aligned_dict.items():
        # Select appropriate columns based on sensor type
        if sensor_name == 'Board_gps_df':
            cols = ['vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s']
        else:
            cols = ['x', 'y', 'z']

        # Create filtered dataframe with numeric columns only
        df_numeric = df[cols].copy()

        stats = {
            'stationarity': {
                col: adfuller(df_numeric[col].dropna())[1]
                for col in cols
            },
            'normality': {
                col: normaltest(df_numeric[col].dropna())[1]
                for col in cols
            },
            'correlation_matrix': df_numeric.corr(),
            'basic_stats': df_numeric.describe()
        }
        stats_report[sensor_name] = stats

        print(f"\n{sensor_name} Statistics:")
        print("\nCorrelation Matrix:")
        print(stats['correlation_matrix'])
        print("\nBasic Statistics:")
        print(stats['basic_stats'])

    return stats_report
