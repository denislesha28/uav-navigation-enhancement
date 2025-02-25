import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal, stats
from scipy.stats import normaltest
from statsmodels.tsa.stattools import acf, adfuller

from training.cnn_feature_extraction import CNNFeatureExtractor, validate_cnn_features, visualize_feature_maps
from data_loader import load_data
from preprocessing.gps_downsampler import staged_gps_downsample
from training.lstm.bi_lstm_training import train_lstm
from preprocessing.sliding_windows import create_sliding_windows, validate_windows
from uav_navigation_kf import UAVNavigationKF, quaternion_to_euler



def resample_ground_truth(ground_truth_df, target_samples=27137):
    """
    Resample ground truth to exact number of samples using scipy.signal.resample

    Args:
        ground_truth_df: Ground truth DataFrame
        target_samples: Desired number of samples (default matches IMU)
    """
    from scipy import signal

    # Get original index for time scaling
    old_index = ground_truth_df.index

    # Resample each column
    resampled_data = {}
    for col in ground_truth_df.columns:
        resampled_data[col] = signal.resample(ground_truth_df[col].values, target_samples)

    # Create new index with same time span
    new_index = pd.timedelta_range(
        start=old_index[0],
        end=old_index[-1],
        periods=target_samples
    )

    # Create new DataFrame
    resampled_df = pd.DataFrame(resampled_data, index=new_index)

    # Print metrics
    print(f"\nGround truth resampling metrics:")
    print(f"Original samples: {len(ground_truth_df)}")
    print(f"Resampled samples: {len(resampled_df)}")

    return resampled_df

def align_sensors_multi_freq(df_dict, target_freq='100ms'):
    """
    Align sensors with possibly different sampling rates.
    """

    # 1. Find common start/end among all dataframes
    common_start = max(df.index.min() for df in df_dict.values() if not df.empty)
    common_end = min(df.index.max() for df in df_dict.values() if not df.empty)

    # 2. Round start & end to multiples of target_freq to avoid float issues
    #    Convert target_freq to a Timedelta
    freq_td = pd.to_timedelta(target_freq)

    #    Snap common_start DOWN to nearest multiple of freq_td
    #    Snap common_end   UP   to nearest multiple of freq_td
    #    (only if you'd like to ensure the entire range is covered)
    def round_down_to_freq(t, freq_td):
        # E.g. floor division in integer microseconds
        n_us = t // pd.Timedelta('1us')  # total microseconds
        freq_us = freq_td // pd.Timedelta('1us')  # freq in microseconds
        rounded_us = (n_us // freq_us) * freq_us
        return pd.Timedelta(rounded_us, unit='us')

    def round_up_to_freq(t, freq_td):
        n_us = t // pd.Timedelta('1us')
        freq_us = freq_td // pd.Timedelta('1us')
        # if there's a remainder, add one chunk
        if (n_us % freq_us) != 0:
            rounded_us = ((n_us // freq_us) + 1) * freq_us
        else:
            rounded_us = n_us
        return pd.Timedelta(rounded_us, unit='us')

    common_start_rounded = round_down_to_freq(common_start, freq_td)
    common_end_rounded = round_up_to_freq(common_end, freq_td)

    # 3. Create target_timestamps using these rounded times
    target_timestamps = pd.timedelta_range(
        start=common_start_rounded,
        end=common_end_rounded,
        freq=target_freq
    )

    # 4. Align each DF to the target_timestamps
    aligned_dict = {}
    for sensor_name, df in df_dict.items():
        if df.empty:
            aligned_dict[sensor_name] = df
            continue

        df_trimmed = df.loc[common_start:common_end].copy()

        if sensor_name == 'Board_gps_df':
            df_aligned = staged_gps_downsample(df_trimmed, target_timestamps, 10)

        elif sensor_name == "Ground_truth_df":
            df_aligned = resample_ground_truth(
                ground_truth_df=df_trimmed,
                target_samples=len(df_dict['IMU_df'])
            )
            # Use linear interpolation and ffill/bfill
            df_aligned = df_aligned.reindex(target_timestamps).interpolate(method='linear')
            df_aligned = df_aligned.ffill().bfill()

        else:
            #Keep using nearest for other sensors
            df_aligned = df_trimmed.reindex(
                target_timestamps,
                method='nearest',
                tolerance=pd.Timedelta('50ms')
            )

        aligned_dict[sensor_name] = df_aligned

        # Print your metrics as before...
        original_samples = len(df_trimmed)
        aligned_samples = len(df_aligned)
        unique_samples = df_aligned.drop_duplicates().shape[0]
        print(f"\n{sensor_name} alignment metrics:")
        print(f"Original samples: {original_samples}")
        print(f"Aligned samples:  {aligned_samples}")
        print(f"Unique samples:   {unique_samples}")
        print(f"Duplicate ratio:  {(aligned_samples - unique_samples) / aligned_samples:.2%}")

    return aligned_dict


def validate_alignment(aligned_dict, target_freq='100ms'):
    """
    Validate alignment results.
    """
    print("\nAlignment Validation:")
    for sensor_name, df in aligned_dict.items():
        if df.empty or len(df) < 2:
            continue

        time_diffs = df.index.to_series().diff()
        expected_diff = pd.Timedelta(target_freq)

        print(f"\n{sensor_name} temporal statistics:")
        print(f"Mean interval: {time_diffs.mean()}")
        print(f"Std interval: {time_diffs.std()}")
        print(f"Max interval: {time_diffs.max()}")

        # Identify any large gaps
        gaps = time_diffs[time_diffs > expected_diff * 1.5]
        if not gaps.empty:
            print(f"Found {len(gaps)} significant gaps")
            print(f"Largest gap: {gaps.max()}")


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
            f'{axis}-axis correlation\nPeak={peak_corr:.3f} at lag={peak_lag}'
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


def validate_ground_truth(gt_df):
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # 1. Quaternion magnitude (should be ~1)
    quat_mag = np.sqrt(gt_df['Attitude_w'] ** 2 + gt_df['Attitude_x'] ** 2 +
                       gt_df['Attitude_y'] ** 2 + gt_df['Attitude_z'] ** 2)
    axes[0].plot(quat_mag)
    axes[0].set_title('Quaternion Magnitude (Should be ~1)')
    axes[0].axhline(y=1, color='r', linestyle='--')

    # 2. Angular rates vs accelerations
    axes[1].plot(gt_df['Omega_x'], label='ωx')
    axes[1].plot(gt_df['Accel_x'], label='ax')
    axes[1].set_title('Angular Rate vs Acceleration (X-axis)')
    axes[1].legend()

    # 3. Velocity continuity
    axes[2].plot(gt_df['Vel_x'], label='Vx')
    axes[2].plot(gt_df['Vel_y'], label='Vy')
    axes[2].plot(gt_df['Vel_z'], label='Vz')
    axes[2].set_title('Velocity Components')
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def calc_error_state_vector(aligned_data: dict):
    imu_data = aligned_data['IMU_df']
    gps_data = aligned_data['Board_gps_df']
    ground_truth = aligned_data['Ground_truth_df']
    timestamp = ground_truth.index[0]
    print(gps_data.head())
    # Initialize KF
    kf = UAVNavigationKF(dt=0.1)  # 10Hz sampling rate

    # Initial state from first ground truth entry
    initial_state = np.array([
        ground_truth.iloc[0]['Accel_x'],  # px
        ground_truth.iloc[0]['Accel_y'],  # py
        ground_truth.iloc[0]['Accel_z'],  # pz
        ground_truth.iloc[0]['Vel_x'],  # vx
        ground_truth.iloc[0]['Vel_y'],  # vy
        ground_truth.iloc[0]['Vel_z'],  # vz
        # Convert quaternion to euler for roll, pitch, yaw
        *quaternion_to_euler(ground_truth.iloc[0][['Attitude_w', 'Attitude_x', 'Attitude_y', 'Attitude_z']])
    ])

    kf.initialize(initial_state)

    error_states = []

    # For each timestamp in your aligned data:
    for idx, row in aligned_data["IMU_df"].iterrows():
        kf.predict()

        # Create measurement vector from GPS data
        measurement = np.array([
            # IMU accelerations
            imu_data.loc[idx]['x'],
            imu_data.loc[idx]['y'],
            imu_data.loc[idx]['z'],

            # GPS velocities
            gps_data.loc[idx]['vel_n_m_s'],
            gps_data.loc[idx]['vel_e_m_s'],
            gps_data.loc[idx]['vel_d_m_s']
        ])

        # Update KF with measurement
        kf.update(measurement)

        # Get ground truth state for this timestamp
        true_state = np.array([
            ground_truth.loc[idx]['Accel_x'],
            ground_truth.loc[idx]['Accel_y'],
            ground_truth.loc[idx]['Accel_z'],
            ground_truth.loc[idx]['Vel_x'],
            ground_truth.loc[idx]['Vel_y'],
            ground_truth.loc[idx]['Vel_z'],
            *quaternion_to_euler(ground_truth.loc[idx][['Attitude_w', 'Attitude_x', 'Attitude_y', 'Attitude_z']])
        ])

        # Calculate error state using ground truth
        error_state = kf.calculate_error_state(true_state)
        error_states.append((idx, error_state))

    return error_states


def plot_error_states(error_states, timestamps, save_path=None):
    """
    Plot error states with proper grouping and formatting

    Args:
        error_states_array: numpy array of shape (n_samples, 15)
        timestamps: array of timestamps
        save_path: optional path to save the figure
    """
    error_states_array = np.array([value for _, value in error_states])
    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)

    # Attitude errors (φE, φN, φU)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(timestamps, error_states_array[:, 0], label='Roll (φE)')
    ax1.plot(timestamps, error_states_array[:, 1], label='Pitch (φN)')
    ax1.plot(timestamps, error_states_array[:, 2], label='Yaw (φU)')
    ax1.set_title('Attitude Errors')
    ax1.set_ylabel('Radians')
    ax1.legend()
    ax1.grid(True)

    # Velocity errors (δυE, δυN, δυU)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(timestamps, error_states_array[:, 3], label='East')
    ax2.plot(timestamps, error_states_array[:, 4], label='North')
    ax2.plot(timestamps, error_states_array[:, 5], label='Up')
    ax2.set_title('Velocity Errors')
    ax2.set_ylabel('m/s')
    ax2.legend()
    ax2.grid(True)

    # Position errors (δL, δλ, δh)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(timestamps, error_states_array[:, 6], label='Latitude')
    ax3.plot(timestamps, error_states_array[:, 7], label='Longitude')
    ax3.plot(timestamps, error_states_array[:, 8], label='Height')
    ax3.set_title('Position Errors')
    ax3.set_ylabel('meters')
    ax3.legend()
    ax3.grid(True)

    # Gyro bias errors (εx, εy, εz)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(timestamps, error_states_array[:, 9], label='X-axis')
    ax4.plot(timestamps, error_states_array[:, 10], label='Y-axis')
    ax4.plot(timestamps, error_states_array[:, 11], label='Z-axis')
    ax4.set_title('Gyroscope Bias Errors')
    ax4.set_ylabel('rad/s')
    ax4.legend()
    ax4.grid(True)

    # Accelerometer bias errors (∇x, ∇y, ∇z)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(timestamps, error_states_array[:, 12], label='X-axis')
    ax5.plot(timestamps, error_states_array[:, 13], label='Y-axis')
    ax5.plot(timestamps, error_states_array[:, 14], label='Z-axis')
    ax5.set_title('Accelerometer Bias Errors')
    ax5.set_ylabel('m/s²')
    ax5.legend()
    ax5.grid(True)

    # Error statistics
    ax6 = fig.add_subplot(gs[2, 1])
    error_means = np.mean(np.abs(error_states_array), axis=0)
    error_stds = np.std(error_states_array, axis=0)
    labels = ['φE', 'φN', 'φU', 'δυE', 'δυN', 'δυU',
              'δL', 'δλ', 'δh', 'εx', 'εy', 'εz',
              '∇x', '∇y', '∇z']
    ax6.bar(labels, error_means, yerr=error_stds, capsize=5)
    ax6.set_title('Error Statistics')
    ax6.set_ylabel('Mean Absolute Error')
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1) Load data via IMUDataProcessor and other logs
    df_data = load_data()

    # 2) Align multi-frequency data
    aligned_data = align_sensors_multi_freq(df_data)
    print("IMU Timestamps (first 5):", aligned_data['IMU_df'].index[:5])
    print("Ground Truth Timestamps (first 5):", aligned_data['Ground_truth_df'].index[:5])
    print("GPS Timestamps (first 5):", aligned_data['Board_gps_df'].index[:5])  # Also check GPS

    error_states = calc_error_state_vector(aligned_data)

    plot_error_states(error_states, aligned_data["Ground_truth_df"].index)

    # plot_error_states(error_states, aligned_data.idx)
    # ground_truth_interpolated = aligned_data.get("Ground_truth_df")
    aligned_data.pop("Ground_truth_df")

    # 3) Print the aligned data for inspection
    print(aligned_data)

    # 4) Validate alignment
    validate_alignment(aligned_data)

    # 5) Analyze sensor correlations
    correlations = analyze_sensor_correlations(aligned_data)
    plot_correlations(correlations)

    # 6) Compare x-axis movement
    compare_x_axis_movement(aligned_data)

    analyze_distributions(aligned_data)
    analyze_noise(aligned_data)

    validate_statistics(aligned_data)

    X, y = create_sliding_windows(aligned_data, error_states)

    print(f"Shape of X before CNN: {X.shape}")  # Should be (num_samples, window_size, num_features)
    print(f"Shape of y before CNN: {y.shape}")
    validate_windows(X, y)

    print(np.isnan(X).any())
    print(np.isnan(y).any())

    cnn = CNNFeatureExtractor()
    X = torch.from_numpy(np.transpose(X, (0, 2, 1)))
    X_spatial = cnn.debug_forward(X)
    print(f"Shape of X_spatial after CNN: {X_spatial[0].shape}")

    validate_cnn_features(X, X_spatial)
    visualize_feature_maps(X_spatial)
    # train_ltc(X_spatial[0], y, device)
    train_lstm(X_spatial[0], y, device)

    # training_params = {
    #     'epochs': 200,
    #     'batch_size': 32,
    #     'learning_rate':s 0.015,
    #     'early_stopping_patience': 15
    # }
    #
    # ncp_wiring = AutoNCP(32, 15)
    # lnn = LTC(9, ncp_wiring)
    # lnn.to(device=device)
    #
    # optimizer = optim.Adam(lnn.parameters(), lr=training_params['learning_rate'])
    # criterion = RMSELoss()
    # early_stopping = EarlyStopping(patience=training_params['early_stopping_patience'])
    #
    # for epoch in range(training_params['epochs']):
    #     optimizer.zero_grad()
    #     output = lnn.forward(X_spatial)
    #     loss = criterion(output, y)
    #     loss.backward()
    #     optimizer.step()
    #
    #     if early_stopping(output['val_loss']):
    #         print(f"Early stopping triggered at epoch {epoch}")
    #         break


if __name__ == "__main__":
    main()
