import pandas as pd


def prepare_df(df, max_time_us=2.10e8):
    # Your existing prepare_df function
    # (keep this as is since it's working well)
    df.columns = df.columns.str.strip()
    df.index.names = ["Timestamp"]

    index_name = df.index.name
    start_time = df.index.min()
    normalized_timestamps = df.index - start_time
    df.index = pd.to_timedelta(normalized_timestamps, unit='us')

    if max_time_us:
        cutoff_delta = pd.Timedelta(microseconds=max_time_us)
        df = df.loc[df.index < cutoff_delta]

    df.index.name = index_name

    for col in df.columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

    time_span = df.index.max() - df.index.min()
    freq = len(df) / time_span.total_seconds() if time_span.total_seconds() > 0 else 0.0
    print(f"Time span: {time_span}")
    print(f"Sampling frequency (approx): {freq:.2f} Hz")

    return df


def load_data():
    """
    Loads sensor data with FilterPy's ImuKalman filter to address lag issues.
    """
    # Load IMU data directly - no extra processing
    imu_df = pd.read_csv(
        'AGZ/log_files/RawAccel.csv',
        index_col="Timpstemp",
        usecols=['Timpstemp', ' x', ' y', ' z', ' temperature']
    )
    imu_df = prepare_df(imu_df)
    imu_df = imu_df[~imu_df.index.duplicated(keep='first')]

    # Load gyro data
    raw_gyro_df = pd.read_csv(
        'AGZ/log_files/RawGyro.csv',
        index_col="Timpstemp",
        usecols=['Timpstemp', ' x', ' y', ' z', ' temperature']
    )
    raw_gyro_df = prepare_df(raw_gyro_df)
    raw_gyro_df = raw_gyro_df[~raw_gyro_df.index.duplicated(keep='first')]

    # Load GPS data
    board_gps_df = pd.read_csv(
        'AGZ/log_files/OnboardGPS.csv',
        index_col="Timpstemp",
        usecols=['Timpstemp', ' vel_n_m_s', ' vel_e_m_s', ' vel_d_m_s']
    )
    board_gps_df = prepare_df(board_gps_df)
    board_gps_df = board_gps_df.groupby(level=0).mean()  # Average duplicates

    # Load ground truth
    ground_truth_df = pd.read_csv(
        'AGZ/log_files/OnboardPose.csv',
        index_col="Timpstemp",
        usecols=[
            'Timpstemp',
            ' Omega_x', ' Omega_y', ' Omega_z',
            ' Accel_x', ' Accel_y', ' Accel_z',
            ' Vel_x', ' Vel_y', ' Vel_z',
            ' AccBias_x', ' AccBias_y', ' AccBias_z',
            ' Attitude_w', ' Attitude_x', ' Attitude_y', ' Attitude_z',
            ' Height'
        ]
    )
    ground_truth_df = prepare_df(ground_truth_df)

    try:
        # First find cross-correlation lag to get initial alignment
        import numpy as np
        from scipy import signal

        # Extract x-axis data for alignment
        imu_x = imu_df['x'].values
        gyro_x = raw_gyro_df['x'].values

        # Estimate lag using cross-correlation
        corr = signal.correlate(imu_x, gyro_x, mode='full')
        lag_index = np.argmax(corr) - len(imu_x) + 1
        max_lag = 500  # Maximum allowed lag in samples

        # Limit lag to a reasonable value
        best_lag = max(min(lag_index, max_lag), -max_lag)
        shift_microseconds = int(best_lag * 1e4)  # Convert to microseconds assuming ~100Hz

        print(f"Best shift = {shift_microseconds} us, lag in samples = {best_lag}")

        # Apply the lag to create initially aligned gyro data
        aligned_gyro_df = raw_gyro_df.copy()
        aligned_gyro_df.index = aligned_gyro_df.index - pd.Timedelta(microseconds=shift_microseconds)

        # Now use FilterPy to further refine the alignment
        try:
            from filterpy.kalman import KalmanFilter
            from filterpy.common import Q_discrete_white_noise
            import numpy as np

            # Setup a simple Kalman filter for each axis to align the data
            dt = 0.01  # Approximate sampling rate (100 Hz)

            def create_1d_filter():
                f = KalmanFilter(dim_x=2, dim_z=1)
                f.x = np.array([0., 0.])  # Initial state (position, velocity)
                f.F = np.array([[1., dt], [0., 1.]])  # State transition matrix
                f.H = np.array([[1., 0.]])  # Measurement function
                f.P *= 1000.  # Initial uncertainty
                f.R = 5.  # Measurement uncertainty
                f.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)  # Process uncertainty
                return f

            # Create filters for x, y, z axes
            x_filter = create_1d_filter()
            y_filter = create_1d_filter()
            z_filter = create_1d_filter()

            # Find common time window with the initially aligned data
            common_start = max(imu_df.index.min(), aligned_gyro_df.index.min())
            common_end = min(imu_df.index.max(), aligned_gyro_df.index.max())

            # Create regular timestamp grid at 10 Hz
            target_idx = pd.timedelta_range(start=common_start, end=common_end, freq='100ms')

            # Resample IMU and gyro to regular grid with the initial alignment
            imu_temp = imu_df.loc[common_start:common_end].reindex(target_idx, method='nearest')
            gyro_temp = aligned_gyro_df.loc[common_start:common_end].reindex(target_idx, method='nearest')

            # Run Kalman filter separately for each axis to smooth and align
            imu_filtered = pd.DataFrame(index=target_idx, columns=[' x', ' y', ' z'])
            gyro_filtered = pd.DataFrame(index=target_idx, columns=[' x', ' y', ' z'])

            # Process each axis
            for axis in [' x', ' y', ' z']:
                # Get measurements
                imu_meas = imu_temp[axis].values
                gyro_meas = gyro_temp[axis].values

                # Filter for IMU
                imu_smoothed = np.zeros(len(target_idx))
                kalman = create_1d_filter()
                for i, z in enumerate(imu_meas):
                    kalman.predict()
                    kalman.update(z)
                    imu_smoothed[i] = kalman.x[0]

                # Filter for gyro
                gyro_smoothed = np.zeros(len(target_idx))
                kalman = create_1d_filter()
                for i, z in enumerate(gyro_meas):
                    kalman.predict()
                    kalman.update(z)
                    gyro_smoothed[i] = kalman.x[0]

                # Store filtered results
                imu_filtered[axis] = imu_smoothed
                gyro_filtered[axis] = gyro_smoothed

            # Add temperature column back
            imu_filtered[' temperature'] = imu_temp[' temperature'].values
            gyro_filtered[' temperature'] = gyro_temp[' temperature'].values

            # Use the filtered data for final alignment
            imu_aligned = imu_filtered
            gyro_aligned = gyro_filtered

        except ImportError:
            print("FilterPy not installed. Using basic alignment with lag correction.")
            # Use the initially aligned data
            imu_aligned = imu_df.loc[common_start:common_end].reindex(target_idx, method='nearest')
            gyro_aligned = aligned_gyro_df.loc[common_start:common_end].reindex(target_idx, method='nearest')

    except Exception as e:
        print(f"Error in alignment: {e}")
        print("Using default alignment without lag correction.")
        # Find common time window for all dataframes
        common_start = max(imu_df.index.min(), raw_gyro_df.index.min(),
                           board_gps_df.index.min(), ground_truth_df.index.min())
        common_end = min(imu_df.index.max(), raw_gyro_df.index.max(),
                         board_gps_df.index.max(), ground_truth_df.index.max())

        # Create regular timestamp grid
        target_idx = pd.timedelta_range(start=common_start, end=common_end, freq='100ms')

        # Use basic nearest-point interpolation
        imu_aligned = imu_df.loc[common_start:common_end].reindex(target_idx, method='nearest')
        gyro_aligned = raw_gyro_df.loc[common_start:common_end].reindex(target_idx, method='nearest')

    # Align GPS and ground truth data with the same target timestamps
    common_start = max(imu_aligned.index.min(), gyro_aligned.index.min(),
                       board_gps_df.index.min(), ground_truth_df.index.min())
    common_end = min(imu_aligned.index.max(), gyro_aligned.index.max(),
                     board_gps_df.index.max(), ground_truth_df.index.max())

    # Create final target timestamps
    target_idx = pd.timedelta_range(start=common_start, end=common_end, freq='100ms')

    # Align all data to the final target timestamps
    imu_aligned = imu_aligned.reindex(target_idx, method='nearest')
    gyro_aligned = gyro_aligned.reindex(target_idx, method='nearest')
    gps_aligned = board_gps_df.loc[common_start:common_end].reindex(target_idx, method='nearest')
    gt_aligned = ground_truth_df.loc[common_start:common_end].reindex(target_idx, method='nearest')

    # Report alignment metrics
    print("\nIMU_df alignment metrics:")
    print(f"Original samples: {len(imu_df)}")
    print(f"Aligned samples: {len(imu_aligned)}")
    print(f"Unique samples: {imu_aligned.drop_duplicates().shape[0]}")
    dupe_ratio = (1 - imu_aligned.drop_duplicates().shape[0] / len(imu_aligned)) * 100 if len(imu_aligned) > 0 else 0
    print(f"Duplicate ratio: {dupe_ratio:.2f}%")

    # Check for alignment issues
    tolerance = pd.Timedelta(microseconds=20000)  # 20ms tolerance
    alignment_issues = 0

    for target_time in target_idx:
        nearest_original = abs(imu_df.index - target_time).min()
        if nearest_original > tolerance:
            alignment_issues += 1

    print(f"Alignment: Dropped {alignment_issues} samples outside tolerance ({tolerance.microseconds} us).")
    print(f"\nIMU Processing Stats: {{'processed_samples': {len(imu_df)}, 'alignment_issues': {alignment_issues}}}")

    return {
        "IMU_df": imu_aligned,
        "RawGyro_df": gyro_aligned,
        "Board_gps_df": gps_aligned,
        "Ground_truth_df": gt_aligned
    }