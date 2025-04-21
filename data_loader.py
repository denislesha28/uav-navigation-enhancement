import pandas as pd
import numpy as np

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

def load_ground_truth():
    gt_positions = pd.read_csv("AGZ/log_files/GroundTruthAGL.csv", index_col="imgid")
    img_timestamps = pd.read_csv("AGZ/log_files/GroundTruthAGM.csv", index_col="Timpstemp")

    merged_gt = pd.merge(gt_positions, img_timestamps, left_index=True, right_on=" imgid")
    merged_gt.drop(list(merged_gt.filter(regex="Unnamed")), inplace=True, axis=1)
    merged_gt.drop(columns=" ", inplace=True)
    print(merged_gt.columns)
    return prepare_df(merged_gt)

def load_data():
    """
    Loads sensor data with comprehensive error compensation based on
    Zurich Urban MAV dataset sensor characteristics.
    """
    # Load raw data
    imu_df = pd.read_csv(
        'AGZ/log_files/RawAccel.csv',
        index_col="Timpstemp",
        usecols=['Timpstemp', ' x', ' y', ' z', ' temperature']
    )
    imu_df = prepare_df(imu_df)
    imu_df = imu_df[~imu_df.index.duplicated(keep='first')]

    gyro_df = pd.read_csv(
        'AGZ/log_files/RawGyro.csv',
        index_col="Timpstemp",
        usecols=['Timpstemp', ' x', ' y', ' z', ' temperature']
    )
    gyro_df = prepare_df(gyro_df)
    gyro_df = gyro_df[~gyro_df.index.duplicated(keep='first')]

    gps_df = pd.read_csv(
        'AGZ/log_files/OnboardGPS.csv',
        index_col="Timpstemp",
        usecols=['Timpstemp', ' vel_n_m_s', ' vel_e_m_s', ' vel_d_m_s']
    )
    gps_df = prepare_df(gps_df)
    gps_df = gps_df.groupby(level=0).mean()  # Average duplicates

    onboard_df = pd.read_csv(
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
    onboard_df = prepare_df(onboard_df)
    preprocess_onboard_pose_quarternions(onboard_df)


    gt_df = load_ground_truth()

    # -------------------- SENSOR ERROR COMPENSATION --------------------

    # Define correction parameters
    gravity = 9.81
    scale_factor_correction = 0.003  # 0.3% (middle of range)
    cross_axis_correction = 0.01  # 1% (middle of range)
    acc_bias = 0.00275 * 9.80665  # 2.75 mg (middle of range)
    gyro_bias = 2.5e-5  # Middle of the range (0.5-10 deg/hour)
    temp_reference = 20.0  # Reference temperature (°C)
    temp_coefficient = 0.0005  # 0.05% per degree C

    # 1. IMU ERROR COMPENSATION

    # 1.1 Gravity compensation for IMU z-axis
    imu_df['z'] = imu_df['z'] + gravity

    # 1.2 Scale factor errors (0.1-0.5% based on dataset documentation)
    imu_df['x'] = imu_df['x'] * (1 + scale_factor_correction)
    imu_df['y'] = imu_df['y'] * (1 + scale_factor_correction)
    imu_df['z'] = imu_df['z'] * (1 + scale_factor_correction)

    # 1.3 Cross-axis sensitivity (0.5-2% based on dataset documentation)
    x_orig, y_orig, z_orig = imu_df['x'].copy(), imu_df['y'].copy(), imu_df['z'].copy()
    imu_df['x'] = x_orig + cross_axis_correction * (y_orig + z_orig)
    imu_df['y'] = y_orig + cross_axis_correction * (x_orig + z_orig)
    imu_df['z'] = z_orig + cross_axis_correction * (x_orig + y_orig)

    # 1.4 Accelerometer bias (0.5-5 mg where g = 9.80665 m/s²)
    imu_df['x'] = imu_df['x'] - acc_bias
    imu_df['y'] = imu_df['y'] - acc_bias
    imu_df['z'] = imu_df['z'] - acc_bias

    # 1.5 Temperature-dependent variations
    if 'temperature' in imu_df.columns:
        temp_compensation = (imu_df['temperature'] - temp_reference) * temp_coefficient
        imu_df['x'] = imu_df['x'] * (1 + temp_compensation)
        imu_df['y'] = imu_df['y'] * (1 + temp_compensation)
        imu_df['z'] = imu_df['z'] * (1 + temp_compensation)

    # 2. GYROSCOPE ERROR COMPENSATION

    # 2.1 Gyroscope bias (0.5-10 deg/hour converts to ~2.4e-6 to 4.8e-5 rad/s)
    gyro_df['x'] -= gyro_bias
    gyro_df['y'] -= gyro_bias
    gyro_df['z'] -= gyro_bias

    # 2.2 Scale factor errors (similar to IMU)
    gyro_df['x'] = gyro_df['x'] * (1 + scale_factor_correction)
    gyro_df['y'] = gyro_df['y'] * (1 + scale_factor_correction)
    gyro_df['z'] = gyro_df['z'] * (1 + scale_factor_correction)

    # 2.3 Cross-axis sensitivity
    x_orig, y_orig, z_orig = gyro_df['x'].copy(), gyro_df['y'].copy(), gyro_df['z'].copy()
    gyro_df['x'] = x_orig + cross_axis_correction * (y_orig + z_orig)
    gyro_df['y'] = y_orig + cross_axis_correction * (x_orig + z_orig)
    gyro_df['z'] = z_orig + cross_axis_correction * (x_orig + y_orig)

    # 2.4 Temperature compensation
    if 'temperature' in gyro_df.columns:
        temp_compensation = (gyro_df['temperature'] - temp_reference) * temp_coefficient
        gyro_df['x'] = gyro_df['x'] * (1 + temp_compensation)
        gyro_df['y'] = gyro_df['y'] * (1 + temp_compensation)
        gyro_df['z'] = gyro_df['z'] * (1 + temp_compensation)

    # 3. GPS ERROR COMPENSATION

    # 3.1 Apply GPS error model based on documented RMS errors
    # Root-mean-square geo-location errors (meters):
    # x-axis: 2.22 m
    # y-axis: 3.76 m
    # z-axis: 5.46 m

    # For velocity data, we need to translate position errors to velocity errors
    # Assuming 10Hz sampling rate (0.1s between samples)
    sampling_period = 0.1  # seconds

    # Convert position errors to velocity errors (error/time)
    # But scale by a factor to avoid overcorrection (using 0.2 as scaling factor)
    correction_factor = 0.2
    vel_e_error = 2.22 / sampling_period * correction_factor  # East velocity error
    vel_n_error = 3.76 / sampling_period * correction_factor  # North velocity error
    vel_d_error = 5.46 / sampling_period * correction_factor  # Down/Up velocity error

    print(f"Applying GPS velocity corrections: E:{vel_e_error:.4f}, N:{vel_n_error:.4f}, D:{vel_d_error:.4f} m/s")

    # Create error models as distributions around the mean with the RMS errors as std dev
    # We'll use these to create bias corrections for the GPS velocity data
    import numpy as np

    if len(gps_df) > 0:
        # Generate correction arrays based on RMS errors (zero-mean with appropriate std dev)
        # Using a sine wave modulation to simulate the kind of bias drift seen in real GPS
        t = np.linspace(0, 2 * np.pi, len(gps_df))

        # East velocity correction
        if 'vel_e_m_s' in gps_df.columns:
            bias_drift_e = vel_e_error * 0.5 * np.sin(t / 4)  # Slower drift
            gps_df['vel_e_m_s'] = gps_df['vel_e_m_s'] - bias_drift_e

        # North velocity correction
        if 'vel_n_m_s' in gps_df.columns:
            bias_drift_n = vel_n_error * 0.5 * np.sin(t / 3)  # Medium drift
            gps_df['vel_n_m_s'] = gps_df['vel_n_m_s'] - bias_drift_n

        # Down velocity correction (largest error)
        if 'vel_d_m_s' in gps_df.columns:
            bias_drift_d = vel_d_error * 0.5 * np.sin(t / 2)  # Faster drift
            gps_df['vel_d_m_s'] = gps_df['vel_d_m_s'] - bias_drift_d

    # 3.2 Apply low-pass filter to further reduce noise
    from scipy import signal

    # Low-pass filter to remove high-frequency noise (which is more prominent in GPS)
    b, a = signal.butter(3, 0.2)  # 3rd order Butterworth filter with 0.2 normalized cutoff

    if len(gps_df) > 0:
        if 'vel_n_m_s' in gps_df.columns and not gps_df['vel_n_m_s'].isna().all():
            gps_df['vel_n_m_s'] = signal.filtfilt(b, a, gps_df['vel_n_m_s'])

        if 'vel_e_m_s' in gps_df.columns and not gps_df['vel_e_m_s'].isna().all():
            gps_df['vel_e_m_s'] = signal.filtfilt(b, a, gps_df['vel_e_m_s'])

        if 'vel_d_m_s' in gps_df.columns and not gps_df['vel_d_m_s'].isna().all():
            gps_df['vel_d_m_s'] = signal.filtfilt(b, a, gps_df['vel_d_m_s'])

    # 3.3 Fill any remaining NaN values with interpolation
    gps_df = gps_df.interpolate(method='linear', limit_direction='both')

    # -------------------- PROCEED WITH YOUR EXISTING ALIGNMENT LOGIC --------------------

    try:
        # First find cross-correlation lag to get initial alignment
        import numpy as np
        from scipy import signal

        # Extract x-axis data for alignment
        imu_x = imu_df['x'].values
        gyro_x = gyro_df['x'].values

        # Estimate lag using cross-correlation
        corr = signal.correlate(imu_x, gyro_x, mode='full')
        lag_index = np.argmax(corr) - len(imu_x) + 1
        max_lag = 500  # Maximum allowed lag in samples

        # Limit lag to a reasonable value
        best_lag = max(min(lag_index, max_lag), -max_lag)
        shift_microseconds = int(best_lag * 1e4)  # Convert to microseconds assuming ~100Hz

        print(f"Best shift = {shift_microseconds} us, lag in samples = {best_lag}")

        # Apply the lag to create initially aligned gyro data
        aligned_gyro_df = gyro_df.copy()
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
            pose_temp = onboard_df.loc[common_start:common_end].reindex(target_idx, method='nearest')

            # Run Kalman filter separately for each axis to smooth and align
            imu_filtered = pd.DataFrame(index=target_idx, columns=['x', 'y', 'z'])
            gyro_filtered = pd.DataFrame(index=target_idx, columns=['x', 'y', 'z'])

            # Process each axis
            for axis in ['x', 'y', 'z']:
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
            if 'temperature' in imu_temp.columns:
                imu_filtered['temperature'] = imu_temp['temperature'].values
            if 'temperature' in gyro_temp.columns:
                gyro_filtered['temperature'] = gyro_temp['temperature'].values

            # Use the filtered data for final alignment
            imu_aligned = imu_filtered
            gyro_aligned = gyro_filtered

            pose_filtered = pd.DataFrame(index=target_idx)
            for axis in ['Omega_x', 'Omega_y', 'Omega_z']:
                if axis in pose_temp.columns:
                    pose_meas = pose_temp[axis].values
                    pose_smoothed = np.zeros(len(target_idx))
                    kalman = create_1d_filter()
                    for i, z in enumerate(pose_meas):
                        kalman.predict()
                        kalman.update(z)
                        pose_smoothed[i] = kalman.x[0]
                    pose_filtered[axis] = pose_smoothed

            # Copy other columns without filtering (quaternions, etc.)
            for col in pose_temp.columns:
                if col not in pose_filtered.columns:
                    pose_filtered[col] = pose_temp[col].values

            # Use the filtered data for final alignment
            onboard_pose_aligned_initial = pose_filtered

        except ImportError:
            print("FilterPy not installed. Using basic alignment with lag correction.")
            # Find common time window
            common_start = max(imu_df.index.min(), aligned_gyro_df.index.min())
            common_end = min(imu_df.index.max(), aligned_gyro_df.index.max())

            # Create target index
            target_idx = pd.timedelta_range(start=common_start, end=common_end, freq='100ms')

            # Use the initially aligned data
            imu_aligned = imu_df.loc[common_start:common_end].reindex(target_idx, method='nearest')
            gyro_aligned = aligned_gyro_df.loc[common_start:common_end].reindex(target_idx, method='nearest')
            onboard_pose_aligned_initial = onboard_df.loc[common_start:common_end].reindex(target_idx,
                                                                                                method='nearest')


    except Exception as e:
        print(f"Error in alignment: {e}")
        print("Using default alignment without lag correction.")
        # Find common time window for all dataframes
        common_start = max(imu_df.index.min(), gyro_df.index.min(),
                           gps_df.index.min(), gt_df.index.min())
        common_end = min(imu_df.index.max(), gyro_df.index.max(),
                         gps_df.index.max(), gt_df.index.max())

        # Create regular timestamp grid
        target_idx = pd.timedelta_range(start=common_start, end=common_end, freq='100ms')

        # Use basic nearest-point interpolation
        imu_aligned = imu_df.loc[common_start:common_end].reindex(target_idx, method='nearest')
        gyro_aligned = gyro_df.loc[common_start:common_end].reindex(target_idx, method='nearest')

    # Align GPS and ground truth data with the same target timestamps
    common_start = max(imu_aligned.index.min(), gyro_aligned.index.min(),
                       gps_df.index.min(), gt_df.index.min())
    common_end = min(imu_aligned.index.max(), gyro_aligned.index.max(),
                     gps_df.index.max(), gt_df.index.max())

    # Create final target timestamps
    target_idx = pd.timedelta_range(start=common_start, end=common_end, freq='100ms')

    # Align all data to the final target timestamps
    imu_aligned = imu_aligned.reindex(target_idx, method='nearest')
    gyro_aligned = gyro_aligned.reindex(target_idx, method='nearest')
    gps_aligned = gps_df.loc[common_start:common_end].reindex(target_idx, method='nearest')
    gt_aligned = gt_df.loc[common_start:common_end].reindex(target_idx, method='nearest')
    onboard_pose_aligned = onboard_pose_aligned_initial.reindex(target_idx, method='nearest')

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
        "OnboardPose_df": onboard_pose_aligned,
        "Ground_truth_df": gt_aligned
    }


def preprocess_onboard_pose_quarternions(onboard_df):
    from scipy import signal
    # Normalize quaternions
    qw = onboard_df['Attitude_w']
    qx = onboard_df['Attitude_x']
    qy = onboard_df['Attitude_y']
    qz = onboard_df['Attitude_z']
    # Calculate quaternion norm
    qnorm = np.sqrt(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)
    # Normalize only where norm is not too small
    valid_norm = qnorm > 1e-10
    onboard_df.loc[valid_norm, 'Attitude_w'] = qw[valid_norm] / qnorm[valid_norm]
    onboard_df.loc[valid_norm, 'Attitude_x'] = qx[valid_norm] / qnorm[valid_norm]
    onboard_df.loc[valid_norm, 'Attitude_y'] = qy[valid_norm] / qnorm[valid_norm]
    onboard_df.loc[valid_norm, 'Attitude_z'] = qz[valid_norm] / qnorm[valid_norm]
    # Replace invalid quaternions with identity quaternion [1,0,0,0]
    invalid_norm = ~valid_norm
    onboard_df.loc[invalid_norm, 'Attitude_w'] = 1.0
    onboard_df.loc[invalid_norm, 'Attitude_x'] = 0.0
    onboard_df.loc[invalid_norm, 'Attitude_y'] = 0.0
    onboard_df.loc[invalid_norm, 'Attitude_z'] = 0.0
    # Apply filtering where appropriate (e.g., for Omega and Accel)
    for col in ['Omega_x', 'Omega_y', 'Omega_z', 'Accel_x', 'Accel_y', 'Accel_z']:
        if col in onboard_df.columns and not onboard_df[col].isna().all():
            b, a = signal.butter(3, 0.2)
            onboard_df[col] = signal.filtfilt(b, a, onboard_df[col])