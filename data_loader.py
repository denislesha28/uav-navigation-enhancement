import pandas as pd

from preprocessing.imu_data_processor import IMUDataProcessor


def prepare_df(df, max_time_us=2.10e8):
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
    Loads, processes IMU data with the new simpler IMUDataProcessor,
    and also loads additional sensor data for multi-sensor analysis.
    """
    # Create a minimal IMUDataProcessor
    # We no longer do advanced bias removal or orientation, just alignment & resampling.
    processor = IMUDataProcessor(
        sampling_rate=100.0,
        timestamp_tolerance_ms=20.0,
        enable_logging=True,
        imu_axis_for_shift="x",  # correlate this axis in IMU
        pose_axis_for_shift="Omega_x",  # with this axis in Pose
        visualize_correlation=True
    )

    imu_df = processor.process_dataset(
        imu_path='AGZ/log_files/RawAccel.csv',
        pose_path='AGZ/log_files/OnboardPose.csv',
        output_path='AGZ/log_files/processed_imu.csv'
    )

    # Print stats
    stats = processor.get_statistics()
    print("\nIMU Processing Stats:", stats)

    # Now load additional data for your multi-sensor alignment
    raw_gyro_df = pd.read_csv(
        'AGZ/log_files/RawGyro.csv',
        index_col="Timpstemp",
        usecols=['Timpstemp', ' x', ' y', ' z', ' temperature']
    )
    raw_gyro_df = prepare_df(raw_gyro_df)
    raw_gyro_df = raw_gyro_df[~raw_gyro_df.index.duplicated(keep='first')]

    board_gps_df = pd.read_csv(
        'AGZ/log_files/OnboardGPS.csv',
        index_col="Timpstemp",
        usecols=['Timpstemp', ' vel_n_m_s', ' vel_e_m_s', ' vel_d_m_s']
    )
    board_gps_df = prepare_df(board_gps_df)
    board_gps_df = board_gps_df.groupby(level=0).mean()  # average duplicates
    # TODO refer back to this mean

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

    print(raw_gyro_df.head())
    print(board_gps_df.head())
    print(ground_truth_df.head())
    print("\n", board_gps_df.index.to_series().diff().head())

    # Return dictionaries for further multi-sensor steps
    return {
        "IMU_df": imu_df,
        "RawGyro_df": raw_gyro_df,
        "Board_gps_df": board_gps_df,
        "Ground_truth_df": ground_truth_df
    }
