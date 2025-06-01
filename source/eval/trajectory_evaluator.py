import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

from synthetic_data_generator import SyntheticFlightDataGenerator
from training.lstm.bi_lstm_training import EnhancedNavigationLSTM
from training.lstm.lstm_training import SensorFusionLSTM
from trajectory_visualizer import TrajectoryVisualizer
from kalman_filtering.uav_navigation_kf import UAVNavigationKF
from kalman_filtering.kf_vector_calc import calc_error_state_vector
from preprocessing.sliding_windows import create_sliding_windows
from training.cnn_feature_extraction import CNNFeatureExtractor


def predict_error_corrections(model, X, scaler=None):
    """
    Use trained BiLSTM model to predict error corrections

    Args:
        model: Trained BiLSTM model
        X: Input features
        scaler: Optional scaler for normalizing predictions

    Returns:
        Array of predicted error corrections
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X)

    # Convert to numpy
    predictions = predictions.cpu().numpy()

    # Inverse transform if scaler provided
    if scaler:
        predictions = scaler.inverse_transform(predictions)

    return predictions


def compare_trajectories(model_path, duration=100, pattern='figure8', noise_level='medium'):
    """
    Generate synthetic data, run KF, and apply LNN corrections

    Args:
        duration: Duration of flight in seconds
        pattern: Flight pattern to generate
        noise_level: Amount of sensor noise ('low', 'medium', 'high')
        model_path: Path to the trained BiLSTM model (if None, skips LNN correction)

    Returns:
        Visualization of trajectory comparison
    """
    # Set noise levels based on parameter
    if noise_level == 'low':
        noise = {'imu': 0.02, 'gyro': 0.005, 'gps': 0.05}
    elif noise_level == 'medium':
        noise = {'imu': 0.05, 'gyro': 0.01, 'gps': 0.1}
    else:  # high
        noise = {'imu': 0.1, 'gyro': 0.02, 'gps': 0.2}

    print(f"Generating {duration}s of '{pattern}' flight data with {noise_level} noise...")

    # Generate synthetic data
    data_gen = SyntheticFlightDataGenerator(
        duration=duration,
        sample_rate=10,  # 10Hz to match your pipeline
        noise_levels=noise
    )

    # Generate trajectory and dataframes
    data_gen.generate_trajectory(pattern=pattern)
    aligned_data = data_gen.generate_dataframes()

    # Store raw ground truth for visualization
    ground_truth = np.column_stack([
        aligned_data['Ground_truth_df']['p_RS_R_x'].values,
        aligned_data['Ground_truth_df']['p_RS_R_y'].values,
        aligned_data['Ground_truth_df']['p_RS_R_z'].values
    ])

    # Extract timestamps for plotting
    timestamps = aligned_data['Ground_truth_df'].index
    time_seconds = np.array([t.total_seconds() for t in timestamps])

    print("Running Kalman filter estimation...")

    # Initialize Kalman filter
    kf = UAVNavigationKF(dt=0.1)  # 10Hz sampling

    # Initialize with first ground truth position
    initial_state = np.zeros(9)
    try:
        # Position
        initial_state[0:3] = ground_truth[0]

        # Velocity
        initial_state[3:6] = np.array([
            aligned_data['Ground_truth_df']['Vel_x'].iloc[0],
            aligned_data['Ground_truth_df']['Vel_y'].iloc[0],
            aligned_data['Ground_truth_df']['Vel_z'].iloc[0]
        ])

        # Attitude
        quat = np.array([
            aligned_data['Ground_truth_df']['Attitude_w'].iloc[0],
            aligned_data['Ground_truth_df']['Attitude_x'].iloc[0],
            aligned_data['Ground_truth_df']['Attitude_y'].iloc[0],
            aligned_data['Ground_truth_df']['Attitude_z'].iloc[0]
        ])
        euler = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
        initial_state[6:9] = euler
    except Exception as e:
        print(f"Error initializing Kalman filter: {e}")
        print("Using default zero initialization")

    # Initialize Kalman filter
    kf.initialize(initial_state)

    # Storage for estimated positions (KF only)
    kf_positions = np.zeros_like(ground_truth)

    # Process each timestamp
    for i, idx in enumerate(aligned_data['IMU_df'].index):
        # Prediction step
        kf.predict()

        # Measurement vector
        measurement = np.array([
            # IMU accelerations
            aligned_data['IMU_df'].loc[idx]['x'],
            aligned_data['IMU_df'].loc[idx]['y'],
            aligned_data['IMU_df'].loc[idx]['z'],

            # IMU angular velocities
            aligned_data['RawGyro_df'].loc[idx]['x'],
            aligned_data['RawGyro_df'].loc[idx]['y'],
            aligned_data['RawGyro_df'].loc[idx]['z'],

            # GPS velocities
            aligned_data['Board_gps_df'].loc[idx]['vel_n_m_s'],
            aligned_data['Board_gps_df'].loc[idx]['vel_e_m_s'],
            aligned_data['Board_gps_df'].loc[idx]['vel_d_m_s']
        ])

        # Update KF with measurement
        kf.update(measurement)

        # Store KF position estimate
        kf_positions[i] = kf.get_state()[:3]

    # Calculate error states for model prediction
    error_states = calc_error_state_vector(aligned_data)

    # Apply trained model for error correction if provided
    if model_path is not None:
        print("Applying LNN/BiLSTM corrections...")

        # Create sliding windows for model input
        X, _ = create_sliding_windows(aligned_data, error_states)

        # Apply CNN feature extraction
        cnn = CNNFeatureExtractor()
        X = torch.from_numpy(np.transpose(X, (0, 2, 1)))
        X_spatial = cnn.debug_forward(X)

        # Load trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = X_spatial[0].shape[1]

        model = EnhancedNavigationLSTM()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        # Predict error corrections
        X_tensor = torch.tensor(X_spatial[0], dtype=torch.float32).to(device)
        predictions = predict_error_corrections(model, X_tensor)

        # Apply corrections to Kalman filter positions
        lnn_positions = np.copy(kf_positions)

        # The predictions are for the end of each window, so we need to align them
        window_size = 100  # As defined in your sliding_windows function
        stride = 10  # As defined in your sliding_windows function

        # Apply position corrections from predictions
        for i in range(len(predictions)):
            idx = i * stride + window_size - 1  # Index in the original trajectory
            if idx < len(lnn_positions):
                # Extract position error corrections
                pos_correction = predictions[i, 6:9]  # Position errors are at indices 6, 7, 8

                # Apply correction to the Kalman filter position
                lnn_positions[idx] = kf_positions[idx] - pos_correction

        # Smooth LNN positions (optional)
        # Simple moving average filter to smooth transitions
        window_avg = 5
        for i in range(window_avg, len(lnn_positions) - window_avg):
            lnn_positions[i] = np.mean(lnn_positions[i - window_avg:i + window_avg + 1], axis=0)
    else:
        # If no model provided, LNN positions are the same as KF positions
        lnn_positions = kf_positions
        print("No model provided. Showing only Kalman filter results.")

    print("Creating visualization...")

    # Visualize trajectories
    visualizer = TrajectoryVisualizer(
        ground_truth=ground_truth,
        kf_estimate=kf_positions,
        lnn_enhanced_estimate=lnn_positions,
        timestamps=time_seconds
    )

    # Create static plots
    fig_3d, _ = visualizer.plot_3d_comparison(
        title=f"{pattern.capitalize()} Flight Trajectory - {noise_level.capitalize()} Noise")
    fig_error, _ = visualizer.plot_error_comparison()

    # Create animation (optional)
    # ani = visualizer.create_animation(interval=50)

    plt.show()

    return {
        'ground_truth': ground_truth,
        'kf_estimate': kf_positions,
        'lnn_estimate': lnn_positions,
        'timestamps': time_seconds,
        'visualizer': visualizer
    }


def main():
    """Main function to run the visualization demo"""
    import argparse

    parser = argparse.ArgumentParser(description='UAV Position Estimation Visualization')
    parser.add_argument('--duration', type=int, default=60, help='Duration of flight in seconds')
    parser.add_argument('--pattern', type=str, default='spiral',
                        choices=['figure8', 'spiral', 'circle', 'hover_with_drift'],
                        help='Flight pattern to generate')
    parser.add_argument('--noise', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Amount of sensor noise')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained BiLSTM model (optional)')

    args = parser.parse_args()

    print("Compare trajectories")
    results = compare_trajectories(
        model_path="../training/models/best_lstm_model.pth",
        duration=60,
        pattern="figure8",
        noise_level="high"
    )

    print(f"Visualization complete. KF RMSE: {results['visualizer'].kf_rmse:.4f}m")

    if args.model is not None:
        print(f"LNN-Enhanced RMSE: {results['visualizer'].lnn_rmse:.4f}m")
        improvement = (results['visualizer'].kf_rmse - results['visualizer'].lnn_rmse) / results[
            'visualizer'].kf_rmse * 100
        print(f"Improvement: {improvement:.2f}%")

    # Save animation (optional)
    # results['visualizer'].create_animation(save_path=f'{args.pattern}_{args.noise}_animation.mp4')

if __name__ == "__main__":
    main()
