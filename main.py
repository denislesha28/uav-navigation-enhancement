import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from data_loader import load_data
from kalman_filtering.kf_vector_calc import calc_error_state_vector, validate_error_states
from preprocessing.analysis.sensor_analysis import analyze_sensor_correlations, plot_correlations, \
    compare_x_axis_movement, analyze_distributions, analyze_noise, validate_statistics
from preprocessing.sensor_alignment import validate_alignment, align_sensors_multi_freq
from preprocessing.sliding_windows import create_sliding_windows, validate_windows
from training.cnn_feature_extraction import CNNFeatureExtractor, validate_cnn_features, visualize_feature_maps
from training.lstm.bi_lstm_training import train_lstm
from kalman_filtering.uav_navigation_kf import UAVNavigationKF, quaternion_to_euler


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
    validate_error_states(error_states)

#    plot_error_states(error_states, aligned_data["Ground_truth_df"].index)

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
