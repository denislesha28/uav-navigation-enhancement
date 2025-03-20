import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from kalman_filtering.uav_navigation_kf import UAVNavigationKF, quaternion_to_euler


def calc_error_state_vector(aligned_data: dict):
    """
    Calculate error state vector for each timestep using Kalman filter with robust quaternion handling

    Returns:
        List of (timestamp, error_state) tuples
    """
    imu_data = aligned_data['IMU_df']
    gyro_data = aligned_data['RawGyro_df']
    gps_data = aligned_data['Board_gps_df']
    ground_truth = aligned_data['Ground_truth_df']

    # Initialize KF
    kf = UAVNavigationKF(dt=0.1)  # 10Hz sampling rate

    # Initialize with safer defaults
    initial_position = np.zeros(3)
    initial_velocity = np.zeros(3)
    initial_attitude = np.zeros(3)

    # Try to get initial state from first ground truth entry
    try:
        if not ground_truth.empty and len(ground_truth) > 0:
            # Position
            if 'p_RS_R_x' in ground_truth.columns:
                initial_position = np.array([
                    ground_truth.iloc[0]['p_RS_R_x'],
                    ground_truth.iloc[0]['p_RS_R_y'],
                    ground_truth.iloc[0]['p_RS_R_z']
                ])

            # Velocity
            if 'Vel_x' in ground_truth.columns:
                initial_velocity = np.array([
                    ground_truth.iloc[0]['Vel_x'],
                    ground_truth.iloc[0]['Vel_y'],
                    ground_truth.iloc[0]['Vel_z']
                ])

            # Attitude - with robust quaternion handling
            quat_cols = ['Attitude_w', 'Attitude_x', 'Attitude_y', 'Attitude_z']
            if all(col in ground_truth.columns for col in quat_cols):
                quat_data = ground_truth.iloc[0][quat_cols]
                # Check for valid quaternion
                quat_values = [quat_data['Attitude_w'], quat_data['Attitude_x'],
                               quat_data['Attitude_y'], quat_data['Attitude_z']]

                if all(np.isfinite(quat_values)) and np.linalg.norm(quat_values) > 1e-10:
                    initial_attitude = quaternion_to_euler(quat_data)
    except Exception as e:
        print(f"Error initializing from ground truth: {e}")
        print("Using default zero initialization")

    # Combine into initial state vector
    initial_state = np.concatenate([
        initial_position,
        initial_velocity,
        initial_attitude
    ])

    # Initialize Kalman filter
    kf.initialize(initial_state)

    # Storage for error states and previous values
    error_states = []
    prev_idx = imu_data.index[0]
    prev_position = initial_position.copy()

    # Counters for diagnostics
    nan_warning_count = 0
    processed_count = 0
    error_count = 0

    # Process each timestamp
    for idx, _ in imu_data.iterrows():
        try:
            # Prediction step
            kf.predict()

            # Create measurement vector from IMU and GPS data
            # Ensure all required data is present
            if idx not in gyro_data.index or idx not in gps_data.index:
                continue

            measurement = np.array([
                # IMU accelerations
                imu_data.loc[idx]['x'],
                imu_data.loc[idx]['y'],
                imu_data.loc[idx]['z'],

                # IMU angular velocities
                gyro_data.loc[idx]['x'],
                gyro_data.loc[idx]['y'],
                gyro_data.loc[idx]['z'],

                # GPS velocities
                gps_data.loc[idx]['vel_n_m_s'],
                gps_data.loc[idx]['vel_e_m_s'],
                gps_data.loc[idx]['vel_d_m_s']
            ])

            # Check for NaN values in measurement
            if np.isnan(measurement).any():
                if nan_warning_count < 5:
                    print(f"NaN values in measurement at index {idx}, skipping")
                    nan_warning_count += 1
                continue

            # Update KF with measurement
            kf.update(measurement)
            processed_count += 1

            # Process ground truth if available
            if idx in ground_truth.index:
                # Get true position
                true_position = np.zeros(3)
                if 'p_RS_R_x' in ground_truth.columns:
                    true_position = np.array([
                        ground_truth.loc[idx]['p_RS_R_x'],
                        ground_truth.loc[idx]['p_RS_R_y'],
                        ground_truth.loc[idx]['p_RS_R_z']
                    ])
                else:
                    # If position isn't available, estimate from velocity
                    if idx > imu_data.index[0]:
                        dt = (idx - prev_idx).total_seconds()
                        vel = np.array([
                            ground_truth.loc[idx]['Vel_x'],
                            ground_truth.loc[idx]['Vel_y'],
                            ground_truth.loc[idx]['Vel_z']
                        ])
                        true_position = prev_position + vel * dt
                    else:
                        true_position = initial_position

                # Get velocity
                true_velocity = np.array([
                    ground_truth.loc[idx]['Vel_x'],
                    ground_truth.loc[idx]['Vel_y'],
                    ground_truth.loc[idx]['Vel_z']
                ])

                # Get attitude with validation
                quat_cols = ['Attitude_w', 'Attitude_x', 'Attitude_y', 'Attitude_z']
                quat_data = ground_truth.loc[idx][quat_cols]

                # Validate quaternion
                quat_values = [quat_data['Attitude_w'], quat_data['Attitude_x'],
                               quat_data['Attitude_y'], quat_data['Attitude_z']]

                if any(np.isnan(quat_values)) or np.linalg.norm(quat_values) < 1e-10:
                    # Skip error calculation for this frame if quaternion is invalid
                    continue

                # Convert quaternion to euler angles
                true_euler = quaternion_to_euler(quat_data)

                # Check for valid euler angles
                if not np.all(np.isfinite(true_euler)):
                    continue

                # Assemble true state vector
                true_state = np.concatenate([
                    true_position,
                    true_velocity,
                    true_euler
                ])

                # Calculate error state
                error_state = kf.calculate_error_state(true_state)

                # Validate error state
                if np.any(np.isnan(error_state)):
                    error_count += 1
                    if error_count < 5:
                        print(f"Warning: NaN values in error state at index {idx}")
                    continue

                # Add valid error state to result list
                error_states.append((idx, error_state))

                # Update previous values for next iteration
                prev_idx = idx
                prev_position = true_position

        except Exception as e:
            print(f"Error processing frame at {idx}: {e}")
            continue

    # Final diagnostics
    print(f"Processed {processed_count} frames, generated {len(error_states)} valid error states")
    if error_count > 0:
        print(f"Encountered {error_count} frames with invalid error states")

    return error_states

def validate_error_states(error_states):
    """
    Validate error states have reasonable values

    Args:
        error_states: List of (timestamp, error_state) tuples
    """
    if not error_states:
        print("No error states to validate!")
        return

    error_array = np.array([e for _, e in error_states])

    # Check for NaN values
    nan_count = np.isnan(error_array).sum()
    if nan_count > 0:
        print(f"WARNING: Found {nan_count} NaN values in error states!")

    # Check for all-zero components (problem indicator)
    zero_cols = np.where(np.all(error_array == 0, axis=0))[0]
    if zero_cols.size > 0:
        component_names = ['φE', 'φN', 'φU', 'δυE', 'δυN', 'δυU',
                           'δL', 'δλ', 'δh', 'εx', 'εy', 'εz',
                           '∇x', '∇y', '∇z']
        print(f"WARNING: These components are all zeros: {[component_names[i] for i in zero_cols]}")

    # Print summary statistics
    print("\nError State Statistics:")
    component_names = ['φE', 'φN', 'φU', 'δυE', 'δυN', 'δυU',
                       'δL', 'δλ', 'δh', 'εx', 'εy', 'εz',
                       '∇x', '∇y', '∇z']

    for i, name in enumerate(component_names):
        values = error_array[:, i]
        print(f"{name}: min={np.min(values):.4f}, max={np.max(values):.4f}, "
              f"mean={np.mean(values):.4f}, std={np.std(values):.4f}")

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
