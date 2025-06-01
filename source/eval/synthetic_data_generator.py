import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


class SyntheticFlightDataGenerator:
    def __init__(self,
                 duration=100,  # seconds
                 sample_rate=10,  # Hz, matches your 100ms intervals
                 noise_levels={
                     'imu': 2.0,  # m/s² - significantly increased
                     'gyro': 0.3,  # rad/s - significantly increased
                     'gps': 1.5  # m/s - significantly increased
                 }):
        """
        Generate synthetic UAV flight data compatible with your Kalman filter pipeline.

        Args:
            duration: Length of flight in seconds
            sample_rate: Sampling frequency in Hz
            noise_levels: Noise standard deviation for each sensor type
        """
        self.duration = duration
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels
        self.n_samples = int(duration * sample_rate)

        # Initialize timestamps with 100ms intervals (10Hz) to match your pipeline
        self.timestamps = pd.timedelta_range(
            start=pd.Timedelta(0),
            periods=self.n_samples,
            freq=f'{int(1000 / sample_rate)}ms'
        )

        # Flight path parameters will be set in generate_trajectory()
        self.position = None
        self.velocity = None
        self.acceleration = None
        self.attitude = None
        self.angular_velocity = None
        self.accel_bias = np.zeros((self.n_samples, 3))
        self.gyro_bias = np.zeros((self.n_samples, 3))

    def generate_trajectory(self, pattern='figure8'):
        """
        Generate a flight trajectory following the specified pattern.
        Coordinate system: X-forward, Y-right, Z-down (standard NED frame)

        Args:
            pattern: Flight pattern type ('figure8', 'spiral', 'hover', 'circle', etc.)

        Returns:
            self: For method chaining
        """
        # Time vector
        t = np.linspace(0, self.duration, self.n_samples)
        dt = 1.0 / self.sample_rate

        if pattern == 'figure8':
            # Figure-8 pattern
            scale = 10.0  # Scale of the figure-8
            freq = 0.05  # Oscillation frequency

            # Position (NED frame: x-North, y-East, z-Down)
            self.position = np.zeros((self.n_samples, 3))
            self.position[:, 0] = scale * np.sin(2 * np.pi * freq * t)  # North
            self.position[:, 1] = scale * np.sin(2 * np.pi * freq * t) * np.cos(2 * np.pi * freq * t)  # East
            self.position[:, 2] = -5 - 0.5 * np.sin(2 * np.pi * freq * t * 0.5)  # Down (negative altitude)

            # Velocity (numerical differentiation)
            self.velocity = np.zeros_like(self.position)
            self.velocity[1:] = (self.position[1:] - self.position[:-1]) / dt
            self.velocity[0] = self.velocity[1]  # Copy first value

            # Acceleration (numerical differentiation of velocity)
            self.acceleration = np.zeros_like(self.velocity)
            self.acceleration[1:] = (self.velocity[1:] - self.velocity[:-1]) / dt
            self.acceleration[0] = self.acceleration[1]  # Copy first value

            # Add gravity to z-acceleration (9.81 m/s² downward in NED frame)
            self.acceleration[:, 2] += 9.81

            # Calculate attitude (roll, pitch, yaw) from velocity direction
            self.attitude = np.zeros((self.n_samples, 3))
            for i in range(self.n_samples):
                # Heading (yaw) from velocity direction in x-y plane
                if np.linalg.norm(self.velocity[i, :2]) > 1e-6:
                    self.attitude[i, 2] = np.arctan2(self.velocity[i, 1], self.velocity[i, 0])

                    # Pitch based on vertical velocity component (positive pitch means nose up)
                vel_norm = np.linalg.norm(self.velocity[i])
                if vel_norm > 1e-6:
                    self.attitude[i, 1] = -np.arcsin(self.velocity[i, 2] / vel_norm)

                # Roll - bank into turns (positive roll means right wing down)
                if i > 0:
                    # Calculate turn rate (yaw rate)
                    yaw_rate = (self.attitude[i, 2] - self.attitude[i - 1, 2]) / dt
                    # Bank proportional to turn rate
                    self.attitude[i, 0] = 0.2 * yaw_rate  # Scale factor for reasonable roll angles

            # Angular velocity (derived from attitude)
            self.angular_velocity = np.zeros_like(self.attitude)
            self.angular_velocity[1:] = (self.attitude[1:] - self.attitude[:-1]) / dt
            self.angular_velocity[0] = self.angular_velocity[1]

        elif pattern == 'spiral':
            # Ascending spiral in NED frame
            radius_start = 2.0
            radius_end = 15.0
            height_start = -10.0  # Starting altitude (negative in NED)
            height_end = -20.0  # Final altitude (negative in NED)
            revolutions = 3.0

            # Linearly increasing radius and height
            radius = np.linspace(radius_start, radius_end, self.n_samples)
            height = np.linspace(height_start, height_end, self.n_samples)
            angle = np.linspace(0, revolutions * 2 * np.pi, self.n_samples)

            # Position (NED frame)
            self.position = np.zeros((self.n_samples, 3))
            self.position[:, 0] = radius * np.cos(angle)  # North
            self.position[:, 1] = radius * np.sin(angle)  # East
            self.position[:, 2] = height  # Down (negative altitude)

            # Velocity (numerical differentiation)
            self.velocity = np.zeros_like(self.position)
            self.velocity[1:] = (self.position[1:] - self.position[:-1]) / dt
            self.velocity[0] = self.velocity[1]

            # Acceleration (numerical differentiation of velocity)
            self.acceleration = np.zeros_like(self.velocity)
            self.acceleration[1:] = (self.velocity[1:] - self.velocity[:-1]) / dt
            self.acceleration[0] = self.acceleration[1]

            # Add gravity to z-acceleration (9.81 m/s² downward in NED frame)
            self.acceleration[:, 2] += 9.81

            # Calculate attitude (similar to figure8)
            self.attitude = np.zeros((self.n_samples, 3))
            for i in range(self.n_samples):
                # Heading (yaw) - tangent to the spiral in NED frame
                self.attitude[i, 2] = angle[i] + np.pi / 2

                # Pitch - based on vertical velocity (positive is nose up)
                vel_norm = np.linalg.norm(self.velocity[i])
                if vel_norm > 1e-6:
                    self.attitude[i, 1] = -np.arcsin(self.velocity[i, 2] / vel_norm)

                # Roll - bank into the spiral (positive is right wing down)
                distance_from_center = np.sqrt(self.position[i, 0] ** 2 + self.position[i, 1] ** 2)
                if distance_from_center > 1e-6:
                    # Bank angle proportional to centripetal acceleration
                    centripetal_accel = np.linalg.norm(self.velocity[i, :2]) ** 2 / distance_from_center
                    self.attitude[i, 0] = np.arctan(centripetal_accel / 9.81) * 0.7  # Scale to realistic values

            # Angular velocity (derived from attitude changes)
            self.angular_velocity = np.zeros_like(self.attitude)
            self.angular_velocity[1:] = (self.attitude[1:] - self.attitude[:-1]) / dt
            self.angular_velocity[0] = self.angular_velocity[1]

        elif pattern == 'circle':
            # Simple circle pattern at constant altitude in NED frame
            radius = 10.0
            altitude = -5.0  # Negative in NED frame
            freq = 0.05

            # Position (NED frame)
            self.position = np.zeros((self.n_samples, 3))
            self.position[:, 0] = radius * np.cos(2 * np.pi * freq * t)  # North
            self.position[:, 1] = radius * np.sin(2 * np.pi * freq * t)  # East
            self.position[:, 2] = altitude  # Down (negative altitude)

            # Velocity (numerical differentiation)
            self.velocity = np.zeros_like(self.position)
            self.velocity[1:] = (self.position[1:] - self.position[:-1]) / dt
            self.velocity[0] = self.velocity[1]

            # Acceleration
            self.acceleration = np.zeros_like(self.velocity)
            self.acceleration[1:] = (self.velocity[1:] - self.velocity[:-1]) / dt
            self.acceleration[0] = self.acceleration[1]

            # Add gravity to z-acceleration (9.81 m/s² downward in NED frame)
            self.acceleration[:, 2] += 9.81

            # Calculate attitude
            self.attitude = np.zeros((self.n_samples, 3))
            for i in range(self.n_samples):
                # Heading (yaw) - tangent to circle in NED
                self.attitude[i, 2] = np.arctan2(self.velocity[i, 1], self.velocity[i, 0])

                # Pitch - level flight
                self.attitude[i, 1] = 0.0

                # Roll - constant bank in circular flight
                # Bank angle for coordinated turn
                v_horizontal = np.linalg.norm(self.velocity[i, :2])
                self.attitude[i, 0] = np.arctan(v_horizontal ** 2 / (radius * 9.81)) * 0.9  # Slightly less than ideal

            # Angular velocity
            self.angular_velocity = np.zeros_like(self.attitude)
            self.angular_velocity[1:] = (self.attitude[1:] - self.attitude[:-1]) / dt
            self.angular_velocity[0] = self.angular_velocity[1]

        elif pattern == 'hover_with_drift':
            # Hovering with slight drift (more realistic) in NED frame
            # Central position
            center_pos = np.array([0.0, 0.0, -5.0])  # NED frame, negative altitude

            # Random walk for position
            self.position = np.zeros((self.n_samples, 3))
            self.position[0] = center_pos

            # Small random movements
            pos_noise_scale = 0.02
            for i in range(1, self.n_samples):
                # Random walk with tendency to return to center
                random_step = np.random.normal(0, pos_noise_scale, 3)
                return_to_center = (center_pos - self.position[i - 1]) * 0.01  # Slight pull to center
                self.position[i] = self.position[i - 1] + random_step + return_to_center

            # Add small oscillation for more realism
            osc_freq = 0.2
            osc_amp = 0.1
            self.position[:, 0] += osc_amp * np.sin(2 * np.pi * osc_freq * t)  # North
            self.position[:, 1] += osc_amp * np.cos(2 * np.pi * osc_freq * t * 1.3)  # East
            self.position[:, 2] += osc_amp * 0.5 * np.sin(2 * np.pi * osc_freq * t * 0.7)  # Down

            # Velocity (numerical differentiation)
            self.velocity = np.zeros_like(self.position)
            self.velocity[1:] = (self.position[1:] - self.position[:-1]) / dt
            self.velocity[0] = self.velocity[1]

            # Acceleration
            self.acceleration = np.zeros_like(self.velocity)
            self.acceleration[1:] = (self.velocity[1:] - self.velocity[:-1]) / dt
            self.acceleration[0] = self.acceleration[1]

            # Add gravity to z-acceleration (9.81 m/s² downward in NED frame)
            self.acceleration[:, 2] += 9.81

            # Set attitudes to small fluctuations
            self.attitude = np.zeros((self.n_samples, 3))
            attitude_noise = 0.03  # Small attitude changes

            # Generate random walk for attitude
            for i in range(1, self.n_samples):
                # Random walk with tendency to return to level
                random_step = np.random.normal(0, attitude_noise, 3)
                return_to_level = -self.attitude[i - 1] * 0.05  # Slight pull to level
                self.attitude[i] = self.attitude[i - 1] + random_step + return_to_level

            # Angular velocity
            self.angular_velocity = np.zeros_like(self.attitude)
            self.angular_velocity[1:] = (self.attitude[1:] - self.attitude[:-1]) / dt
            self.angular_velocity[0] = self.angular_velocity[1]

        # Generate bias values with significant drift
        # Initial biases
        initial_accel_bias = np.random.normal(0, 0.05, 3)  # Increased initial bias
        initial_gyro_bias = np.random.normal(0, 0.02, 3)  # Increased initial bias

        # Random walk for biases with strong drift
        self.accel_bias[0] = initial_accel_bias
        self.gyro_bias[0] = initial_gyro_bias

        # Add directional drift component
        drift_direction = np.array([0.0003, 0.0002, 0.0004])  # Consistent drift direction

        for i in range(1, self.n_samples):
            # Much stronger random walk for biases
            self.accel_bias[i] = self.accel_bias[i - 1] + np.random.normal(0, 0.001, 3)
            self.gyro_bias[i] = self.gyro_bias[i - 1] + np.random.normal(0, 0.0005, 3)

            # Add directional drift component
            self.accel_bias[i] += drift_direction

        return self

    def add_sensor_noise(self):
        """
        Add realistic sensor noise to the generated trajectory

        Returns:
            tuple: (noisy_accel, noisy_gyro, noisy_gps_vel)
        """
        # IMU noise (accelerometer)
        noisy_accel = self.acceleration + np.random.normal(
            0, self.noise_levels['imu'], size=self.acceleration.shape)

        # Add accelerometer bias
        for i in range(self.n_samples):
            noisy_accel[i] += self.accel_bias[i]

        # Add strong structured noise components
        freq1 = 0.25  # Hz
        amplitude1 = 2.0  # m/s² - significantly increased
        structured_noise1 = amplitude1 * np.sin(2 * np.pi * freq1 * np.linspace(0, self.duration, self.n_samples))
        noisy_accel[:, 0] += structured_noise1

        # Add second frequency component
        freq2 = 0.7  # Hz
        amplitude2 = 1.5  # m/s² - significantly increased
        structured_noise2 = amplitude2 * np.sin(2 * np.pi * freq2 * np.linspace(0, self.duration, self.n_samples))
        noisy_accel[:, 1] += structured_noise2

        # Add occasional outliers (non-Gaussian noise)
        outlier_indices = np.random.choice(self.n_samples, size=int(self.n_samples * 0.05), replace=False)
        for idx in outlier_indices:
            noisy_accel[idx] += np.random.normal(0, 5.0, 3)  # Large outliers

        z_axis_noise = np.zeros_like(self.position[:, 2])
        # Add state-dependent noise (higher noise at higher velocities)
        for i in range(self.n_samples):
            vel_magnitude = np.linalg.norm(self.velocity[i])
            velocity_dependent_noise = 0.1 * vel_magnitude * np.random.normal(0, 1, 3)
            noisy_accel[i] += velocity_dependent_noise

            altitude = -self.position[i, 2]  # Convert from NED to altitude
            altitude_dependent_bias = 0.1 * altitude  # Higher altitude = more bias
            z_axis_noise[i] = altitude_dependent_bias

            # Add sudden pressure changes (common in barometric sensors)
            if i > 0 and i % 50 == 0:  # Every 5 seconds at 10Hz
                pressure_jump = np.random.normal(0, 3.0)  # Random pressure jumps
                z_axis_noise[i:min(i + 20, self.n_samples)] += pressure_jump * np.exp(
                    -np.arange(min(20, self.n_samples - i)) / 5)  # Exponential decay

        noisy_accel[:, 2] += z_axis_noise

        # Add specific bias drift to z-axis measurements
        z_drift = np.cumsum(np.random.normal(0, 0.01, self.n_samples))
        noisy_accel[:, 2] += z_drift

        # Add occasional outliers (non-Gaussian noise)
        outlier_indices = np.random.choice(self.n_samples, size=int(self.n_samples * 0.05), replace=False)
        for idx in outlier_indices:
            noisy_accel[idx] += np.random.normal(0, 5.0, 3)  # Large outliers
        # Gyroscope noise
        noisy_gyro = self.angular_velocity + np.random.normal(
            0, self.noise_levels['gyro'], size=self.angular_velocity.shape)

        # Add gyroscope bias
        for i in range(self.n_samples):
            noisy_gyro[i] += self.gyro_bias[i]

        # Add occasional gyro outliers
        for idx in outlier_indices:
            noisy_gyro[idx] += np.random.normal(0, 0.5, 3)

        # GPS velocity noise
        noisy_gps_vel = self.velocity + np.random.normal(
            0, self.noise_levels['gps'], size=self.velocity.shape)

        # Add structured GPS noise (e.g., multipath effects)
        gps_error_freq = 0.1  # Hz
        gps_error_amp = 1.0  # m/s
        gps_structured_error = gps_error_amp * np.sin(
            2 * np.pi * gps_error_freq * np.linspace(0, self.duration, self.n_samples))
        noisy_gps_vel[:, 0] += gps_structured_error

        return noisy_accel, noisy_gyro, noisy_gps_vel

    def generate_dataframes(self):
        """
        Generate dataframes in the format expected by your pipeline

        Returns:
            dict: Dictionary containing IMU_df, RawGyro_df, Board_gps_df, and Ground_truth_df
        """
        if self.position is None:
            self.generate_trajectory()

        # Add sensor noise
        noisy_accel, noisy_gyro, noisy_gps_vel = self.add_sensor_noise()

        # Create quaternions from euler angles for ground truth
        quaternions = np.zeros((self.n_samples, 4))
        for i in range(self.n_samples):
            quat = Rotation.from_euler('xyz', self.attitude[i]).as_quat()
            # Convert from scipy's [x,y,z,w] to [w,x,y,z] format
            quaternions[i] = [quat[3], quat[0], quat[1], quat[2]]

        # Create IMU DataFrame (accelerometer data already in body frame)
        imu_df = pd.DataFrame(
            data=noisy_accel,
            index=self.timestamps,
            columns=['x', 'y', 'z']
        )
        imu_df['temperature'] = 25.0 + np.random.normal(0, 0.5, size=self.n_samples)  # Fake temperature data

        # Create Gyro DataFrame (angular velocity data already in body frame)
        gyro_df = pd.DataFrame(
            data=noisy_gyro,
            index=self.timestamps,
            columns=['x', 'y', 'z']
        )
        gyro_df['temperature'] = 25.0 + np.random.normal(0, 0.5, size=self.n_samples)  # Fake temperature data

        # Create GPS DataFrame (velocity directly in NED frame, matching KF expectation)
        gps_df = pd.DataFrame(
            data=noisy_gps_vel,
            index=self.timestamps,
            columns=['vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s']
        )

        # Create Ground Truth DataFrame
        gt_df = pd.DataFrame(
            index=self.timestamps
        )

        # Add position data
        gt_df['p_RS_R_x'] = self.position[:, 0]  # North
        gt_df['p_RS_R_y'] = self.position[:, 1]  # East
        gt_df['p_RS_R_z'] = self.position[:, 2]  # Down

        # Add velocity data
        gt_df['Vel_x'] = self.velocity[:, 0]  # North
        gt_df['Vel_y'] = self.velocity[:, 1]  # East
        gt_df['Vel_z'] = self.velocity[:, 2]  # Down

        # Add attitude quaternion
        gt_df['Attitude_w'] = quaternions[:, 0]
        gt_df['Attitude_x'] = quaternions[:, 1]
        gt_df['Attitude_y'] = quaternions[:, 2]
        gt_df['Attitude_z'] = quaternions[:, 3]

        # Add other fields present in your original data
        gt_df['Omega_x'] = self.angular_velocity[:, 0]
        gt_df['Omega_y'] = self.angular_velocity[:, 1]
        gt_df['Omega_z'] = self.angular_velocity[:, 2]

        gt_df['Accel_x'] = self.acceleration[:, 0]
        gt_df['Accel_y'] = self.acceleration[:, 1]
        gt_df['Accel_z'] = self.acceleration[:, 2]

        # Add bias values
        gt_df['AccBias_x'] = self.accel_bias[:, 0]
        gt_df['AccBias_y'] = self.accel_bias[:, 1]
        gt_df['AccBias_z'] = self.accel_bias[:, 2]

        # Add b_w fields (matching the structure in applsci-14-05493-v2.pdf)
        gt_df['b_w_RS_S_x'] = self.gyro_bias[:, 0]
        gt_df['b_w_RS_S_y'] = self.gyro_bias[:, 1]
        gt_df['b_w_RS_S_z'] = self.gyro_bias[:, 2]

        # Add b_a fields for accelerometer bias
        gt_df['b_a_RS_S_x'] = self.accel_bias[:, 0]
        gt_df['b_a_RS_S_y'] = self.accel_bias[:, 1]
        gt_df['b_a_RS_S_z'] = self.accel_bias[:, 2]

        gt_df['Height'] = -self.position[:, 2]  # Convert Down to Height (positive up)

        return {
            "IMU_df": imu_df,
            "RawGyro_df": gyro_df,
            "Board_gps_df": gps_df,
            "Ground_truth_df": gt_df
        }

    def export_to_csvs(self, output_dir="synthetic_data"):
        """
        Export generated data to CSV files in the same format as your real data

        Args:
            output_dir: Directory to save CSV files

        Returns:
            str: Path to the output directory
        """
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataframes = self.generate_dataframes()

        # Export each dataframe
        for name, df in dataframes.items():
            filename = name.replace("_df", ".csv")
            # Convert timedelta index to timestamps
            df_export = df.copy()
            df_export.index = df_export.index.total_seconds() * 1e6  # Convert to microseconds
            df_export.index.name = "Timpstemp"  # Match your original format

            # Fix column names to match your original format
            df_export.columns = [f" {col}" if not col.startswith(" ") else col for col in df_export.columns]

            df_export.to_csv(os.path.join(output_dir, filename))

        return output_dir

    def get_ground_truth_trajectory(self):
        """
        Return the ground truth trajectory for visualization

        Returns:
            numpy.ndarray: Ground truth position trajectory (n_samples x 3)
        """
        if self.position is None:
            self.generate_trajectory()

        return self.position

    def get_time_vector(self):
        """
        Return time vector in seconds for plotting

        Returns:
            numpy.ndarray: Time vector in seconds
        """
        return np.array([t.total_seconds() for t in self.timestamps])