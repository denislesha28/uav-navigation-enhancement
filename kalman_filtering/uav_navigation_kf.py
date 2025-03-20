import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial.transform import Rotation


class UAVNavigationKF:
    def __init__(self, dt=0.1):
        """
        Initialize Error-State Kalman Filter for UAV navigation
        Implements Chen et al. (2024) loosely-coupled GNSS/INS fusion architecture
        """
        # Error state dimensions: attitude(3), velocity(3), position(3), gyro_bias(3), accel_bias(3)
        self.dim_x = 15

        # Measurement dimensions: acceleration(3) + angular_velocity(3) from IMU, velocity(3) from GPS
        self.dim_z = 9

        # Configure filter
        self.dt = dt
        self.kf = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)

        # Nominal state (what we're actually tracking)
        self.nominal_state = np.zeros(9)  # position(3), velocity(3), attitude(3)

        self.tau_gyro = 100.0  # seconds
        self.tau_accel = 100.0 # seconds

        # Initialize filter matrices
        self.initialize_matrices()

        # Bias estimation
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)



    def initialize_matrices(self):
        """Initialize all Kalman filter matrices"""
        # State transition matrix (F)
        self.kf.F = np.eye(self.dim_x)

        # Attitude to velocity coupling
        self.kf.F[3:6, 0:3] = np.eye(3) * self.dt

        # Velocity to position coupling
        self.kf.F[6:9, 3:6] = np.eye(3) * self.dt

        # Bias dynamics (first-order Markov process)
        self.kf.F[9:12, 9:12] = np.eye(3) * np.exp(-self.dt / self.tau_gyro)
        self.kf.F[12:15, 12:15] = np.eye(3) * np.exp(-self.dt / self.tau_accel)

        # Measurement matrix (H)
        # Relates error state to IMU and GPS measurements
        self.kf.H = np.zeros((self.dim_z, self.dim_x))

        # IMU accelerometer affected by attitude and accel bias
        self.kf.H[0:3, 0:3] = 0.1 * np.eye(3)  # Small coupling from attitude error to acceleration
        self.kf.H[0:3, 12:15] = np.eye(3)  # Direct effect of accel bias

        # IMU gyro affected by gyro bias
        self.kf.H[3:6, 9:12] = np.eye(3)  # Direct effect of gyro bias

        # GPS velocity directly measures velocity error
        self.kf.H[6:9, 3:6] = np.eye(3)  # Direct velocity measurement

        # Process noise (Q)
        # Based on expected sensor noise levels from Chen
        self.kf.Q = np.eye(self.dim_x) * 0.01
        self.kf.Q[0:3, 0:3] *= 0.01  # Attitude process noise
        self.kf.Q[3:6, 3:6] *= 0.1  # Velocity process noise
        self.kf.Q[6:9, 6:9] *= 0.5  # Position process noise
        self.kf.Q[9:12, 9:12] *= 0.001  # Gyro bias process noise
        self.kf.Q[12:15, 12:15] *= 0.01  # Reduce accel bias process noise, especially for Z-axis
        self.kf.Q[14, 14] *= 0.1

        # Measurement noise (R)
        self.kf.R = np.eye(self.dim_z)
        self.kf.R[0:3, 0:3] *= 0.5  # Accelerometer noise
        self.kf.R[3:6, 3:6] *= 0.1  # Gyroscope noise
        self.kf.R[6:9, 6:9] *= 0.25  # GPS velocity noise

        # Initial state covariance (P)
        self.kf.P = np.eye(self.dim_x)
        self.kf.P[0:3, 0:3] *= 0.1  # Initial attitude uncertainty
        self.kf.P[3:6, 3:6] *= 1.0  # Initial velocity uncertainty
        self.kf.P[6:9, 6:9] *= 10.0  # Initial position uncertainty
        self.kf.P[9:12, 9:12] *= 0.01  # Initial gyro bias uncertainty
        self.kf.P[12:15, 12:15] *= 0.1  # Initial accel bias uncertainty

        g = 9.81  # Gravity constant
        # Create skew-symmetric gravity matrix for attitude-to-acceleration coupling
        g_skew = np.array([
            [0, g, 0],
            [-g, 0, 0],
            [0, 0, 0]
        ])
        self.kf.H[0:3, 0:3] = 0.2 * g_skew  # Better coupling from attitude to acceleration

    def initialize(self, initial_state):
        """
        Initialize nominal state with values from ground truth

        Args:
            initial_state: array-like [position(3), velocity(3), attitude(3)]
        """
        self.nominal_state = initial_state.copy()

        # Initialize error state to zeros (perfect initial estimate)
        self.kf.x = np.zeros(self.dim_x)

        # Reset biases
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)

    def predict(self):
        """
        Prediction step of the filter with numerical stability improvements
        """
        # Ensure covariance symmetry before prediction
        self.kf.P = (self.kf.P + self.kf.P.T) / 2

        # Predict error state
        self.kf.predict()

        # Check for numerical issues after prediction
        min_eigenval = np.min(np.linalg.eigvals(self.kf.P))
        if min_eigenval < 1e-10:
            # Add small regularization to prevent singularity
            self.kf.P += np.eye(self.dim_x) * 1e-10

        # Update nominal state based on INS mechanization equations
        # In a real system, this would integrate accelerations and angular rates
        # For simplicity, we're just propagating with constant velocity
        self.nominal_state[0:3] += self.nominal_state[3:6] * self.dt  # Position += Velocity * dt

        return self.get_state()

    def update(self, measurement):
        """
        Update step using IMU and GPS measurements with adaptive noise handling
        """
        # Compute expected measurement based on nominal state
        expected_accel = np.zeros(3)  # Simplified - would use gravity model in real implementation
        expected_gyro = np.zeros(3)  # No expected rotation in level flight
        expected_vel = self.nominal_state[3:6]  # Expected velocity is nominal velocity

        expected_measurement = np.concatenate([
            expected_accel,
            expected_gyro,
            expected_vel
        ])

        # Measurement residual (innovation)
        residual = measurement - expected_measurement

        # Identify possible outliers
        residual_norm = np.linalg.norm(residual)
        outlier_threshold = 5.0  # Adjust based on your data

        # Adaptive noise covariance
        if residual_norm > outlier_threshold:
            # Temporarily increase measurement noise for outliers
            R_adaptive = self.kf.R.copy()
            # Increase noise for components with large residuals
            for i in range(len(residual)):
                if abs(residual[i]) > outlier_threshold:
                    R_adaptive[i, i] *= (1.0 + abs(residual[i]) / outlier_threshold)

            # Update with adaptive noise
            self.kf.update(residual, R=R_adaptive)
        else:
            # Normal update
            self.kf.update(residual)

        # Apply error state corrections to nominal state
        self.correct_nominal_state()

        # Return corrected state
        return self.get_state()

    def correct_nominal_state(self):
        """
        Apply error state corrections to nominal state and reset error state
        with robust quaternion handling
        """
        # 1. Apply attitude correction with robust quaternion handling
        try:
            # Get current attitude and error values
            attitude = self.nominal_state[6:9].copy()
            error_angles = self.kf.x[0:3].copy()

            # Check if both values are valid
            if np.all(np.isfinite(attitude)) and np.all(np.isfinite(error_angles)):
                error_norm = np.linalg.norm(error_angles)

                # Only apply rotation if there's a meaningful error
                if error_norm > 1e-10:
                    # Create rotation objects
                    try:
                        current_rotation = Rotation.from_euler('xyz', attitude)
                        error_rotation = Rotation.from_euler('xyz', error_angles)

                        # Compose rotations
                        new_rotation = error_rotation * current_rotation

                        # Update attitude with composed rotation
                        self.nominal_state[6:9] = new_rotation.as_euler('xyz')
                    except Exception as e:
                        # If rotation creation fails, use direct addition for small angles
                        if np.all(np.abs(error_angles) < 0.1):  # Small angle threshold
                            self.nominal_state[6:9] += error_angles
                # No else needed - if error is too small, we keep current attitude
        except Exception as e:
            print(f"Warning: Rotation composition failed: {e}")
            print(f"Attitude: {self.nominal_state[6:9]}, Error: {self.kf.x[0:3]}")
            # Use conservative approach - only apply small corrections
            if np.all(np.abs(self.kf.x[0:3]) < 0.05):
                self.nominal_state[6:9] += self.kf.x[0:3]

        # 2. Apply velocity correction with rate limiting
        velocity_error = self.kf.x[3:6].copy()
        # Apply rate limiting to prevent large jumps
        velocity_limit = 1.0  # m/s
        velocity_correction = np.clip(velocity_error, -velocity_limit, velocity_limit)
        self.nominal_state[3:6] += velocity_correction

        # 3. Apply position correction with rate limiting
        position_error = self.kf.x[6:9].copy()
        # Apply rate limiting
        position_limit = 5.0  # meters
        position_correction = np.clip(position_error, -position_limit, position_limit)
        self.nominal_state[0:3] += position_correction

        # 4. Update bias estimates with damping factor
        damping = 0.5  # Apply 50% of correction to avoid oscillations

        # Ensure valid bias updates
        gyro_bias_error = self.kf.x[9:12].copy()
        if np.all(np.isfinite(gyro_bias_error)):
            self.gyro_bias += damping * gyro_bias_error

        accel_bias_error = self.kf.x[12:15].copy()
        if np.all(np.isfinite(accel_bias_error)):
            self.accel_bias += damping * accel_bias_error

        # 5. Reset error state (error state feedback)
        self.kf.x = np.zeros(self.dim_x)

    def calculate_error_state(self, true_state):
        """
        Calculate error state vector with proper scaling and validation

        Args:
            true_state: array-like [position(3), velocity(3), attitude(3)]
        Returns:
            error_state: 15D error state vector
        """
        # Initialize empty error state
        error_state = np.zeros(15)

        try:
            # 1. Position error with scaling for large errors
            position_error = true_state[0:3] - self.nominal_state[0:3]
            # Validate position error
            if np.all(np.isfinite(position_error)):
                # Apply soft normalization - keeps small errors unchanged but scales down large errors
                position_scale = 10.0  # Adjust based on data scale
                error_state[6:9] = position_error / (1.0 + np.abs(position_error) / position_scale)
            else:
                # Set to zero if invalid
                error_state[6:9] = np.zeros(3)

            # 2. Velocity error with scaling
            velocity_error = true_state[3:6] - self.nominal_state[3:6]
            # Validate velocity error
            if np.all(np.isfinite(velocity_error)):
                velocity_scale = 5.0  # Adjust based on data characteristics
                error_state[3:6] = velocity_error / (1.0 + np.abs(velocity_error) / velocity_scale)
            else:
                error_state[3:6] = np.zeros(3)

            # 3. Attitude error (small angle approximation with normalization)
            # Get true and nominal attitudes
            true_attitude = true_state[6:9]
            nominal_attitude = self.nominal_state[6:9]

            # Validate attitudes
            if np.all(np.isfinite(true_attitude)) and np.all(np.isfinite(nominal_attitude)):
                # Normalize angle differences properly
                attitude_error = np.zeros(3)
                for i in range(3):
                    diff = true_attitude[i] - nominal_attitude[i]
                    # Normalize to [-π, π]
                    attitude_error[i] = np.arctan2(np.sin(diff), np.cos(diff))

                # Apply scaling to avoid very large attitude errors
                attitude_scale = 1.0  # radians
                error_state[0:3] = attitude_error / (1.0 + np.abs(attitude_error) / attitude_scale)
            else:
                error_state[0:3] = np.zeros(3)

            # 4. Include current bias estimates
            # Note: these are absolute values, not errors
            error_state[9:12] = self.gyro_bias
            error_state[12:15] = self.accel_bias

            # Final validation - ensure no NaN values
            if np.any(np.isnan(error_state)):
                print("Warning: NaN values in calculated error state")
                # Replace NaNs with zeros
                error_state = np.nan_to_num(error_state, nan=0.0)

        except Exception as e:
            print(f"Error calculating error state: {e}")
            # Return zeros on failure
            error_state = np.zeros(15)

        return error_state

    def get_state(self):
        """Return current nominal state estimate"""
        return self.nominal_state.copy()

    def get_complete_state(self):
        """Return full state including nominal state and bias estimates"""
        return np.concatenate([
            self.nominal_state,
            self.gyro_bias,
            self.accel_bias
        ])


def quaternion_to_euler(quaternion):
    """
    Convert quaternion to euler angles with proper validation

    Args:
        quaternion: dictionary or array-like with quaternion components
    Returns:
        euler: array-like, [roll, pitch, yaw] in radians or zeros if conversion fails
    """
    try:
        # Extract components, handling both dict and array inputs
        if isinstance(quaternion, dict):
            qw = quaternion['Attitude_w']
            qx = quaternion['Attitude_x']
            qy = quaternion['Attitude_y']
            qz = quaternion['Attitude_z']
        else:
            qw, qx, qy, qz = quaternion

        # Check for zero norm quaternion
        norm = np.sqrt(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)
        if norm < 1e-10:
            # Default to zero euler angles if quaternion is effectively zero
            return np.zeros(3)

        # Normalize quaternion
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

        # Create rotation object
        rot = Rotation.from_quat([qx, qy, qz, qw])  # scipy expects [x,y,z,w] order

        # Convert to euler angles
        return rot.as_euler('xyz')

    except Exception as e:
        # Fallback to zeros with warning
        print(f"Quaternion conversion failed: {e}")
        return np.zeros(3)