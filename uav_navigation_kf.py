from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
from scipy.spatial.transform import Rotation


class UAVNavigationKF:
    def __init__(self, dt=0.1):  # dt=0.1 for 10Hz
        """
        Initialize Kalman Filter for UAV navigation error state estimation
        Based on Chen et al. (2024) loosely-coupled architecture
        """
        # State vector dimension: position(3), velocity(3), orientation(3)
        self.dim_state = 9
        # Measurement dimension: GPS position(3), velocity(3)
        self.dim_measurement = 6

        self.dt = dt
        self.kf = KalmanFilter(dim_x=self.dim_state, dim_z=self.dim_measurement)

        # Initialize state transition matrix (F)
        self.kf.F = self._create_F()

        # Initialize measurement matrix (H)
        self.kf.H = self._create_H()

        # Initialize process noise (Q)
        self.kf.Q = self._create_Q()

        # Initialize measurement noise (R)
        self.kf.R = self._create_R()

        # Initialize state covariance (P)
        self.kf.P = self._create_P()

    def _create_F(self):
        """State transition matrix"""
        F = np.eye(self.dim_state)
        # Update position from velocity
        F[0:3, 3:6] = np.eye(3) * self.dt
        return F

    def _create_H(self):
        """Measurement matrix"""
        H = np.zeros((self.dim_measurement, self.dim_state))
        H[0:3, 0:3] = np.eye(3)  # Position measurements
        H[3:6, 3:6] = np.eye(3)  # Velocity measurements
        return H

    def _create_Q(self):
        """Process noise matrix"""
        q = Q_discrete_white_noise(
            dim=3,  # position, velocity, acceleration
            dt=self.dt,  # sample time
            var=0.1,  # process variance
            block_size=3  # for 3D motion
        )
        # Extend Q matrix for orientation states
        Q = np.zeros((self.dim_state, self.dim_state))
        Q[0:6, 0:6] = q[0:6, 0:6]  # Position and velocity
        Q[6:9, 6:9] = np.eye(3) * 0.01  # Orientation uncertainty
        return Q

    def _create_R(self):
        """Measurement noise matrix"""
        # Based on typical GPS accuracy from Chen et al.
        R = np.zeros((self.dim_measurement, self.dim_measurement))
        R[0:3, 0:3] = np.eye(3) * 5.0  # Position measurement noise (m²)
        R[3:6, 3:6] = np.eye(3) * 0.1  # Velocity measurement noise (m²/s²)
        return R

    def _create_P(self):
        """Initial state covariance"""
        P = np.eye(self.dim_state)
        P[0:3, 0:3] *= 10.0  # Position uncertainty
        P[3:6, 3:6] *= 1.0  # Velocity uncertainty
        P[6:9, 6:9] *= 0.1  # Orientation uncertainty
        return P

    def initialize(self, initial_state):
        """Initialize state vector"""
        self.kf.x = initial_state

    def predict(self):
        """Prediction step"""
        self.kf.predict()
        return self.kf.x

    def update(self, measurement):
        """Update step using GPS measurement"""
        self.kf.update(measurement)
        return self.kf.x

    def calculate_error_state(self, true_state):
        error_state = np.zeros(15)

        # true_state structure:
        # [0:3] - Accel_xyz
        # [3:6] - Vel_xyz
        # [6:9] - Euler angles

        error_state[0:3] = true_state[6:9] - self.kf.x[6:9]  # Attitude
        error_state[3:6] = true_state[3:6] - self.kf.x[3:6]  # Velocity
        error_state[6:9] = true_state[0:3] - self.kf.x[0:3]  # Position
        error_state[9:12] = np.zeros(3)  # Gyro bias
        error_state[12:15] = np.zeros(3)  # Acc bias

        return error_state

    def get_state(self):
        """Return current state estimate"""
        return self.kf.x.copy()

def quaternion_to_euler(quaternion):
    """
    Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw]

    Args:
        quaternion: array-like, [w, x, y, z]
    Returns:
        euler: array-like, [roll, pitch, yaw] in radians
    """
    # Create rotation object
    rot = Rotation.from_quat([
        quaternion['Attitude_x'],  # x
        quaternion['Attitude_y'],  # y
        quaternion['Attitude_z'],  # z
        quaternion['Attitude_w']   # w - note scipy expects w last
    ])
    # Convert to euler angles
    return rot.as_euler('xyz')
