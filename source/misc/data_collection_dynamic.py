import numpy as np
import time
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
from orbit import SafeSurveyNavigator


class DynamicFlightNavigator(SafeSurveyNavigator):
    def __init__(self,
                 save_dir="airsim_collected_data",
                 session_name="dynamic_flight",
                 sampling_freq=100,
                 duration=180,
                 base_speed=5.0,
                 **kwargs):

        super().__init__(**kwargs)
        self.save_dir = Path(save_dir)
        self.session_dir = self.save_dir / session_name
        self.sampling_freq = sampling_freq
        self.duration = duration
        self.base_speed = base_speed

        # Create directories
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / 'imu0').mkdir(exist_ok=True)
        (self.session_dir / 'state_groundtruth_estimate0').mkdir(exist_ok=True)

        # Initialize data storage
        self.imu_data = []
        self.ground_truth = []
        self.timestamps = []

        # Flight pattern parameters
        self.phase_duration = 15  # seconds per maneuver phase
        self.current_phase = 0
        self.phase_start_time = 0
        self.last_position = None
        self.target_position = None

        # Initialize noise parameters (based on typical IMU characteristics)
        self.noise_params = {
            'acc': {
                'white_noise_std': 0.02,  # m/s^2
                'random_walk_std': 0.001,  # m/s^2/sqrt(Hz)
                'bias_std': 0.01,  # m/s^2
                'bias_instability': 0.0002  # m/s^2/s
            },
            'gyro': {
                'white_noise_std': 0.001,  # rad/s
                'random_walk_std': 0.0001,  # rad/s/sqrt(Hz)
                'bias_std': 0.001,  # rad/s
                'bias_instability': 0.00002  # rad/s/s
            }
        }

        # Initialize environmental parameters
        self.env_params = {
            'vibration_amplitude': 0.05,  # base amplitude
            'vibration_freq': 100.0,  # Hz (motor frequency)
            'turbulence_std': 0.1,  # base turbulence
            'temp_sensitivity': 0.001  # sensitivity to temperature changes
        }

        # Initialize state for noise generation
        self.acc_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.prev_random_walk_acc = np.zeros(3)
        self.prev_random_walk_gyro = np.zeros(3)
        self.temperature = 20.0  # Starting temperature in Celsius

    def simulate_sensor_noise(self, sensor_type='acc'):
        """Generate realistic sensor noise including various components"""
        params = self.noise_params[sensor_type]

        # White noise (Gaussian)
        white_noise = np.random.normal(0, params['white_noise_std'], 3)

        # Random walk (integrated white noise)
        random_walk_step = np.random.normal(0, params['random_walk_std'], 3)
        if sensor_type == 'acc':
            self.prev_random_walk_acc += random_walk_step / np.sqrt(self.sampling_freq)
            random_walk = self.prev_random_walk_acc
        else:
            self.prev_random_walk_gyro += random_walk_step / np.sqrt(self.sampling_freq)
            random_walk = self.prev_random_walk_gyro

        # Bias instability
        if sensor_type == 'acc':
            self.acc_bias += np.random.normal(0, params['bias_instability'], 3)
            bias = self.acc_bias
        else:
            self.gyro_bias += np.random.normal(0, params['bias_instability'], 3)
            bias = self.gyro_bias

        # Temperature effects
        self.temperature += np.random.normal(0, 0.1)  # Temperature variation
        temp_effect = (self.temperature - 20.0) * self.env_params['temp_sensitivity']

        return white_noise + random_walk + bias + temp_effect

    def simulate_environmental_effects(self):
        """Simulate environmental effects like vibrations and turbulence"""
        # Motor vibrations (multiple frequencies)
        t = time.time()
        vibration = (
                self.env_params['vibration_amplitude'] *
                (np.sin(2 * np.pi * self.env_params['vibration_freq'] * t) +
                 0.5 * np.sin(4 * np.pi * self.env_params['vibration_freq'] * t))
        )

        # Air turbulence (random walk process)
        turbulence = np.random.normal(0, self.env_params['turbulence_std'])

        return vibration + turbulence

    def add_realistic_effects(self, imu_data):
        """Add all realistic effects to IMU data"""
        # Add sensor noise
        acc_noise = self.simulate_sensor_noise('acc')
        gyro_noise = self.simulate_sensor_noise('gyro')

        # Add environmental effects
        env_effects = self.simulate_environmental_effects()

        # Apply to IMU data
        imu_data.linear_acceleration.x_val += acc_noise[0] + env_effects
        imu_data.linear_acceleration.y_val += acc_noise[1] + env_effects
        imu_data.linear_acceleration.z_val += acc_noise[2] + env_effects

        imu_data.angular_velocity.x_val += gyro_noise[0] + env_effects * 0.1
        imu_data.angular_velocity.y_val += gyro_noise[1] + env_effects * 0.1
        imu_data.angular_velocity.z_val += gyro_noise[2] + env_effects * 0.1

        return imu_data

    def generate_next_waypoint(self):
        """Generate next target position based on current phase"""
        current_pos = self.client.getMultirotorState().kinematics_estimated.position
        current_pos_array = np.array([current_pos.x_val, current_pos.y_val, current_pos.z_val])

        # Initialize last_position if None
        if self.last_position is None:
            self.last_position = current_pos_array
            self.target_position = current_pos_array  # Initialize target to current position

        # Get current phase
        phase = (int(time.time() - self.phase_start_time) // self.phase_duration) % 5

        # Generate new target only if phase changes or no target exists
        if phase != self.current_phase or self.target_position is None:
            self.current_phase = phase
            base_altitude = 30.0  # base flight altitude

            if phase == 0:
                # Spiral ascent
                radius = 20.0 * np.random.uniform(0.8, 1.2)
                angle = np.random.uniform(0, 2 * np.pi)
                x = current_pos.x_val + radius * np.cos(angle)
                y = current_pos.y_val + radius * np.sin(angle)
                z = base_altitude + 20.0 * np.random.uniform(0.8, 1.2)

            elif phase == 1:
                # Quick direction change
                direction = np.random.uniform(0, 2 * np.pi)
                distance = 15.0 * np.random.uniform(0.8, 1.2)
                x = current_pos.x_val + distance * np.cos(direction)
                y = current_pos.y_val + distance * np.sin(direction)
                z = current_pos.z_val + np.random.uniform(-5, 5)

            elif phase == 2:
                # Figure-8 pattern segment
                t = time.time() * 0.5
                x = current_pos.x_val + 25 * np.sin(t)
                y = current_pos.y_val + 15 * np.sin(2 * t)
                z = base_altitude + 10 * np.sin(t / 2)

            elif phase == 3:
                # Hover with small variations
                x = current_pos.x_val + np.random.uniform(-2, 2)
                y = current_pos.y_val + np.random.uniform(-2, 2)
                z = current_pos.z_val + np.random.uniform(-1, 1)

            else:
                # Complex curve
                t = time.time() * 0.3
                x = current_pos.x_val + 20 * np.cos(t) * np.sin(t / 2)
                y = current_pos.y_val + 20 * np.sin(t) * np.cos(t / 2)
                z = base_altitude + 15 * np.sin(t / 3)

            self.target_position = np.array([x, y, z])
            self.last_position = current_pos_array

        return self.target_position  # Now always returns a valid numpy array

    def update_movement(self):
        """Update drone movement with varying speeds"""
        target = self.generate_next_waypoint()
        current_pos = self.client.getMultirotorState().kinematics_estimated.position
        current_pos_array = np.array([current_pos.x_val, current_pos.y_val, current_pos.z_val])

        # Calculate direction and distance
        direction = target - current_pos_array  # Now using the array version
        distance = np.linalg.norm(direction)

        if distance > 0.1:
            # Vary speed based on phase and distance
            speed_multiplier = np.clip(distance / 10.0, 0.5, 2.0)
            phase_multiplier = 1.0 + 0.3 * np.sin(time.time() * 2)  # Periodic speed variation
            current_speed = self.base_speed * speed_multiplier * phase_multiplier

            # Add small random variations for more natural movement
            direction += np.random.normal(0, 0.1, 3)

            # Normalize direction and apply speed
            direction = direction / np.linalg.norm(direction) * current_speed

            # Move drone
            self.client.moveByVelocityAsync(
                direction[0],
                direction[1],
                direction[2],
                0.1  # Duration - shorter for more frequent updates
            ).join()

    def collect_frame(self):
        """Collect one frame of data with realistic noise"""
        timestamp = time.time()

        # Get IMU data and add realistic effects
        imu_data = self.client.getImuData()
        imu_data = self.add_realistic_effects(imu_data)
        state = self.client.getMultirotorState()

        # Store IMU data (matching MH_01_easy format)
        imu_frame = {
            'timestamp': timestamp,
            'w_RS_S_x [rad s^-1]': imu_data.angular_velocity.x_val,
            'w_RS_S_y [rad s^-1]': imu_data.angular_velocity.y_val,
            'w_RS_S_z [rad s^-1]': imu_data.angular_velocity.z_val,
            'a_RS_S_x [m s^-2]': imu_data.linear_acceleration.x_val,
            'a_RS_S_y [m s^-2]': imu_data.linear_acceleration.y_val,
            'a_RS_S_z [m s^-2]': imu_data.linear_acceleration.z_val
        }

        # Store ground truth (matching MH_01_easy format)
        ground_truth_frame = {
            'timestamp': timestamp,
            'p_RS_R_x [m]': state.kinematics_estimated.position.x_val,
            'p_RS_R_y [m]': state.kinematics_estimated.position.y_val,
            'p_RS_R_z [m]': state.kinematics_estimated.position.z_val,
            'q_RS_w []': state.kinematics_estimated.orientation.w_val,
            'q_RS_x []': state.kinematics_estimated.orientation.x_val,
            'q_RS_y []': state.kinematics_estimated.orientation.y_val,
            'q_RS_z []': state.kinematics_estimated.orientation.z_val,
            'v_RS_R_x [m s^-1]': state.kinematics_estimated.linear_velocity.x_val,
            'v_RS_R_y [m s^-1]': state.kinematics_estimated.linear_velocity.y_val,
            'v_RS_R_z [m s^-1]': state.kinematics_estimated.linear_velocity.z_val
        }

        self.imu_data.append(imu_frame)
        self.ground_truth.append(ground_truth_frame)
        self.timestamps.append(timestamp)

    def save_data(self):
        """Save collected data in MH_01_easy format"""
        # Save IMU data
        imu_df = pd.DataFrame(self.imu_data)
        imu_df.to_csv(self.session_dir / 'imu0/data.csv', index=False)

        # Save ground truth
        gt_df = pd.DataFrame(self.ground_truth)
        gt_df.to_csv(self.session_dir / 'state_groundtruth_estimate0/data.csv', index=False)

        # Save metadata with detailed parameters
        metadata = {
            'session_name': self.session_dir.name,
            'timestamp': time.strftime('%Y-%m-%d-%H-%M-%S'),
            'n_samples': len(self.timestamps),
            'duration': self.timestamps[-1] - self.timestamps[0],
            'sampling_frequency': self.sampling_freq,
            'environment': 'AirSim Dynamic Flight',
            'maneuver_phases': ['spiral', 'quick_turn', 'figure8', 'hover', 'complex'],
            'noise_parameters': self.noise_params,
            'environmental_parameters': self.env_params
        }

        with open(self.session_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Data saved to {self.session_dir}")

    def start(self):
        """Execute dynamic flight pattern while collecting data"""
        print("Starting dynamic flight data collection...")

        total_samples = int(self.duration * self.sampling_freq)
        sleep_time = 1.0 / self.sampling_freq

        # Take off
        super().start()
        self.phase_start_time = time.time()

        sample_count = 0
        start_time = time.time()

        while sample_count < total_samples:
            self.update_movement()
            self.collect_frame()
            sample_count += 1

            if sample_count % self.sampling_freq == 0:
                elapsed = time.time() - start_time
                print(f"Collected {sample_count}/{total_samples} samples ({elapsed:.1f}s)")
                print(f"Current phase: {self.current_phase}")

            time.sleep(sleep_time)

        print("Data collection complete")
        self.save_data()


if __name__ == "__main__":
    args = sys.argv
    args.pop(0)
    arg_parser = argparse.ArgumentParser("DynamicFlightNavigator.py collects dynamic flight data")
    arg_parser.add_argument("--duration", type=int, help="duration in seconds", default=180)
    arg_parser.add_argument("--freq", type=int, help="sampling frequency in Hz", default=100)
    arg_parser.add_argument("--speed", type=float, help="base speed in meters/second", default=5.0)
    args = arg_parser.parse_args(args)

    navigator = DynamicFlightNavigator(
        duration=args.duration,
        sampling_freq=args.freq,
        base_speed=args.speed
    )
    navigator.start()