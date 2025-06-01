import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from orbit import SafeSurveyNavigator


class DataCollectionNavigator(SafeSurveyNavigator):
    def __init__(self,
                 save_dir="airsim_collected_data",
                 session_name="africa_orbit",
                 sampling_freq=100,  # 100 Hz to match typical IMU rates
                 duration=180,  # 3 minutes of data
                 **kwargs):

        super().__init__(**kwargs)
        self.save_dir = Path(save_dir)
        self.session_dir = self.save_dir / session_name
        self.sampling_freq = sampling_freq
        self.duration = duration

        # Create directories
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / 'imu0').mkdir(exist_ok=True)
        (self.session_dir / 'state_groundtruth_estimate0').mkdir(exist_ok=True)

        # Initialize data storage
        self.imu_data = []
        self.ground_truth = []
        self.timestamps = []

    def collect_frame(self):
        """Collect one frame of data in MH_01_easy format"""
        timestamp = time.time()

        # Get IMU data
        imu_data = self.client.getImuData()
        state = self.client.getMultirotorState()

        # Store IMU data (matching MH_01_easy format)
        imu_frame = {
            'timestamp': timestamp,
            'angular_velocity_x': imu_data.angular_velocity.x_val,
            'angular_velocity_y': imu_data.angular_velocity.y_val,
            'angular_velocity_z': imu_data.angular_velocity.z_val,
            'linear_acceleration_x': imu_data.linear_acceleration.x_val,
            'linear_acceleration_y': imu_data.linear_acceleration.y_val,
            'linear_acceleration_z': imu_data.linear_acceleration.z_val
        }

        # Store ground truth (matching MH_01_easy format)
        ground_truth_frame = {
            'timestamp': timestamp,
            'p_RS_R_x': state.kinematics_estimated.position.x_val,
            'p_RS_R_y': state.kinematics_estimated.position.y_val,
            'p_RS_R_z': state.kinematics_estimated.position.z_val,
            'q_RS_w': state.kinematics_estimated.orientation.w_val,
            'q_RS_x': state.kinematics_estimated.orientation.x_val,
            'q_RS_y': state.kinematics_estimated.orientation.y_val,
            'q_RS_z': state.kinematics_estimated.orientation.z_val,
            'v_RS_R_x': state.kinematics_estimated.linear_velocity.x_val,
            'v_RS_R_y': state.kinematics_estimated.linear_velocity.y_val,
            'v_RS_R_z': state.kinematics_estimated.linear_velocity.z_val
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

        # Save metadata
        metadata = {
            'session_name': self.session_dir.name,
            'timestamp': time.strftime('%Y-%m-%d-%H-%M-%S'),
            'n_samples': len(self.timestamps),
            'duration': self.timestamps[-1] - self.timestamps[0],
            'sampling_frequency': self.sampling_freq,
            'environment': 'AirSim Africa',
            'trajectory_type': 'orbit'
        }

        with open(self.session_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Data saved to {self.session_dir}")

    def start(self):
        """Execute orbit pattern while collecting data"""
        print("Starting data collection orbit...")

        # Calculate total samples needed
        total_samples = int(self.duration * self.sampling_freq)
        sleep_time = 1.0 / self.sampling_freq

        # Take off and start orbit
        super().start()

        sample_count = 0
        start_time = time.time()

        while sample_count < total_samples:
            self.collect_frame()
            sample_count += 1

            # Progress update every second
            if sample_count % self.sampling_freq == 0:
                elapsed = time.time() - start_time
                print(f"Collected {sample_count}/{total_samples} samples ({elapsed:.1f}s)")

            time.sleep(sleep_time)

        print("Data collection complete")
        self.save_data()


if __name__ == "__main__":
    args = sys.argv
    args.pop(0)
    arg_parser = argparse.ArgumentParser("DataCollectionNavigator.py collects orbit data in MH_01_easy format")
    arg_parser.add_argument("--altitude", type=float, help="survey altitude in meters", default=50)
    arg_parser.add_argument("--speed", type=float, help="survey speed in meters/second", default=5)
    arg_parser.add_argument("--size", type=float, help="size of survey area", default=100)
    arg_parser.add_argument("--duration", type=int, help="duration in seconds", default=180)
    arg_parser.add_argument("--freq", type=int, help="sampling frequency in Hz", default=100)
    args = arg_parser.parse_args(args)

    navigator = DataCollectionNavigator(
        survey_altitude=args.altitude,
        survey_speed=args.speed,
        survey_square_size=args.size,
        sampling_freq=args.freq,
        duration=args.duration
    )
    navigator.start()