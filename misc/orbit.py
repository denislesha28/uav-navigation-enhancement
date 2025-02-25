import setup_path
import airsim
import os
import sys
import math
import time
import argparse
import numpy as np


class Position:
    def __init__(self, pos):
        self.x = pos.x_val
        self.y = pos.y_val
        self.z = pos.z_val


class SafeSurveyNavigator:
    def __init__(self,
                 survey_altitude=50,  # Higher altitude for safety
                 survey_speed=5,  # Moderate speed for stability
                 survey_square_size=100,  # Size of the survey area
                 safety_buffer=10):  # Minimum distance from obstacles

        self.altitude = survey_altitude
        self.speed = survey_speed
        self.square_size = survey_square_size
        self.safety_buffer = safety_buffer
        self.takeoff = False

        # Create survey pattern (square spiral moving outward)
        self.generate_survey_points()

        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        # Store home position
        self.home = self.client.getMultirotorState().kinematics_estimated.position
        self.wait_for_position_stabilization()

    def wait_for_position_stabilization(self):
        """Wait for drone position to stabilize"""
        print("Waiting for position to stabilize...")
        start = time.time()
        count = 0
        while count < 100:
            pos = self.client.getMultirotorState().kinematics_estimated.position
            if abs(pos.z_val - self.home.z_val) > 1:
                count = 0
                self.home = pos
                if time.time() - start > 10:
                    print("Drone position is drifting, waiting to settle...")
                    start = time
            else:
                count += 1

    def generate_survey_points(self):
        """Generate survey waypoints in an expanding square pattern"""
        self.waypoints = []
        x, y = 0, 0
        dx, dy = self.square_size, 0

        # Generate spiral square pattern
        for _ in range(4):
            for _ in range(4):
                x += dx
                y += dy
                self.waypoints.append((x, y))
            dx, dy = -dy, dx  # Rotate 90 degrees

    def check_altitude_safety(self, current_position):
        """Ensure we maintain safe altitude"""
        # Get height of ground below us using ray tracing
        ground_height = self.client.simGetCollisionInfo().impact_point.z_val
        height_above_ground = abs(current_position.z_val - ground_height)

        return height_above_ground >= self.safety_buffer

    def start(self):
        """Begin the survey pattern"""
        print("Arming and taking off...")
        self.client.armDisarm(True)

        # Take off if needed
        landed = self.client.getMultirotorState().landed_state
        if not self.takeoff and landed == airsim.LandedState.Landed:
            self.takeoff = True
            self.client.takeoffAsync().join()

        # Move to survey altitude
        print(f"Climbing to survey altitude: {self.altitude}m")
        takeoff_pos = self.client.getMultirotorState().kinematics_estimated.position
        self.client.moveToPositionAsync(takeoff_pos.x_val, takeoff_pos.y_val,
                                        -self.altitude, self.speed).join()

        # Execute survey pattern
        print("Beginning survey pattern...")
        for waypoint in self.waypoints:
            current_pos = self.client.getMultirotorState().kinematics_estimated.position

            # Safety check
            if not self.check_altitude_safety(current_pos):
                print("Warning: Too close to ground, adjusting altitude...")
                self.client.moveToPositionAsync(current_pos.x_val, current_pos.y_val,
                                                -self.altitude, self.speed).join()

            print(f"Moving to waypoint: {waypoint}")
            self.client.moveToPositionAsync(waypoint[0], waypoint[1], -self.altitude,
                                            self.speed, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(False, 0)).join()

            # Collect data here - can be expanded based on your needs
            self.collect_navigation_data()

        # Return home
        print("Survey complete, returning home...")
        self.client.moveToPositionAsync(self.home.x_val, self.home.y_val,
                                        -self.altitude, self.speed).join()

        if self.takeoff:
            print("Landing...")
            self.client.landAsync().join()
            self.client.armDisarm(False)

    def collect_navigation_data(self):
        """Collect relevant data for navigation enhancement"""
        state = self.client.getMultirotorState()

        # Get various sensor data
        imu_data = self.client.getImuData()
        gps_data = state.gps_location
        position = state.kinematics_estimated.position

        # Store or process the data as needed for your navigation enhancement
        navigation_data = {
            'position': (position.x_val, position.y_val, position.z_val),
            'gps': (gps_data.latitude, gps_data.longitude, gps_data.altitude),
            'imu': {
                'angular_velocity': (imu_data.angular_velocity.x_val,
                                     imu_data.angular_velocity.y_val,
                                     imu_data.angular_velocity.z_val),
                'linear_acceleration': (imu_data.linear_acceleration.x_val,
                                        imu_data.linear_acceleration.y_val,
                                        imu_data.linear_acceleration.z_val)
            }
        }

        # You can save this data or process it in real-time
        # This is where you'd implement your CNN+LNN processing


if __name__ == "__main__":
    args = sys.argv
    args.pop(0)
    arg_parser = argparse.ArgumentParser("SafeSurvey.py performs a safe high-altitude survey pattern")
    arg_parser.add_argument("--altitude", type=float, help="survey altitude in meters", default=50)
    arg_parser.add_argument("--speed", type=float, help="survey speed in meters/second", default=5)
    arg_parser.add_argument("--size", type=float, help="size of survey area", default=100)
    args = arg_parser.parse_args(args)

    navigator = SafeSurveyNavigator(survey_altitude=args.altitude,
                                    survey_speed=args.speed,
                                    survey_square_size=args.size)
    navigator.start()