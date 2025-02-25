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


class UrbanNavigator:
    def __init__(self,
                 min_altitude=100,  # Increased minimum altitude significantly
                 cruise_altitude=120,  # Higher cruise altitude
                 speed=3,  # Reduced speed for safety
                 square_size=80,  # Larger squares to stay away from buildings
                 building_buffer=20):  # Increased building buffer

        self.min_altitude = min_altitude
        self.cruise_altitude = cruise_altitude
        self.speed = speed
        self.square_size = square_size
        self.building_buffer = building_buffer
        self.takeoff = False

        # Initialize AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        # Store home position
        self.home = self.client.getMultirotorState().kinematics_estimated.position
        self.wait_for_position_stabilization()

        # Start position (moved away from buildings)
        self.start_x = 0
        self.start_y = 0

        # Generate urban navigation pattern
        self.generate_urban_waypoints()

    def wait_for_position_stabilization(self):
        print("Waiting for position to stabilize...")
        start = time.time()
        count = 0
        while count < 100:
            pos = self.client.getMultirotorState().kinematics_estimated.position
            if abs(pos.z_val - self.home.z_val) > 1:
                count = 0
                self.home = pos
                if time.time() - start > 10:
                    print("Position drifting, waiting to settle...")
                    start = time
            else:
                count += 1

    def generate_urban_waypoints(self):
        """Generate safer waypoints for urban navigation"""
        self.waypoints = []
        grid_size = 3  # Reduced grid size for safety

        # Generate a safer grid pattern
        for i in range(grid_size):
            for j in range(grid_size):
                # Alternate between two high altitudes for safety
                altitude = self.cruise_altitude if (i + j) % 2 == 0 else self.min_altitude

                # Create waypoint with offset from start position
                waypoint = (
                    self.start_x + i * self.square_size,
                    self.start_y + j * self.square_size,
                    -altitude  # Negative because AirSim uses NED coordinates
                )
                self.waypoints.append(waypoint)

    def start(self):
        """Execute urban navigation pattern with additional safety checks"""
        print("Starting urban navigation sequence...")
        self.client.armDisarm(True)

        # Initial takeoff
        landed = self.client.getMultirotorState().landed_state
        if not self.takeoff and landed == airsim.LandedState.Landed:
            self.takeoff = True
            print("Taking off...")
            self.client.takeoffAsync().join()

        # First climb to safe altitude before any lateral movement
        print(f"Climbing to safe altitude: {self.cruise_altitude}m")
        current_pos = self.client.getMultirotorState().kinematics_estimated.position

        # Vertical climb first
        self.client.moveToPositionAsync(
            current_pos.x_val,
            current_pos.y_val,
            -self.cruise_altitude,  # Negative for AirSim NED coordinates
            self.speed
        ).join()

        print("Reached safe altitude, beginning urban navigation pattern...")

        # Execute urban navigation pattern
        for i, waypoint in enumerate(self.waypoints):
            print(f"Moving to waypoint {i + 1}/{len(self.waypoints)}: {waypoint}")

            # Move to waypoint
            self.client.moveToPositionAsync(
                waypoint[0],  # x
                waypoint[1],  # y
                waypoint[2],  # z (altitude)
                self.speed,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0)
            ).join()

            # Collect navigation data at each waypoint
            nav_data = self.collect_urban_navigation_data()

            # Pause at each waypoint
            time.sleep(2)

        # Return home at safe altitude
        print("Navigation complete, returning to start...")
        self.client.moveToPositionAsync(
            self.home.x_val,
            self.home.y_val,
            -self.cruise_altitude,
            self.speed
        ).join()

        if self.takeoff:
            print("Descending...")
            self.client.moveToPositionAsync(
                self.home.x_val,
                self.home.y_val,
                -10,  # Safe descent altitude
                1  # Slower speed for descent
            ).join()

            print("Landing...")
            self.client.landAsync().join()
            self.client.armDisarm(False)

    def collect_urban_navigation_data(self):
        """Collect navigation data"""
        state = self.client.getMultirotorState()
        gps_data = state.gps_location
        position = state.kinematics_estimated.position

        navigation_data = {
            'timestamp': time.time(),
            'position': {
                'x': position.x_val,
                'y': position.y_val,
                'z': position.z_val
            },
            'gps': {
                'lat': gps_data.latitude,
                'lon': gps_data.longitude,
                'alt': gps_data.altitude
            }
        }
        return navigation_data


if __name__ == "__main__":
    args = sys.argv
    args.pop(0)
    arg_parser = argparse.ArgumentParser("UrbanNavigator.py performs safe urban environment navigation")
    arg_parser.add_argument("--min_altitude", type=float, help="minimum safe altitude", default=100)
    arg_parser.add_argument("--cruise_altitude", type=float, help="preferred cruise altitude", default=120)
    arg_parser.add_argument("--speed", type=float, help="navigation speed", default=3)
    arg_parser.add_argument("--square_size", type=float, help="size of grid squares", default=80)
    args = arg_parser.parse_args(args)

    navigator = UrbanNavigator(
        min_altitude=args.min_altitude,
        cruise_altitude=args.cruise_altitude,
        speed=args.speed,
        square_size=args.square_size
    )
    navigator.start()