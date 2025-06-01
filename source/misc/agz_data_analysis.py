import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_and_analyze_dataset(data_path):
    """Load and perform basic analysis of the dataset files"""

    # Load the data files
    accel_df = pd.read_csv(data_path / 'RawAccel.csv')
    gyro_df = pd.read_csv(data_path / 'RawGyro.csv')
    gps_df = pd.read_csv(data_path / 'OnboardGPS.csv')
    pose_df = pd.read_csv(data_path / 'OnboardPose.csv')

    analysis_results = {
        "sampling_rates": {},
        "noise_characteristics": {},
        "data_completeness": {},
        "sensor_ranges": {}
    }

    # Calculate sampling rates
    for name, df in [("Accel", accel_df), ("Gyro", gyro_df),
                     ("GPS", gps_df), ("Pose", pose_df)]:
        timestamps = df['Timpstemp'].values
        avg_rate = 1e6 / np.mean(np.diff(timestamps))  # Convert to Hz
        analysis_results["sampling_rates"][name] = avg_rate

    # Analyze IMU noise characteristics
    for axis in ['x', 'y', 'z']:
        # Accelerometer noise
        accel_noise = {
            'mean': accel_df[axis].mean(),
            'std': accel_df[axis].std(),
            'range': (accel_df[axis].min(), accel_df[axis].max())
        }
        analysis_results["noise_characteristics"][f"accel_{axis}"] = accel_noise

        # Gyroscope noise
        gyro_noise = {
            'mean': gyro_df[axis].mean(),
            'std': gyro_df[axis].std(),
            'range': (gyro_df[axis].min(), gyro_df[axis].max())
        }
        analysis_results["noise_characteristics"][f"gyro_{axis}"] = gyro_noise

    # Check data completeness
    for name, df in [("Accel", accel_df), ("Gyro", gyro_df),
                     ("GPS", gps_df), ("Pose", pose_df)]:
        missing = df.isnull().sum()
        analysis_results["data_completeness"][name] = {
            "total_samples": len(df),
            "missing_values": missing.to_dict()
        }

    # Analyze sensor ranges
    analysis_results["sensor_ranges"] = {
        "accel_range": accel_df['range_rad_s'].iloc[0],
        "gyro_range": gyro_df['range_rad_s'].iloc[0],
        "gps_accuracy": {
            "mean_eph": gps_df['eph_m'].mean(),
            "mean_epv": gps_df['epv_m'].mean()
        }
    }

    return analysis_results


def print_analysis(results):
    """Print analysis results in a readable format"""
    print("\n=== Dataset Analysis Results ===\n")

    print("Sampling Rates (Hz):")
    for sensor, rate in results["sampling_rates"].items():
        print(f"{sensor}: {rate:.2f} Hz")

    print("\nNoise Characteristics:")
    for sensor, noise in results["noise_characteristics"].items():
        print(f"\n{sensor}:")
        print(f"  Mean: {noise['mean']:.6f}")
        print(f"  Std: {noise['std']:.6f}")
        print(f"  Range: {noise['range']}")

    print("\nData Completeness:")
    for sensor, stats in results["data_completeness"].items():
        print(f"\n{sensor}:")
        print(f"  Total samples: {stats['total_samples']}")
        if any(stats['missing_values'].values()):
            print("  Missing values:", stats['missing_values'])

    print("\nSensor Ranges:")
    print(f"Accelerometer range: {results['sensor_ranges']['accel_range']}")
    print(f"Gyroscope range: {results['sensor_ranges']['gyro_range']}")
    print("GPS accuracy:")
    print(f"  Mean EPH: {results['sensor_ranges']['gps_accuracy']['mean_eph']:.2f} m")
    print(f"  Mean EPV: {results['sensor_ranges']['gps_accuracy']['mean_epv']:.2f} m")


if __name__ == "__main__":
    # Update this path to your dataset location
    data_path = Path("../AGZ_subset/Log Files")

    try:
        results = load_and_analyze_dataset(data_path)
        print_analysis(results)
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")