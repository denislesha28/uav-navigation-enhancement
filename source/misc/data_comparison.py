import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json


class DatasetAnalyzer:
    """
    Analyzes and compares IMU datasets collected from AirSim with MH_01_easy dataset
    Based on methodology from Chen et al. (2024) for data validation
    """

    def __init__(self, airsim_data_path, mh01_data_path):
        self.airsim_path = Path(airsim_data_path)
        self.mh01_path = Path(mh01_data_path)

        # Define column mapping from AirSim to MH01 format
        self.column_mapping = {
            'angular_velocity_x': 'w_RS_S_x [rad s^-1]',
            'angular_velocity_y': 'w_RS_S_y [rad s^-1]',
            'angular_velocity_z': 'w_RS_S_z [rad s^-1]',
            'linear_acceleration_x': 'a_RS_S_x [m s^-2]',
            'linear_acceleration_y': 'a_RS_S_y [m s^-2]',
            'linear_acceleration_z': 'a_RS_S_z [m s^-2]'
        }

        # Load and initialize data
        self._initialize_data()

    def _initialize_data(self):
        """Initialize and load all required datasets"""
        # Load IMU data
        airsim_imu_raw = pd.read_csv(self.airsim_path / 'imu0/data.csv')
        self.airsim_imu = self._standardize_airsim_columns(airsim_imu_raw)
        self.mh01_imu = pd.read_csv(self.mh01_path / 'imu0/data.csv')

        # Load ground truth data
        self.airsim_gt = pd.read_csv(self.airsim_path / 'state_groundtruth_estimate0/data.csv')
        self.mh01_gt = pd.read_csv(self.mh01_path / 'state_groundtruth_estimate0/data.csv')

        # Load metadata if available
        try:
            with open(self.airsim_path / 'metadata.json', 'r') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            self.metadata = None

    def _standardize_airsim_columns(self, df):
        """Convert AirSim column names to match MH01 format"""
        df = df.copy()
        for airsim_col, mh01_col in self.column_mapping.items():
            if airsim_col in df.columns:
                df[mh01_col] = df[airsim_col]
                df.drop(columns=[airsim_col], inplace=True)
        return df

        # Load datasets
        # Load and standardize column names for IMU data
        self.airsim_imu = pd.read_csv(self.airsim_path / 'imu0/data.csv')
        self.airsim_imu = self._standardize_airsim_columns(self.airsim_imu)

        self.mh01_imu = pd.read_csv(self.mh01_path / 'imu0/data.csv')

        # Load ground truth data
        self.airsim_gt = pd.read_csv(self.airsim_path / 'state_groundtruth_estimate0/data.csv')
        self.mh01_gt = pd.read_csv(self.mh01_path / 'state_groundtruth_estimate0/data.csv')

        # Load metadata if available
        try:
            with open(self.airsim_path / 'metadata.json', 'r') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            self.metadata = None

    def compute_statistics(self, data):
        """Compute basic statistics for IMU measurements"""
        stats_dict = {
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'sampling_rate': len(data) / (data['timestamp'].max() - data['timestamp'].min())
        }
        return pd.DataFrame(stats_dict)

    def analyze_frequency_distribution(self, imu_data):
        """Analyze the frequency content of IMU signals"""
        # Calculate sampling frequency
        dt = np.diff(imu_data['timestamp']).mean()
        fs = 1 / dt

        # Compute FFT for accelerometer and gyroscope data
        freq_analysis = {}
        for col in imu_data.columns:
            if col.startswith(('angular_velocity', 'linear_acceleration')):
                signal = imu_data[col].values
                freq = np.fft.fftfreq(len(signal), d=dt)
                fft = np.abs(np.fft.fft(signal))
                freq_analysis[col] = {'freq': freq, 'magnitude': fft}

        return freq_analysis, fs

    def compare_datasets(self):
        """Compare statistical properties of both datasets"""
        airsim_stats = self.compute_statistics(self.airsim_imu)
        mh01_stats = self.compute_statistics(self.mh01_imu)

        comparison = pd.DataFrame({
            'AirSim': airsim_stats['mean'],
            'MH01': mh01_stats['mean'],
            'Difference %': ((airsim_stats['mean'] - mh01_stats['mean']) / mh01_stats['mean']) * 100
        })

        return comparison

    def plot_comparisons(self):
        """Generate comparison plots between datasets"""
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('IMU Data Comparison: AirSim vs MH_01_easy')

        # Accelerometer comparisons
        for i, axis in enumerate(['x', 'y', 'z']):
            col = f'a_RS_S_{axis} [m s^-2]'
            axs[0, i].hist(self.airsim_imu[col], bins=50, alpha=0.5, label='AirSim')
            axs[0, i].hist(self.mh01_imu[col], bins=50, alpha=0.5, label='MH01')
            axs[0, i].set_title(f'Acceleration {axis}')
            axs[0, i].legend()

        # Gyroscope comparisons
        for i, axis in enumerate(['x', 'y', 'z']):
            col = f'w_RS_S_{axis} [rad s^-1]'
            axs[1, i].hist(self.airsim_imu[col], bins=50, alpha=0.5, label='AirSim')
            axs[1, i].hist(self.mh01_imu[col], bins=50, alpha=0.5, label='MH01')
            axs[1, i].set_title(f'Angular Velocity {axis}')
            axs[1, i].legend()

        plt.tight_layout()
        return fig

    def validate_data_quality(self):
        """
        Validate if the collected data meets requirements for LNN training
        Based on criteria from Chen et al. (2024) and project requirements
        """
        validation_results = {
            'sampling_rate_valid': False,
            'range_coverage_valid': False,
            'noise_characteristics_valid': False,
            'overall_valid': False,
            'details': {}
        }

        # Check sampling rate
        airsim_dt = np.diff(self.airsim_imu['timestamp']).mean()
        airsim_fs = 1 / airsim_dt
        validation_results['sampling_rate_valid'] = bool(abs(airsim_fs - 100) < 5)  # Should be close to 100Hz
        validation_results['details']['sampling_rate'] = float(airsim_fs)

        # Check sensor range coverage
        range_coverage = {}
        for col in self.column_mapping.values():
            if col in self.airsim_imu.columns:
                airsim_range = float(self.airsim_imu[col].max() - self.airsim_imu[col].min())
                mh01_range = float(self.mh01_imu[col].max() - self.mh01_imu[col].min())
                coverage_ratio = airsim_range / mh01_range if mh01_range != 0 else 0
                range_coverage[col] = coverage_ratio

        validation_results['range_coverage_valid'] = bool(all(ratio >= 0.5 for ratio in range_coverage.values()))
        validation_results['details']['range_coverage'] = range_coverage

        # Check noise characteristics
        noise_characteristics = {}
        for col in self.column_mapping.values():
            if col in self.airsim_imu.columns:
                # Compare distribution shapes using KS test
                _, p_value = stats.ks_2samp(self.airsim_imu[col], self.mh01_imu[col])
                noise_characteristics[col] = float(p_value)

        validation_results['noise_characteristics_valid'] = bool(
            all(p_value >= 0.01 for p_value in noise_characteristics.values()))
        validation_results['details']['noise_characteristics'] = noise_characteristics

        # Overall validation
        validation_results['overall_valid'] = bool(all([
            validation_results['sampling_rate_valid'],
            validation_results['range_coverage_valid'],
            validation_results['noise_characteristics_valid']
        ]))

        return validation_results

    def generate_report(self):
        """Generate a comprehensive analysis report"""
        report = {
            'metadata': self.metadata,
            'statistics_comparison': self.compare_datasets(),
            'validation_results': self.validate_data_quality()
        }

        return report


def main():
    # Example usage
    analyzer = DatasetAnalyzer(
        airsim_data_path='../airsim_collected_data/dynamic_flight',
        mh01_data_path='MH_01_easy/mav0'
    )

    # Generate and print report
    report = analyzer.generate_report()
    print("\nDataset Analysis Report")
    print("=====================")
    print("\nMetadata:", json.dumps(report['metadata'], indent=2))
    print("\nStatistics Comparison:")
    print(report['statistics_comparison'])
    print("\nValidation Results:", json.dumps(report['validation_results'], indent=2))

    # Generate plots
    fig = analyzer.plot_comparisons()
    plt.show()


if __name__ == "__main__":
    main()