"""
imu_data_processor.py

A simpler IMU data processor that:
 1) Loads IMU & pose CSV
 2) Optionally finds a single-axis shift to align signals
 3) Shifts pose timestamps
 4) Resamples both dataframes to a fixed frequency
 5) Returns the processed IMU dataframe (and optionally the pose df)

We omit complex steps like Madgwick orientation, gravity removal,
and advanced bias corrections for clarity.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Union, Optional, Dict, Any

import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags


def _simple_preprocess(df: pd.DataFrame, timestamp_col: str = "Timpstemp") -> pd.DataFrame:
    """
    1) Drop NaNs
    2) Sort by timestamp
    3) Convert 'timestamp_col' -> Timedelta index
    """
    df = df.dropna()
    df = df.copy()  # ensure no SettingWithCopy issues

    start_time = df[timestamp_col].min()
    df.index = pd.to_timedelta(df[timestamp_col] - start_time, unit='us')
    df = df.sort_index()
    return df


def _find_time_shift(
    imu_df: pd.DataFrame,
    pose_df: pd.DataFrame,
    imu_axis: str = "x",
    pose_axis: str = "Omega_x",
    sampling_rate: float = 100.0,
    logger: Optional[logging.Logger] = None
) -> int:
    """
    Find a single best shift (in microseconds) by cross-correlating
    imu_df[imu_axis] with pose_df[pose_axis].
    """
    if imu_axis not in imu_df.columns or pose_axis not in pose_df.columns:
        if logger:
            logger.info("Cannot find shift. Missing required columns.")
        return 0

    imu_signal = imu_df[imu_axis].values
    pose_signal = pose_df[pose_axis].values

    if len(imu_signal) == 0 or len(pose_signal) == 0:
        return 0

    c = correlate(imu_signal, pose_signal, mode='full')
    lags = correlation_lags(len(imu_signal), len(pose_signal), mode='full')

    best_idx = np.argmax(abs(c))
    best_lag_samples = lags[best_idx]
    sample_period_us = 1e6 / sampling_rate
    shift_us = int(best_lag_samples * sample_period_us)

    if logger:
        logger.info(f"Best shift = {shift_us} us, lag in samples = {best_lag_samples}")

    return shift_us


class IMUDataProcessor:
    """
    A simpler IMU data processor.
    """

    def __init__(
        self,
        sampling_rate: float = 100.0,
        timestamp_tolerance_ms: float = 20.0,
        enable_logging: bool = True,
        imu_axis_for_shift: str = "x",
        pose_axis_for_shift: str = "Omega_x",
        visualize_correlation: bool = True
    ):
        """
        Parameters
        ----------
        sampling_rate : float
            IMU sampling frequency (Hz).
        timestamp_tolerance_ms : float
            Tolerance for final alignment in ms.
        enable_logging : bool
            If True, logs at INFO level.
        imu_axis_for_shift : str
            Axis name in IMU dataframe to use for correlation-based shift.
        pose_axis_for_shift : str
            Axis name in Pose dataframe to use for correlation-based shift.
        visualize_correlation : bool
            If True, show a before/after correlation plot to confirm shift alignment.
        """
        self.sampling_rate = sampling_rate
        self.tolerance_us = timestamp_tolerance_ms * 1000.0
        self.imu_axis_for_shift = imu_axis_for_shift
        self.pose_axis_for_shift = pose_axis_for_shift
        self.visualize_correlation = visualize_correlation

        if enable_logging:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.stats: Dict[str, Any] = {
            "processed_samples": 0,
            "alignment_issues": 0,
        }

    def process_dataset(
        self,
        imu_path: Union[str, Path],
        pose_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Steps:
         1) Load CSV data
         2) Basic preprocessing => Timedelta index, drop NaNs, etc.
         3) Find single-axis shift
         4) Shift pose timestamps
         5) Nearest reindex within tolerance
         6) Resample to a fixed freq (100Hz by default)
         7) (Optional) visualize correlation shift
         8) Return processed IMU DataFrame
        """
        # 1) Load CSV
        try:
            imu_df = pd.read_csv(imu_path)
            pose_df = pd.read_csv(pose_path)
        except Exception as e:
            self.logger.error(f"Failed to read CSVs: {e}")
            raise

        # 2) Basic preprocessing
        imu_df.columns = imu_df.columns.str.strip()
        pose_df.columns = pose_df.columns.str.strip()

        imu_df = _simple_preprocess(imu_df, "Timpstemp")
        pose_df = _simple_preprocess(pose_df, "Timpstemp")

        # 3) Find single-axis shift
        shift_us = _find_time_shift(
            imu_df,
            pose_df,
            imu_axis=self.imu_axis_for_shift,
            pose_axis=self.pose_axis_for_shift,
            sampling_rate=self.sampling_rate,
            logger=self.logger
        )

        # Apply shift
        if shift_us != 0:
            pose_df["Timpstemp"] += -shift_us
            pose_df = _simple_preprocess(pose_df, "Timpstemp")

        # 5) Nearest reindex within tolerance
        imu_aligned, pose_aligned = self._final_align(imu_df, pose_df)

        # 6) Resample to 100 Hz
        imu_aligned = self._resample_fixed_freq(imu_aligned, freq_hz=100)
        pose_aligned = self._resample_fixed_freq(pose_aligned, freq_hz=100)

        # Update stats
        self.stats["processed_samples"] = len(imu_aligned)

        # 7) Optionally save
        if output_path:
            imu_aligned.to_csv(output_path)
            self.logger.info(f"Saved processed IMU data to {output_path}")

        # Return only IMU for simplicity (though you could return both if needed)
        return imu_aligned

    def _final_align(
        self, imu_df: pd.DataFrame, pose_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Final alignment by nearest reindex within 'tolerance_us'.
        """
        common_start = max(imu_df.index.min(), pose_df.index.min())
        common_end = min(imu_df.index.max(), pose_df.index.max())

        imu_df = imu_df[common_start:common_end]
        pose_df = pose_df[common_start:common_end]

        aligned_pose = pose_df.reindex(
            imu_df.index,
            method='nearest',
            tolerance=pd.Timedelta(microseconds=self.tolerance_us)
        )

        valid_mask = ~aligned_pose.isna().any(axis=1)
        imu_aligned = imu_df[valid_mask]
        pose_aligned = aligned_pose[valid_mask]

        dropped = len(imu_df) - len(imu_aligned)
        self.stats["alignment_issues"] = dropped
        if dropped > 0:
            self.logger.info(
                f"Alignment: Dropped {dropped} samples outside tolerance ({self.tolerance_us} us)."
            )

        return imu_aligned, pose_aligned

    def _resample_fixed_freq(
        self, df: pd.DataFrame, freq_hz: float = 100.0
    ) -> pd.DataFrame:
        """
        Resample 'df' to a fixed freq_hz (default=100Hz).
        We'll do nearest reindex with a small tolerance (e.g. 1 period).
        """
        if df.empty:
            return df

        start, end = df.index.min(), df.index.max()
        if start == end:
            return df

        period_sec = 1.0 / freq_hz  # e.g., 0.01 s for 100 Hz
        freq_timedelta = pd.Timedelta(seconds=period_sec)

        # Build a new uniform index
        new_index = pd.timedelta_range(start, end, freq=freq_timedelta)

        # Nearest reindex
        df_resampled = df.reindex(new_index, method='nearest', tolerance=freq_timedelta)
        df_resampled = df_resampled.dropna()

        return df_resampled

    def get_statistics(self) -> Dict[str, Any]:
        return self.stats
