import numpy as np
import pandas as pd
from scipy import interpolate

def regularize_gps_sampling(gps_df, target_freq_hz=30):
    """
    Regularize GPS sampling to a target frequency using *linear interpolation*.
    Handles NaNs and avoids extrapolation.
    """
    start_time = gps_df.index.min()
    end_time = gps_df.index.max()

    step_ms = (1000 / target_freq_hz)
    regular_times = pd.timedelta_range(
        start=start_time,
        end=end_time,
        freq=pd.Timedelta(milliseconds=step_ms)
    )

    regular_data = {}
    for col in ['vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s']:
        # 1. Convert to numeric and handle non-numeric values:
        gps_df[col] = pd.to_numeric(gps_df[col], errors='coerce')

        # 2. Create a Series with the column data, indexed by time in seconds:
        time_seconds = gps_df.index.total_seconds()
        series = pd.Series(gps_df[col].values, index=time_seconds)

        # 3. Remove NaN values *before* interpolation, keeping track of original indices:
        valid_series = series.dropna()

        # 4. Check for empty series after dropping NaNs. Return early to prevent error in interp1d.
        if valid_series.empty:
          regular_data[col] = np.full(len(regular_times), np.nan) #fill with nans
          continue #skip this column

        # 5. Interpolate (only if there are at least 2 valid points):
        if len(valid_series) >= 2:
            interpolator = interpolate.interp1d(
                valid_series.index,
                valid_series.values,
                kind='linear',
                bounds_error=False,  # Allow filling outside the original range *but we handle it below*
                fill_value=np.nan # fill with NaN
            )

             # Interpolate to the new regular time grid:
            regular_data[col] = interpolator(regular_times.total_seconds())

            # Handle leading/trailing NaNs explicitly (NO EXTRAPOLATION):
            #  - Fill leading NaNs with the first valid value.
            #  - Fill trailing NaNs with the last valid value.
            first_valid_index = np.where(~np.isnan(regular_data[col]))[0][0]
            last_valid_index = np.where(~np.isnan(regular_data[col]))[0][-1]
            regular_data[col][:first_valid_index] = regular_data[col][first_valid_index]
            regular_data[col][last_valid_index+1:] = regular_data[col][last_valid_index]
        elif len(valid_series) == 1:
            # if only one value available, fill with it
            regular_data[col] = np.full(len(regular_times), valid_series.iloc[0])
        # else case already handeled above with the continue



    return pd.DataFrame(regular_data, index=regular_times)


def downsample_regular_gps(regular_gps_df, target_freq_hz=10):
    """
    Downsample regularly sampled GPS data using *averaging*.
    """
    if regular_gps_df.empty:
        return regular_gps_df

    target_freq_str = f"{int(1000 / target_freq_hz)}ms"  # e.g., "100ms" for 10 Hz
    # Use Pandas' resample for robust downsampling with averaging:
    downsampled_df = regular_gps_df.resample(target_freq_str).mean()
    return downsampled_df



def staged_gps_downsample(gps_df, target_timestamps, target_freq_hz=10):
    """
    Two-stage GPS downsampling: first regularize, then downsample,
    then *interpolate* to the target timestamps.
    """
    print("Stage 1: Regularizing GPS sampling...")
    regular_gps = regularize_gps_sampling(gps_df, target_freq_hz=30)
    print(f"\nAfter regularization:")
    print(f"Samples: {len(regular_gps)}")
    if len(regular_gps) > 1:
        print(f"Time intervals mean: {regular_gps.index.to_series().diff().mean()}")
        print(f"Time intervals std: {regular_gps.index.to_series().diff().std()}")

    print(f"\nStage 2: Downsampling to {target_freq_hz} Hz...")
    downsampled_gps = downsample_regular_gps(regular_gps, target_freq_hz=target_freq_hz)

    print(f"\nAfter downsampling:")
    print(f"Samples: {len(downsampled_gps)}")
    if len(downsampled_gps) > 1:
        print(f"Time intervals mean: {downsampled_gps.index.to_series().diff().mean()}")
        print(f"Time intervals std: {downsampled_gps.index.to_series().diff().std()}")
    unique_samples = downsampled_gps.drop_duplicates().shape[0]
    print(f"Unique samples: {unique_samples}")
    print(f"Duplicate ratio: {(len(downsampled_gps) - unique_samples) / len(downsampled_gps):.2%}")

    # Interpolate to target_timestamps using linear interpolation:
    aligned_data = {}
    for col in downsampled_gps.columns:
        interpolator = interpolate.interp1d(
            downsampled_gps.index.to_series().astype(np.int64), #Important: convert to int64 for interp1d
            downsampled_gps[col],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan  # Fill with NaN *outside* the range.
        )
        aligned_data[col] = interpolator(target_timestamps.astype(np.int64))

    aligned_gps = pd.DataFrame(aligned_data, index=target_timestamps)

     #Forward and backward fill to remove NaNs
    aligned_gps = aligned_gps.ffill().bfill()

    return aligned_gps