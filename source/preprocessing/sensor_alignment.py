import pandas as pd

from preprocessing.gps_downsampler import resample_ground_truth_with_direct_mapping, staged_gps_downsample


def align_sensors_multi_freq(df_dict, target_freq='100ms'):
    """
    Align sensors with possibly different sampling rates.
    """

    # 1. Find common start/end among all dataframes
    common_start = max(df.index.min() for df in df_dict.values() if not df.empty)
    common_end = min(df.index.max() for df in df_dict.values() if not df.empty)

    # 2. Round start & end to multiples of target_freq to avoid float issues
    #    Convert target_freq to a Timedelta
    freq_td = pd.to_timedelta(target_freq)

    #    Snap common_start DOWN to nearest multiple of freq_td
    #    Snap common_end   UP   to nearest multiple of freq_td
    #    (only if you'd like to ensure the entire range is covered)
    def round_down_to_freq(t, freq_td):
        # E.g. floor division in integer microseconds
        n_us = t // pd.Timedelta('1us')  # total microseconds
        freq_us = freq_td // pd.Timedelta('1us')  # freq in microseconds
        rounded_us = (n_us // freq_us) * freq_us
        return pd.Timedelta(rounded_us, unit='us')

    def round_up_to_freq(t, freq_td):
        n_us = t // pd.Timedelta('1us')
        freq_us = freq_td // pd.Timedelta('1us')
        # if there's a remainder, add one chunk
        if (n_us % freq_us) != 0:
            rounded_us = ((n_us // freq_us) + 1) * freq_us
        else:
            rounded_us = n_us
        return pd.Timedelta(rounded_us, unit='us')

    common_start_rounded = round_down_to_freq(common_start, freq_td)
    common_end_rounded = round_up_to_freq(common_end, freq_td)

    # 3. Create target_timestamps using these rounded times
    target_timestamps = pd.timedelta_range(
        start=common_start_rounded,
        end=common_end_rounded,
        freq=target_freq
    )

    # 4. Align each DF to the target_timestamps
    aligned_dict = {}
    for sensor_name, df in df_dict.items():
        if df.empty:
            aligned_dict[sensor_name] = df
            continue

        df_trimmed = df.loc[common_start:common_end].copy()

        if sensor_name == 'Board_gps_df':
            df_aligned = staged_gps_downsample(df_trimmed, target_timestamps, 10)

        elif sensor_name == "Ground_truth_df":
            pass
            # df_aligned = resample_ground_truth_with_direct_mapping(
            #     ground_truth_df=df_trimmed,
            #     target_timestamps=target_timestamps
            # )

        else:
            #Keep using nearest for other sensors
            df_aligned = df_trimmed.reindex(
                target_timestamps,
                method='nearest',
                tolerance=pd.Timedelta('50ms')
            )

        aligned_dict[sensor_name] = df_aligned

        # Print your metrics as before...
        original_samples = len(df_trimmed)
        aligned_samples = len(df_aligned)
        unique_samples = df_aligned.drop_duplicates().shape[0]
        print(f"\n{sensor_name} alignment metrics:")
        print(f"Original samples: {original_samples}")
        print(f"Aligned samples:  {aligned_samples}")
        print(f"Unique samples:   {unique_samples}")
        print(f"Duplicate ratio:  {(aligned_samples - unique_samples) / aligned_samples:.2%}")

    return aligned_dict


def validate_alignment(aligned_dict, target_freq='100ms'):
    """
    Validate alignment results.
    """
    print("\nAlignment Validation:")
    for sensor_name, df in aligned_dict.items():
        if df.empty or len(df) < 2:
            continue

        time_diffs = df.index.to_series().diff()
        expected_diff = pd.Timedelta(target_freq)

        print(f"\n{sensor_name} temporal statistics:")
        print(f"Mean interval: {time_diffs.mean()}")
        print(f"Std interval: {time_diffs.std()}")
        print(f"Max interval: {time_diffs.max()}")

        # Identify any large gaps
        gaps = time_diffs[time_diffs > expected_diff * 1.5]
        if not gaps.empty:
            print(f"Found {len(gaps)} significant gaps")
            print(f"Largest gap: {gaps.max()}")

