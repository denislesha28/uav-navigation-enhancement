import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
WINDOW_SIZE = 100
STRIDE = 10


def validate_window_input_data(aligned_data: Dict[str, pd.DataFrame]) -> bool:
    """Validate input data structure and content"""
    required_dfs = ['IMU_df', 'RawGyro_df', 'Board_gps_df']
    required_columns = {
        'IMU_df': ['x', 'y', 'z'],
        'RawGyro_df': ['x', 'y', 'z'],
        'Board_gps_df': ['vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s']
    }

    for df_name in required_dfs:
        if df_name not in aligned_data:
            logging.error(f"Missing required dataframe: {df_name}")
            return False

        df = aligned_data[df_name]
        required_cols = required_columns[df_name]

        if not all(col in df.columns for col in required_cols):
            logging.error(f"Missing required columns in {df_name}")
            return False

        if df.empty:
            logging.error(f"Empty dataframe: {df_name}")
            return False

    return True

def analyze_error_states(error_states: List[Tuple], n_samples: int) -> Dict:
    """
    Perform detailed analysis of error states

    Returns:
        Dict containing analysis results
    """
    analysis = {
        'total_states': len(error_states),
        'expected_states': n_samples,
        'nan_counts': np.zeros(15),  # Count NaNs per dimension
        'dimension_errors': [],
        'value_ranges': [],
        'nan_locations': [],
        'inf_locations': []
    }

    # Check each error state
    for idx, (timestamp, error_state) in enumerate(error_states):
        try:
            # Dimension check
            if len(error_state) != 15:
                analysis['dimension_errors'].append({
                    'index': idx,
                    'found_dim': len(error_state),
                    'expected_dim': 15
                })
                continue

            # NaN check per component
            nan_mask = np.isnan(error_state)
            if nan_mask.any():
                analysis['nan_counts'] += nan_mask
                analysis['nan_locations'].append({
                    'index': idx,
                    'timestamp': timestamp,
                    'components': np.where(nan_mask)[0].tolist()
                })

            # Inf check
            inf_mask = np.isinf(error_state)
            if inf_mask.any():
                analysis['inf_locations'].append({
                    'index': idx,
                    'timestamp': timestamp,
                    'components': np.where(inf_mask)[0].tolist()
                })

            # Value range for non-NaN components
            valid_mask = ~(nan_mask | inf_mask)
            if valid_mask.any():
                valid_values = error_state[valid_mask]
                analysis['value_ranges'].append({
                    'index': idx,
                    'min': np.min(valid_values),
                    'max': np.max(valid_values),
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values)
                })

        except Exception as e:
            logging.error(f"Error analyzing state at index {idx}: {str(e)}")

    # Summarize findings
    analysis['summary'] = {
        'total_nan_states': len(analysis['nan_locations']),
        'total_inf_states': len(analysis['inf_locations']),
        'dimension_error_count': len(analysis['dimension_errors']),
        'missing_states': analysis['expected_states'] - analysis['total_states']
    }

    # Component-wise statistics for valid values
    if analysis['value_ranges']:
        ranges = pd.DataFrame(analysis['value_ranges'])
        analysis['component_stats'] = {
            'min_value': ranges['min'].min(),
            'max_value': ranges['max'].max(),
            'mean_value': ranges['mean'].mean(),
            'std_value': ranges['std'].mean()
        }

    return analysis


def log_error_state_analysis(analysis: Dict):
    """Log detailed analysis results"""
    logging.info("\n=== Error State Analysis ===")

    # Basic counts
    logging.info(f"\nCounts:")
    logging.info(f"Total states: {analysis['total_states']}")
    logging.info(f"Expected states: {analysis['expected_states']}")
    logging.info(f"Missing states: {analysis['summary']['missing_states']}")

    # NaN analysis
    logging.info(f"\nNaN Analysis:")
    logging.info(f"States with NaN: {analysis['summary']['total_nan_states']}")
    logging.info("NaN counts per component:")
    for i, count in enumerate(analysis['nan_counts']):
        if count > 0:
            logging.info(f"Component {i}: {int(count)} NaNs")

    # First few NaN locations
    if analysis['nan_locations']:
        logging.info("\nSample NaN Locations (first 5):")
        for loc in analysis['nan_locations'][:5]:
            logging.info(f"Index {loc['index']}: Components {loc['components']}")

    # Dimension errors
    if analysis['dimension_errors']:
        logging.info(f"\nDimension Errors: {analysis['summary']['dimension_error_count']}")
        logging.info("Sample errors (first 5):")
        for err in analysis['dimension_errors'][:5]:
            logging.info(f"Index {err['index']}: Found {err['found_dim']} dimensions")

    # Value statistics
    if 'component_stats' in analysis:
        logging.info("\nValue Statistics for Valid Components:")
        logging.info(f"Min: {analysis['component_stats']['min_value']:.4f}")
        logging.info(f"Max: {analysis['component_stats']['max_value']:.4f}")
        logging.info(f"Mean: {analysis['component_stats']['mean_value']:.4f}")
        logging.info(f"Std: {analysis['component_stats']['std_value']:.4f}")


def validate_error_states(error_states: List[Tuple], n_samples: int) -> bool:
    """Validate error states with detailed analysis"""
    analysis = analyze_error_states(error_states, n_samples)
    log_error_state_analysis(analysis)

    # Determine if errors are critical
    critical_errors = (
            analysis['summary']['missing_states'] > 0 or
            analysis['summary']['dimension_error_count'] > 0 or
            analysis['summary']['total_inf_states'] > 0 or
            analysis['summary']['total_nan_states'] > n_samples * 0.1  # Allow up to 10% NaN
    )

    if critical_errors:
        logging.error("Critical errors found in error states")
        return False

    if analysis['summary']['total_nan_states'] > 0:
        logging.warning(f"Non-critical NaN values found in {analysis['summary']['total_nan_states']} states")

    return True


def create_sliding_windows(aligned_data, error_states, window_size=100, stride=10):
    """
    Creates sliding windows from aligned sensor data and error states.

    Args:
        aligned_data: Dictionary containing aligned sensor data
        error_states: List of (timestamp, error_vector) tuples
        window_size: Number of samples per window
        stride: Number of samples to shift window

    Returns:
        X: Array of shape (n_windows, window_size, n_features)
        y: Array of shape (n_windows, error_vector_size)
    """
    import logging
    import numpy as np
    import pandas as pd

    logger = logging.getLogger(__name__)

    try:
        # Check if we have enough error states
        if len(error_states) < 1:
            logger.error(f"Not enough error states: {len(error_states)}")
            return np.array([]), np.array([])

        # Create feature vector from sensor data
        feature_vector = create_feature_vector(aligned_data)

        # Handle NaN values in feature vector
        if np.isnan(feature_vector).any():
            logger.warning("NaN values found in feature vector")
            # Convert to DataFrame for easier handling
            feature_vector = pd.DataFrame(feature_vector).ffill().bfill().values

        logger.info(f"Feature vector shape: {feature_vector.shape}")
        logger.info(f"Feature stats - Mean: {feature_vector.mean():.4f}, Std: {feature_vector.std():.4f}")

        # Ensure we have enough samples for at least one window
        if len(feature_vector) < window_size:
            logger.error(f"Not enough samples for window: {len(feature_vector)} < {window_size}")
            return np.array([]), np.array([])

        # Calculate how many windows we can create
        samples_needed = window_size + (stride * (len(error_states) - 1))
        if len(feature_vector) < samples_needed:
            # Adjust number of windows based on available data
            n_windows = (len(feature_vector) - window_size) // stride + 1
            # Further limit based on error states
            n_windows = min(n_windows, len(error_states))
        else:
            n_windows = len(error_states)

        logger.info(f"Creating {n_windows} windows")

        # Create output arrays
        X = np.zeros((n_windows, window_size, feature_vector.shape[1]))
        y = np.zeros((n_windows, len(error_states[0][1])))  # Error vector size

        # Match error states to windows
        error_state_indices = {}

        # Create mapping from timestamps to error state indices
        for i, (timestamp, _) in enumerate(error_states):
            error_state_indices[timestamp] = i

        valid_windows = 0

        # Fill the arrays
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size

            # Check if we have enough data
            if end_idx > len(feature_vector):
                logger.warning(f"Window {i} exceeds feature vector length")
                break

            # Get the corresponding window
            X[valid_windows] = feature_vector[start_idx:end_idx]

            # Get end timestamp - better to use actual timestamps rather than indices
            end_timestamp = aligned_data['IMU_df'].index[end_idx - 1]

            # Find closest error state
            closest_error_idx = find_closest_error_state(end_timestamp, error_states)

            if closest_error_idx is not None:
                _, error_vector = error_states[closest_error_idx]
                # Check for invalid error vector
                if np.any(np.isnan(error_vector)):
                    logger.warning(f"Invalid error state at window {i}")
                    continue

                y[valid_windows] = error_vector
                valid_windows += 1
            else:
                logger.warning(f"No matching error state for window {i}")

        # Return only valid windows
        logger.info(f"Successfully created {valid_windows} valid windows")

        return X[:valid_windows], y[:valid_windows]

    except Exception as e:
        logger.error(f"Error creating sliding windows: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return np.array([]), np.array([])


def find_closest_error_state(timestamp, error_states):
    """
    Find index of error state closest to given timestamp

    Args:
        timestamp: Target timestamp
        error_states: List of (timestamp, error_vector) tuples

    Returns:
        int: Index of closest error state or None if not found
    """
    if not error_states:
        return None

    # Convert timestamps to numeric values for comparison
    if hasattr(timestamp, 'total_seconds'):
        # Convert to seconds
        target_time = timestamp.total_seconds()
    else:
        # Use as is
        target_time = timestamp

    # Find closest timestamp
    min_diff = float('inf')
    closest_idx = None

    for i, (ts, _) in enumerate(error_states):
        if hasattr(ts, 'total_seconds'):
            ts_time = ts.total_seconds()
        else:
            ts_time = ts

        diff = abs(ts_time - target_time)
        if diff < min_diff:
            min_diff = diff
            closest_idx = i

    return closest_idx


def create_feature_vector(aligned_data):
    """
    Create enhanced 12D feature vector including orientation data from OnboardPose

    Args:
        aligned_data: Dictionary containing aligned sensor data

    Returns:
        numpy array of shape (n_samples, 12) containing feature vectors
    """
    import numpy as np

    features = [
        # IMU/Gyro Features (6D)
        aligned_data['IMU_df'][['x', 'y', 'z']].values,
        aligned_data['RawGyro_df'][['x', 'y', 'z']].values,

        # GNSS Features (3D)
        aligned_data['Board_gps_df'][[
            'vel_n_m_s',
            'vel_e_m_s',
            'vel_d_m_s'
        ]].values,

        # OnboardPose Attitude Features (3D) - Using quaternion components
        # Following Chen et al.'s approach of including attitude information
        aligned_data['OnboardPose_df'][[
            'Attitude_w',  # Quaternion scalar component
            'Attitude_x',  # Quaternion x component
            'Attitude_y'  # Quaternion y component
        ]].values
    ]

    # Concatenate all features along the feature dimension (axis=1)
    return np.concatenate(features, axis=1)

def validate_windows(X: np.ndarray, y: np.ndarray) -> Tuple[bool, Dict]:
    """Validate final window data and return statistics"""
    stats = {
        'X_mean': np.mean(X),
        'X_std': np.std(X),
        'X_min': np.min(X),
        'X_max': np.max(X),
        'y_mean': np.mean(y),
        'y_std': np.std(y)
    }

    is_valid = (
            not np.isnan(X).any() and
            not np.isnan(y).any() and
            X.shape[1] == WINDOW_SIZE and
            X.shape[2] == 9 and
            y.shape[1] == 15
    )

    return is_valid, stats