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


def create_feature_vector(aligned_data: Dict[str, pd.DataFrame]) -> np.ndarray:
    """Create feature vector with validation and error handling"""
    if not validate_window_input_data(aligned_data):
        raise ValueError("Invalid input data")

    try:
        # Extract features
        features = [
            # IMU Features (3D)
            aligned_data['IMU_df'][['x', 'y', 'z']].values,

            # Gyro Features (3D)
            aligned_data['RawGyro_df'][['x', 'y', 'z']].values,

            # GPS Features (3D)
            aligned_data['Board_gps_df'][['vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s']].values
        ]

        # Concatenate and validate
        feature_vector = np.concatenate(features, axis=1)

        # Check for NaN values
        if np.isnan(feature_vector).any():
            logging.warning("NaN values found in feature vector")
            # Handle NaN values using forward fill then backward fill
            feature_vector = pd.DataFrame(feature_vector).fillna(method='ffill').fillna(method='bfill').values

        # Log feature vector statistics
        logging.info(f"Feature vector shape: {feature_vector.shape}")
        logging.info(f"Feature stats - Mean: {np.mean(feature_vector):.4f}, Std: {np.std(feature_vector):.4f}")

        return feature_vector

    except Exception as e:
        logging.error(f"Error creating feature vector: {str(e)}")
        raise


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

def create_sliding_windows(aligned_data: Dict[str, pd.DataFrame],
                           error_states: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows with comprehensive validation and error handling"""
    try:
        # Create feature vector
        feature_vector = create_feature_vector(aligned_data)
        n_samples = len(feature_vector)

        # Validate error states
        if not validate_error_states(error_states, n_samples):
            raise ValueError("Invalid error states")

        # Calculate number of windows
        n_windows = ((n_samples - WINDOW_SIZE) // STRIDE) + 1
        logging.info(f"Creating {n_windows} windows")

        # Initialize arrays
        X = np.zeros((n_windows, WINDOW_SIZE, 9))
        y = np.zeros((n_windows, 15))

        # Create windows
        valid_windows = 0
        for i in range(n_windows):
            start_idx = i * STRIDE
            end_idx = start_idx + WINDOW_SIZE

            if end_idx <= n_samples:
                window_data = feature_vector[start_idx:end_idx]

                # Validate window data
                if not np.isnan(window_data).any():
                    X[valid_windows] = window_data

                    # Get corresponding error state
                    _, error_state = error_states[end_idx - 1]
                    if not np.isnan(error_state).any():
                        y[valid_windows] = error_state
                        valid_windows += 1
                    else:
                        logging.warning(f"Invalid error state at index {end_idx - 1}")

        # Trim arrays to valid windows
        X = X[:valid_windows]
        y = y[:valid_windows]

        # Final validation
        if valid_windows == 0:
            raise ValueError("No valid windows created")

        logging.info(f"Successfully created {valid_windows} valid windows")
        logging.info(f"X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    except Exception as e:
        logging.error(f"Error creating sliding windows: {str(e)}")
        raise


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