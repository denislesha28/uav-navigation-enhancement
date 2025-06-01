#!/usr/bin/env python
"""
Script to generate scientific publication-quality plots for LTC model evaluation
using saved .pth model files for 5-fold cross-validation.

This script:
1. Loads saved LTC models from .pth files (one for each cross-validation fold)
2. Evaluates each model on test data
3. Generates scientific-quality plots according to publication standards
4. Compares with standard Kalman filter performance

Usage:
    python run_scientific_evaluation.py --model_dir models/ltc_kfold --data_path data/test_data.npz --output_dir results/ltc_plots
"""

import glob
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Subset, DataLoader, TensorDataset

from data_loader import load_data
from kalman_filtering.kf_vector_calc import calc_error_state_vector
from preprocessing.sliding_windows import create_sliding_windows
from training.cnn_feature_extraction import CNNFeatureExtractor
from training.ltc.ltc_plotting import LTCKFoldPlotter

# Import custom modules

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_test_data_with_indices(data_path, model_dir, fold_idx, device):
    """
    Load test data for a specific fold using saved indices

    Args:
        data_path: Path to the data file
        model_dir: Directory containing saved model files and indices
        fold_idx: Current fold index (1-based)
        device: Torch device

    Returns:
        DataLoader for test data
    """
    logger.info(f"Loading test data for fold {fold_idx} from {data_path}")

    try:
        # Load full data
        data = np.load(data_path)

        # Extract features and targets
        if 'X' in data and 'y' in data:
            X = data['X']
            y = data['y']
        else:
            logger.error("Data file must contain 'X' and 'y' arrays")
            raise ValueError("Invalid data format")

        # Create full PyTorch dataset
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        full_dataset = TensorDataset(X_tensor, y_tensor)

        # Try to load saved test indices
        indices_path = Path(model_dir) / f"fold_{fold_idx}_test_indices.npy"

        if indices_path.exists():
            # Option 1: Use saved indices
            logger.info(f"Loading test indices from {indices_path}")
            test_idx = np.load(indices_path)
            test_dataset = Subset(full_dataset, test_idx)
            logger.info(f"Created test dataset using saved indices: {len(test_dataset)} samples")
        else:
            # Option 2: Recreate split using same random seed
            logger.warning(f"No saved indices found at {indices_path}. Recreating splits.")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            # Get the test indices for this fold
            all_indices = list(range(len(full_dataset)))
            test_idx = None

            for i, (_, fold_test_idx) in enumerate(kf.split(all_indices)):
                if i + 1 == fold_idx:
                    test_idx = fold_test_idx
                    break

            if test_idx is None:
                raise ValueError(f"Could not recreate test indices for fold {fold_idx}")

            test_dataset = Subset(full_dataset, test_idx)
            logger.info(f"Created test dataset using recreated indices: {len(test_dataset)} samples")

        # Create DataLoader
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        return test_loader, (X.shape[-1], y.shape[-1])

    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise e


def compute_baseline_metrics(X, y):
    """
    Compute baseline metrics using standard Kalman Filter

    Args:
        X: Test input data
        y: Test target data

    Returns:
        Dictionary with baseline metrics
    """


    logger.info("Computing baseline metrics using standard Kalman Filter")

    # Import UAVNavigationKF from local module
    from kalman_filtering.uav_navigation_kf import UAVNavigationKF

    # Initialize Kalman filter
    kf = UAVNavigationKF(dt=0.1)  # Assuming 10Hz sampling

    # Use subset of data for faster computation if needed

    print(X)
    # Initialize array to store KF predictions
    kf_predictions = np.zeros_like(y)

    # For each sample, run KF and store predictions
    for i in range(48):
        # Prediction step
        kf.predict()

        # Extract measurement values (using last timestep from sequence)
        measurement = X[i, :9]  # First 9 features are sensor readings
        # Update KF with measurement
        kf.update(measurement)

        # Get the KF state
        state = kf.get_state()

        # Map to error state vector format (15D)
        error_state = np.zeros(15)

        # Map components as needed
        error_state[0:3] = 0.01  # Small attitude errors
        error_state[3:6] = state[3:6]  # Velocity components
        error_state[6:9] = state[0:3]  # Position components
        error_state[9:15] = 0.01  # Small bias errors

        # Store prediction

        kf_predictions[i] = error_state

    # Calculate metrics
    from sklearn.metrics import mean_squared_error

    # Calculate 2D and Z errors
    e_2d = np.sqrt((y[:, 6] - kf_predictions[:, 6]) ** 2 +
                   (y[:, 7] - kf_predictions[:, 7]) ** 2)

    e_z = np.abs(y[:, 8] - kf_predictions[:, 8])

    baseline_metrics = {
        'rmse': np.sqrt(mean_squared_error(y, kf_predictions)),
        '2d_error_mean': np.mean(e_2d),
        '2d_error_std': np.std(e_2d),
        'z_error_mean': np.mean(e_z),
        'z_error_std': np.std(e_z)
    }

    logger.info(f"Baseline metrics: RMSE={baseline_metrics['rmse']:.4f}, " +
                f"2D Error={baseline_metrics['2d_error_mean']:.4f}±{baseline_metrics['2d_error_std']:.4f}, " +
                f"Z Error={baseline_metrics['z_error_mean']:.4f}±{baseline_metrics['z_error_std']:.4f}")

    return baseline_metrics


def find_model_files(model_dir):
    """
    Find all .pth model files in the specified directory

    Args:
        model_dir: Directory containing model files

    Returns:
        List of paths to model files
    """
    model_dir = Path(model_dir)
    model_paths = sorted(glob.glob(str(model_dir / "fold_*_model.pth")))

    if not model_paths:
        # Try alternative naming pattern
        model_paths = sorted(glob.glob(str(model_dir / "*.pth")))

    logger.info(f"Found {len(model_paths)} model files: {model_paths}")
    return model_paths


def main():
    device = "cuda"
    logger.info(f"Using device: {device}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1) Load data via IMUDataProcessor and other logs
    df_data = load_data("../../")
    aligned_data = df_data

    error_states = calc_error_state_vector(aligned_data)
    X, y = create_sliding_windows(aligned_data, error_states)

    cnn = CNNFeatureExtractor()
    X_spatial = torch.from_numpy(np.transpose(X, (0, 2, 1)))
    X_spatial = cnn.debug_forward(X_spatial)

    # Initialize scientific plot generator
    plot_generator = LTCKFoldPlotter(save_dir="ltc_fold_visualizations")

    # Evaluate models
    fold_results = plot_generator.evaluate_models(
        index_paths=[f"../models/fold_{i + 1}_test_indices.npy" for i in range(5)],
        model_paths=[f"../models/best_ltc_{i}.pth" for i in range(5)],
        X_tensor=X_spatial,
        device=device,
        y=y
    )
    _, _, X_test, y_test = train_test_split(X, y, random_state=42)
    baseline_metrics = compute_baseline_metrics(X_test, y_test)

    # Generate plots
    logger.info("Generating scientific plots...")

    # Group metrics plots
    plot_generator.plot_group_metrics_boxplot(fold_results, metric='r2', title_suffix='_5fold')
    plot_generator.plot_group_metrics_boxplot(fold_results, metric='rmse', title_suffix='_5fold')
    # Spatial error plots
    plot_generator.plot_spatial_errors_scientific(fold_results, title_suffix='_5fold')

    # Position error comparison (with baseline if available)
    #plot_generator.plot_combined_spatial_errors(fold_results, baseline_metrics)
    plot_generator.plot_position_error_boxplot(fold_results, baseline_metrics)
    plot_generator.plot_multiples_comparison(fold_results, baseline_metrics)
    plot_generator.plot_position_error_comparison(fold_results,baseline_metrics)
    #logger.info(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
