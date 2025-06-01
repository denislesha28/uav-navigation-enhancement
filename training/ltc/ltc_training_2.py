import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import logging
import numpy as np
from collections import defaultdict
from sklearn.metrics import r2_score, explained_variance_score
from training.lstm.lstm_eval import PerformanceVisualizer, evaluate_test_set, compare_position_errors, \
    compare_orientation_errors, compare_velocity_errors

# Import LTC components from ncps
from ncps.torch import LTC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedNavigationLTC(nn.Module):
    def __init__(self,
                 input_size=128,
                 hidden_size=64,
                 output_size=15,
                 dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size * 2

        # LTC layer for continuous-time dynamics
        self.ltc = LTC(
            input_size=input_size,
            units=hidden_size,
            return_sequences=True,
            batch_first=True,
            mixed_memory=False,
            ode_unfolds=4
        )

        # No attention mechanism - LTCs model temporal dynamics differently
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Process with LTC - unpack the tuple
        ltc_out, _ = self.ltc(x)

        # Now you can index ltc_out as a tensor
        last_hidden = ltc_out[:, -1, :]

        # Apply normalization and dropout
        out = self.layer_norm(last_hidden)
        out = self.dropout(out)

        # Final prediction
        predictions = self.fc(out)

        return predictions


class NavigationDataset(Dataset):
    def __init__(self, data, target, device):
        self.data = torch.as_tensor(data, dtype=torch.float32).to(device)
        self.labels = torch.as_tensor(target, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MetricsTracker:
    def __init__(self):
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'mae': [],
            'rmse': [],
            'r2': [],
            'explained_variance': []
        }

    def update(self, outputs, targets):
        outputs_np = outputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        mae = self.mae(outputs, targets).item()
        mse = self.mse(outputs, targets).item()
        rmse = np.sqrt(mse)
        r2 = r2_score(targets_np, outputs_np)
        exp_var = explained_variance_score(targets_np, outputs_np)

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'explained_variance': exp_var
        }


def train_ltc(X, y, device):
    logger.info(f"Input shape: {X.shape}, Output shape: {y.shape}")
    logger.info(f"Input dtype: {X.dtype}, Output dtype: {y.dtype}")

    params = {
        'epochs': 150,
        'batch_size': 16,
        'learning_rate': 5 * 1e-4,
        'hidden_size': 128,
        'weight_decay': 1e-5,  # Add weight decay for regularization
        'early_stopping_patience': 30
    }

    model = EnhancedNavigationLTC(
        input_size=X.shape[-1],
        hidden_size=params['hidden_size'],
        output_size=y.shape[-1]
    ).to(device)

    # 2. Adjust optimizer parameters if needed
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    test_size = len(X) - train_size - val_size

    dataset = NavigationDataset(X, y, device)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params['batch_size'])
    test_loader = DataLoader(test_set, batch_size=params['batch_size'])

    metrics_tracker = MetricsTracker()
    visualizer = PerformanceVisualizer()

    best_val_loss = float('inf')
    early_stopping_counter = 0

    logger.info("Starting training loop...")
    for epoch in range(params['epochs']):
        model.train()
        epoch_metrics = defaultdict(float)
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            optimizer.zero_grad(set_to_none=True)

            try:
                output = model(data)
                criterion = nn.MSELoss()

                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=60, eta_min=1e-5
                )
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
                )
                loss = criterion(output, target)

                batch_metrics = metrics_tracker.update(output, target)
                for k, v in batch_metrics.items():
                    epoch_metrics[k] += v

                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            except RuntimeError as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                logger.error(f"Data shape: {data.shape}, Target shape: {target.shape}")
                raise e

        avg_train_loss = epoch_loss / batch_count
        metrics_tracker.history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_metrics = defaultdict(float)
        val_batch_count = 0

        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)

                batch_metrics = metrics_tracker.update(output, target)
                for k, v in batch_metrics.items():
                    val_metrics[k] += v

                val_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        metrics_tracker.history['val_loss'].append(avg_val_loss)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        for metric_name in ['mae', 'rmse', 'r2', 'explained_variance']:
            train_metric = epoch_metrics[metric_name] / batch_count
            val_metric = val_metrics[metric_name] / val_batch_count
            metrics_tracker.history[metric_name].append(train_metric)
            logger.info(f'{metric_name.upper()}: Train = {train_metric:.4f}, Val = {val_metric:.4f}')

        logger.info(f'Epoch {epoch}:')
        logger.info(f'Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
        logger.info(f'Learning Rate = {current_lr:.6f}')

        if epoch % 10 == 0:
            visualizer.plot_training_history(metrics_tracker.history, title_suffix=f'_epoch_{epoch}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            __save_model(model)
            logger.info(f"New best model saved with validation loss: {avg_val_loss:.6f}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= params['early_stopping_patience']:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Final training loss: {metrics_tracker.history['train_loss'][-1]:.6f}")

    logger.info("Evaluating on test set...")
    test_metrics, component_metrics, group_metrics, spatial_metrics, (y_true, y_pred) = evaluate_test_set(
        model, test_loader, device)

    logger.info("\nTest Set Metrics:")
    for metric_name, value in test_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")

    logger.info("\nSpatial Metrics (Chen et al.):")
    logger.info(f"2D Error - Mean: {spatial_metrics['2d_error']['mean']:.4f}, "
                f"Std: {spatial_metrics['2d_error']['std']:.4f}")
    logger.info(f"Z Error - Mean: {spatial_metrics['z_error']['mean']:.4f}, "
                f"Std: {spatial_metrics['z_error']['std']:.4f}")

    logger.info("\nComponent-wise Metrics:")
    for comp in component_metrics:
        logger.info(f"Component {comp['component']}: "
                    f"RMSE={comp['rmse']:.4f}, "
                    f"R²={comp['r2']:.4f}, "
                    f"MAE={comp['mae']:.4f}")

    logger.info("\nGroup Metrics:")
    for group_name, metrics in group_metrics.items():
        logger.info(f"{group_name.capitalize()}: "
                    f"RMSE={metrics['rmse']:.4f}, "
                    f"R²={metrics['r2']:.4f}, "
                    f"MAE={metrics['mae']:.4f}")

    visualizer.plot_training_history(metrics_tracker.history, title_suffix='_final')
    visualizer.plot_prediction_analysis(y_true, y_pred, title_suffix='_final')
    visualizer.plot_group_metrics(group_metrics, title_suffix='_final')
    visualizer.plot_spatial_errors(spatial_metrics, title_suffix='_final')

    logger.info("Running unit-based model comparison...")
    run_model_comparison(model, test_loader, device)

    return model, metrics_tracker.history, test_metrics


def __save_model(model):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "..", "models")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'best_ltc_model.pth'))


def run_model_comparison(enhanced_model, test_loader, device):
    """
    Run unit-based comparison between LNN-enhanced model and standard Kalman filter
    """
    logger.info("Running unit-based model comparison...")

    # Get test data for evaluation
    all_data = []
    all_targets = []
    enhanced_predictions = []

    # First, get enhanced model predictions
    enhanced_model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data_cpu = data.cpu().numpy()
            target_cpu = target.cpu().numpy()

            data, target = data.to(device), target.to(device)
            enhanced_output = enhanced_model(data)

            all_data.append(data_cpu)
            all_targets.append(target_cpu)
            enhanced_predictions.append(enhanced_output.cpu().numpy())

    # Stack all data
    X = np.vstack(all_data)
    y_true = np.vstack(all_targets)
    y_enhanced = np.vstack(enhanced_predictions)

    # Run standard Kalman filter on the same test data
    logger.info("Running standard Kalman filter for comparison...")
    from kalman_filtering.uav_navigation_kf import UAVNavigationKF

    # Initialize Kalman filter
    kf = UAVNavigationKF(dt=0.1)  # Assuming 10Hz sampling as in your pipeline

    # Initialize with first ground truth state (from the first test sample)
    # For simplicity, we'll use zeros for initial state
    initial_state = np.zeros(9)
    kf.initialize(initial_state)

    # Create array to store Kalman filter predictions
    kf_predictions = np.zeros_like(y_true)

    # Process each test sample
    for i in range(len(X)):
        # Prediction step
        kf.predict()

        # Extract measurement values from the test data
        # Assuming X contains the sensor readings in the order expected by the KF
        # Take the last time step of the sequence
        measurement = X[i, -1, :9]  # Last time step, first 9 features are raw sensor readings

        # Update KF with measurement
        kf.update(measurement)

        # Get the KF state estimation (9D state)
        state_estimate = kf.get_state()

        # IMPORTANT: Expand the 9D state to 15D error state format
        # This requires mapping the absolute state to the error state format
        # The mapping depends on your specific implementation, but typically:
        expanded_state = np.zeros(15)

        # First 3 elements: attitude errors (roll, pitch, yaw) - set to small values for KF estimate
        expanded_state[0:3] = 0.01  # Small attitude errors

        # Next 3 elements: velocity errors (east, north, up)
        # Use absolute velocity values from KF state
        expanded_state[3:6] = state_estimate[3:6]

        # Next 3 elements: position errors (lat, long, height)
        # Use absolute position values from KF state
        expanded_state[6:9] = state_estimate[0:3]

        # Last 6 elements: gyroscope and accelerometer bias errors
        # Set to small values for KF estimate
        expanded_state[9:15] = 0.01  # Small sensor bias errors

        # Store the expanded KF prediction
        kf_predictions[i] = expanded_state

    # Now we have both predictions in compatible formats
    y_baseline = kf_predictions

    # Calculate unit-based comparisons
    position_metrics = compare_position_errors(y_baseline, y_enhanced, y_true)
    orientation_metrics = compare_orientation_errors(y_baseline, y_enhanced, y_true)
    velocity_metrics = compare_velocity_errors(y_baseline, y_enhanced, y_true)

    # Log the results
    logger.info("\nUnit-Based Comparison Results (Chen et al. metrics):")
    logger.info("=================================================")

    logger.info("\nPosition Error Comparison (meters):")
    logger.info(f"2D Error - Standard KF: {position_metrics['baseline_2d_error_m']:.4f}m, "
                f"LNN-Enhanced: {position_metrics['enhanced_2d_error_m']:.4f}m, "
                f"Improvement: {position_metrics['2d_improvement_percent']:.2f}%")

    # Rest of the logging remains the same...

    # Create visualizations
    visualizer = PerformanceVisualizer()
    visualizer.plot_position_error_comparison(position_metrics, title_suffix='_final')
    visualizer.plot_orientation_error_comparison(orientation_metrics, title_suffix='_final')
    visualizer.plot_velocity_error_comparison(velocity_metrics, title_suffix='_final')

    return {
        "position": position_metrics,
        "orientation": orientation_metrics,
        "velocity": velocity_metrics
    }
