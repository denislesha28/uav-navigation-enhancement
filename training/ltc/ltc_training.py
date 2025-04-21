import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import logging
import numpy as np
from collections import defaultdict
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error
from training.lstm.lstm_eval import PerformanceVisualizer, evaluate_test_set, compare_position_errors, \
    compare_orientation_errors, compare_velocity_errors, calculate_spatial_errors, calculate_group_metrics

# Import LTC components from ncps
from ncps.torch import LTC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NavigationEnhacementLTC(nn.Module):
    def __init__(self,
                 input_size=128,
                 hidden_size=64,
                 dropout=0.1):
        super().__init__()
        self.input_size = input_size

        # Increase hidden size for complex dynamics
        self.hidden_size = hidden_size * 2

        # Add a pre-processing layer
        self.pre_process = nn.Linear(input_size, input_size)
        self.pre_norm = nn.LayerNorm(input_size)
        self.pre_dropout = nn.Dropout(dropout / 2)  # Lower dropout before LTC

        # Adjust LTC parameters
        self.ltc = LTC(
            input_size=input_size,
            units=self.hidden_size,
            return_sequences=True,
            batch_first=True,
            mixed_memory=False,
            ode_unfolds=4,  # Increase unfolds for more complex dynamics
        )

        # Add multiple projection layers
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Split the prediction into component groups
        self.attitude_fc = nn.Linear(self.hidden_size, 3)  # φE, φN, φU
        self.velocity_fc = nn.Linear(self.hidden_size, 3)  # δυE, δυN, δυU
        self.position_fc = nn.Linear(self.hidden_size, 3)  # δL, δλ, δh
        self.gyro_bias_fc = nn.Linear(self.hidden_size, 3)  # εx, εy, εz
        self.accel_bias_fc = nn.Linear(self.hidden_size, 3)  # ∇x, ∇y, ∇z

    def forward(self, x):
        # Pre-processing
        x = self.pre_process(x)
        x = self.pre_norm(x)
        x = self.pre_dropout(x)

        # Process with LTC
        ltc_out, _ = self.ltc(x)

        # Get the last hidden state
        last_hidden = ltc_out[:, -1, :]

        # Apply normalization and dropout
        out = self.layer_norm(last_hidden)
        out = self.dropout(out)

        # Component-specific predictions
        attitude = self.attitude_fc(out)
        velocity = self.velocity_fc(out)
        position = self.position_fc(out)
        gyro_bias = self.gyro_bias_fc(out)
        accel_bias = self.accel_bias_fc(out)

        # Return predictions by component
        return {
            'attitude': attitude,
            'velocity': velocity,
            'position': position,
            'gyro_bias': gyro_bias,
            'accel_bias': accel_bias
        }


class NavigationDataset(Dataset):
    def __init__(self, data, target, device):
        self.data = torch.as_tensor(data, dtype=torch.float32).to(device)
        self.labels = torch.as_tensor(target, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ComponentMetricsTracker:
    def __init__(self):
        self.mae = nn.L1Loss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'component_losses': defaultdict(list),
            'component_metrics': defaultdict(lambda: defaultdict(list))
        }

        # Define component slices
        self.component_slices = {
            'attitude': slice(0, 3),
            'velocity': slice(3, 6),
            'position': slice(6, 9),
            'gyro_bias': slice(9, 12),
            'accel_bias': slice(12, 15)
        }

    def update(self, outputs_dict, targets):
        """
        Update metrics for each component separately

        Args:
            outputs_dict: Dictionary of component outputs from the model
            targets: Full target tensor with all components
        """
        metrics = {}
        component_metrics = {}

        # Convert all tensors to numpy for sklearn metrics
        targets_np = targets.detach().cpu().numpy()

        # Combine outputs for overall metrics
        combined_outputs = torch.cat([
            outputs_dict['attitude'],
            outputs_dict['velocity'],
            outputs_dict['position'],
            outputs_dict['gyro_bias'],
            outputs_dict['accel_bias']
        ], dim=1).detach().cpu()

        # Calculate overall metrics
        overall_mae = self.mae(combined_outputs, targets).mean().item()
        overall_mse = self.mse(combined_outputs, targets).mean().item()
        overall_rmse = np.sqrt(overall_mse)
        overall_r2 = r2_score(targets_np, combined_outputs.numpy())
        overall_exp_var = explained_variance_score(targets_np, combined_outputs.numpy())

        metrics = {
            'mae': overall_mae,
            'rmse': overall_rmse,
            'r2': overall_r2,
            'explained_variance': overall_exp_var
        }

        # Calculate per-component metrics
        for component, tensor in outputs_dict.items():
            component_slice = self.component_slices[component]
            component_target = targets[:, component_slice]
            component_output = tensor.detach().cpu()
            component_target_np = component_target.detach().cpu().numpy()
            component_output_np = component_output.numpy()

            comp_mae = self.mae(component_output, component_target).mean().item()
            comp_mse = self.mse(component_output, component_target).mean().item()
            comp_rmse = np.sqrt(comp_mse)
            comp_r2 = r2_score(component_target_np, component_output_np)
            comp_exp_var = explained_variance_score(component_target_np, component_output_np)

            component_metrics[component] = {
                'mae': comp_mae,
                'rmse': comp_rmse,
                'r2': comp_r2,
                'explained_variance': comp_exp_var
            }

        return metrics, component_metrics


def train_component_ltc(X, y, device):
    logger.info(f"Input shape: {X.shape}, Output shape: {y.shape}")
    logger.info(f"Input dtype: {X.dtype}, Output dtype: {y.dtype}")

    params = {
        'epochs': 150,
        'batch_size': 16,
        'learning_rate': 0.0005,
        'hidden_size': 128,
        'weight_decay': 1e-5,
        'early_stopping_patience': 10
    }

    model = NavigationEnhacementLTC(
        input_size=X.shape[-1],  # Use actual input size from data
        hidden_size=params['hidden_size']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

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

    metrics_tracker = ComponentMetricsTracker()
    visualizer = PerformanceVisualizer()

    best_val_loss = float('inf')
    early_stopping_counter = 0
    criterion = nn.MSELoss()

    logger.info("Starting training loop...")
    for epoch in range(params['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_component_losses = defaultdict(float)
        batch_count = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            optimizer.zero_grad(set_to_none=True)

            try:
                # Forward pass - get component-specific outputs
                output_dict = model(data)

                # Get component-specific targets
                attitude_target = target[:, :3]
                velocity_target = target[:, 3:6]
                position_target = target[:, 6:9]
                gyro_bias_target = target[:, 9:12]
                accel_bias_target = target[:, 12:15]

                # Calculate component losses
                loss_attitude = criterion(output_dict['attitude'], attitude_target)
                loss_velocity = criterion(output_dict['velocity'], velocity_target)
                loss_position = criterion(output_dict['position'], position_target)
                loss_gyro_bias = criterion(output_dict['gyro_bias'], gyro_bias_target)
                loss_accel_bias = criterion(output_dict['accel_bias'], accel_bias_target)

                # Total loss - unweighted sum of component losses
                loss = loss_attitude + loss_velocity + loss_position + loss_gyro_bias + loss_accel_bias

                # Track component losses
                epoch_component_losses['attitude'] += loss_attitude.item()
                epoch_component_losses['velocity'] += loss_velocity.item()
                epoch_component_losses['position'] += loss_position.item()
                epoch_component_losses['gyro_bias'] += loss_gyro_bias.item()
                epoch_component_losses['accel_bias'] += loss_accel_bias.item()

                # Backward pass and optimization
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            except RuntimeError as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                logger.error(f"Data shape: {data.shape}, Target shape: {target.shape}")
                raise e

        avg_train_loss = epoch_loss / batch_count
        metrics_tracker.history['train_loss'].append(avg_train_loss)

        # Record component losses
        for component, total_loss in epoch_component_losses.items():
            avg_comp_loss = total_loss / batch_count
            metrics_tracker.history['component_losses'][component].append(avg_comp_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_component_losses = defaultdict(float)
        val_batch_count = 0

        with torch.no_grad():
            for data, target in val_loader:
                # Forward pass
                output_dict = model(data)

                # Get component targets
                attitude_target = target[:, :3]
                velocity_target = target[:, 3:6]
                position_target = target[:, 6:9]
                gyro_bias_target = target[:, 9:12]
                accel_bias_target = target[:, 12:15]

                # Calculate component losses
                val_loss_attitude = criterion(output_dict['attitude'], attitude_target)
                val_loss_velocity = criterion(output_dict['velocity'], velocity_target)
                val_loss_position = criterion(output_dict['position'], position_target)
                val_loss_gyro_bias = criterion(output_dict['gyro_bias'], gyro_bias_target)
                val_loss_accel_bias = criterion(output_dict['accel_bias'], accel_bias_target)

                # Total validation loss
                val_batch_loss = val_loss_attitude + val_loss_velocity + val_loss_position + \
                                 val_loss_gyro_bias + val_loss_accel_bias

                val_loss += val_batch_loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        metrics_tracker.history['val_loss'].append(avg_val_loss)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f'Epoch {epoch}:')
        logger.info(f'Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
        logger.info(f'Learning Rate = {current_lr:.6f}')

        # Only log component losses
        for component, total_loss in epoch_component_losses.items():
            avg_comp_loss = total_loss / batch_count
            logger.info(f'{component.capitalize()} Loss: {avg_comp_loss:.4f}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            save_model(model)
            logger.info(f"New best model saved with validation loss: {avg_val_loss:.6f}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= params['early_stopping_patience']:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "models", 'best_ltc_model.pth')))

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics, component_metrics, group_metrics, spatial_metrics, (y_true, y_pred) = evaluate_component_model_test(
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

    # Only generate visualizations at the end of training
    visualizer.plot_training_history(
        {'train_loss': metrics_tracker.history['train_loss'],
         'val_loss': metrics_tracker.history['val_loss']},
        title_suffix='_final'
    )

    visualizer.plot_prediction_analysis(y_true, y_pred, title_suffix='_final')
    visualizer.plot_group_metrics(group_metrics, title_suffix='_final')
    visualizer.plot_spatial_errors(spatial_metrics, title_suffix='_final')

    # Compare with standard Kalman filter
    logger.info("Running model comparison with standard Kalman filter...")
    comparison_results = run_component_model_comparison(model, test_loader, device)

    return model, metrics_tracker.history, test_metrics


def evaluate_component_model_test(model, test_loader, device):
    """Evaluate component-specific model on test set"""
    model.eval()
    all_targets = []
    all_outputs_dict = defaultdict(list)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output_dict = model(data)

            # Store targets
            all_targets.append(target.cpu().numpy())

            # Store component outputs
            for component, output in output_dict.items():
                all_outputs_dict[component].append(output.cpu().numpy())

    # Concatenate all batches
    y_true = np.vstack(all_targets)

    component_outputs = {}
    for component, outputs in all_outputs_dict.items():
        component_outputs[component] = np.vstack(outputs)

    # Combine predictions for full output vector
    y_pred = np.hstack([
        component_outputs['attitude'],
        component_outputs['velocity'],
        component_outputs['position'],
        component_outputs['gyro_bias'],
        component_outputs['accel_bias']
    ])

    # Calculate overall metrics
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred)
    }

    # Calculate spatial metrics
    spatial_metrics = calculate_spatial_errors(y_true, y_pred)

    # Calculate group metrics
    group_metrics = calculate_group_metrics(y_true, y_pred)

    # Calculate component metrics
    components = {
        0: 'Roll (φE)', 1: 'Pitch (φN)', 2: 'Yaw (φU)',
        3: 'East (δυE)', 4: 'North (δυN)', 5: 'Up (δυU)',
        6: 'Latitude (δL)', 7: 'Longitude (δλ)', 8: 'Height (δh)',
        9: 'X-axis (εx)', 10: 'Y-axis (εy)', 11: 'Z-axis (εz)',
        12: 'X-axis (∇x)', 13: 'Y-axis (∇y)', 14: 'Z-axis (∇z)'
    }

    component_metrics = [
        {
            'component': components[i],
            'rmse': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            'r2': r2_score(y_true[:, i], y_pred[:, i]),
            'mae': mean_absolute_error(y_true[:, i], y_pred[:, i]),
            'explained_variance': explained_variance_score(y_true[:, i], y_pred[:, i])
        }
        for i in range(y_true.shape[1])
    ]

    return metrics, component_metrics, group_metrics, spatial_metrics, (y_true, y_pred)


def run_component_model_comparison(model, test_loader, device):
    """Run component model comparison with standard Kalman filter"""
    model.eval()
    all_data = []
    all_targets = []
    enhanced_outputs_dict = defaultdict(list)

    with torch.no_grad():
        for data, target in test_loader:
            data_cpu = data.cpu().numpy()
            target_cpu = target.cpu().numpy()

            data, target = data.to(device), target.to(device)
            output_dict = model(data)

            all_data.append(data_cpu)
            all_targets.append(target_cpu)

            for component, output in output_dict.items():
                enhanced_outputs_dict[component].append(output.cpu().numpy())

    # Stack all data
    X = np.vstack(all_data)
    y_true = np.vstack(all_targets)

    # Process component outputs
    component_outputs = {}
    for component, outputs in enhanced_outputs_dict.items():
        component_outputs[component] = np.vstack(outputs)

    # Combine for full enhanced predictions
    y_enhanced = np.hstack([
        component_outputs['attitude'],
        component_outputs['velocity'],
        component_outputs['position'],
        component_outputs['gyro_bias'],
        component_outputs['accel_bias']
    ])

    # Run standard Kalman filter on the same test data
    logger.info("Running standard Kalman filter for comparison...")
    from kalman_filtering.uav_navigation_kf import UAVNavigationKF

    # Initialize Kalman filter
    kf = UAVNavigationKF(dt=0.1)  # Assuming 10Hz sampling

    # Initialize with zeros for initial state
    initial_state = np.zeros(9)
    kf.initialize(initial_state)

    # Create array to store Kalman filter predictions
    kf_predictions = np.zeros_like(y_true)

    # Process each test sample
    for i in range(len(X)):
        # Prediction step
        kf.predict()

        # Extract measurement values from the test data
        # Take the last time step of the sequence
        measurement = X[i, -1, :9]  # Last time step, first 9 features are raw sensor readings

        # Update KF with measurement
        kf.update(measurement)

        # Get the KF state estimation (9D state)
        state_estimate = kf.get_state()

        # Expand the 9D state to 15D error state format
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
    logger.info(f"Z Error - Standard KF: {position_metrics['baseline_z_error_m']:.4f}m, "
                f"LNN-Enhanced: {position_metrics['enhanced_z_error_m']:.4f}m, "
                f"Improvement: {position_metrics['z_improvement_percent']:.2f}%")

    logger.info("\nOrientation Error Comparison (degrees):")
    logger.info(f"Roll Error - Standard KF: {orientation_metrics['baseline_roll_error_deg']:.4f}°, "
                f"LNN-Enhanced: {orientation_metrics['enhanced_roll_error_deg']:.4f}°, "
                f"Improvement: {orientation_metrics['roll_improvement_percent']:.2f}%")
    logger.info(f"Pitch Error - Standard KF: {orientation_metrics['baseline_pitch_error_deg']:.4f}°, "
                f"LNN-Enhanced: {orientation_metrics['enhanced_pitch_error_deg']:.4f}°, "
                f"Improvement: {orientation_metrics['pitch_improvement_percent']:.2f}%")
    logger.info(f"Yaw Error - Standard KF: {orientation_metrics['baseline_yaw_error_deg']:.4f}°, "
                f"LNN-Enhanced: {orientation_metrics['enhanced_yaw_error_deg']:.4f}°, "
                f"Improvement: {orientation_metrics['yaw_improvement_percent']:.2f}%")

    logger.info("\nVelocity Error Comparison (m/s):")
    logger.info(f"East Velocity Error - Standard KF: {velocity_metrics['baseline_east_vel_error_mps']:.4f}m/s, "
                f"LNN-Enhanced: {velocity_metrics['enhanced_east_vel_error_mps']:.4f}m/s, "
                f"Improvement: {velocity_metrics['east_vel_improvement_percent']:.2f}%")
    logger.info(f"North Velocity Error - Standard KF: {velocity_metrics['baseline_north_vel_error_mps']:.4f}m/s, "
                f"LNN-Enhanced: {velocity_metrics['enhanced_north_vel_error_mps']:.4f}m/s, "
                f"Improvement: {velocity_metrics['north_vel_improvement_percent']:.2f}%")
    logger.info(f"Up Velocity Error - Standard KF: {velocity_metrics['baseline_up_vel_error_mps']:.4f}m/s, "
                f"LNN-Enhanced: {velocity_metrics['enhanced_up_vel_error_mps']:.4f}m/s, "
                f"Improvement: {velocity_metrics['up_vel_improvement_percent']:.2f}%")

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

def save_model(model):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "..", "models")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'best_ltc_model.pth'))