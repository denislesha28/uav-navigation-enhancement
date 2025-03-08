from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


def calculate_spatial_errors(y_true, y_pred):
    """Calculate 2D and Z-Direction errors as in Chen et al."""
    # 2D error (position components 6 and 7 - Latitude/Longitude)
    e_2d = np.sqrt((y_true[:, 6] - y_pred[:, 6]) ** 2 +
                   (y_true[:, 7] - y_pred[:, 7]) ** 2)

    # Z-Direction error (position component 8 - Height)
    e_z = np.abs(y_true[:, 8] - y_pred[:, 8])

    metrics = {
        '2d_error': {
            'mean': np.mean(e_2d),
            'variance': np.var(e_2d),
            'std': np.std(e_2d)
        },
        'z_error': {
            'mean': np.mean(e_z),
            'variance': np.var(e_z),
            'std': np.std(e_z)
        }
    }

    return metrics


def calculate_group_metrics(y_true, y_pred):
    """Calculate metrics for each logical group in the error state vector"""
    groups = {
        'attitude': slice(0, 3),
        'velocity': slice(3, 6),
        'position': slice(6, 9),
        'gyro_bias': slice(9, 12),
        'accel_bias': slice(12, 15)
    }

    group_metrics = {}
    for name, indices in groups.items():
        group_true = y_true[:, indices]
        group_pred = y_pred[:, indices]

        group_metrics[name] = {
            'rmse': np.sqrt(mean_squared_error(group_true, group_pred)),
            'mae': mean_absolute_error(group_true, group_pred),
            'r2': r2_score(group_true, group_pred)
        }

    return group_metrics


class PerformanceVisualizer:
    def __init__(self, save_dir=None):
        if save_dir is None:
            current_file_dir = Path(__file__).resolve().parent
            self.save_dir = current_file_dir.parent / "training_logs"

        else:
            self.save_dir = Path(save_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_prediction_comparison(self, y_true, y_pred, title_suffix=''):
        """Plot true vs predicted values"""
        plt.figure(figsize=(12, 6))

        # Sample plot for first component
        plt.subplot(1, 2, 1)
        plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
        plt.plot([y_true[:, 0].min(), y_true[:, 0].max()],
                 [y_true[:, 0].min(), y_true[:, 0].max()],
                 'r--', lw=2)
        plt.title(f'True vs Predicted (First Component) {title_suffix}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')

        # Error distribution
        plt.subplot(1, 2, 2)
        error = y_pred - y_true
        sns.histplot(error[:, 0], kde=True)
        plt.title(f'Prediction Error Distribution {title_suffix}')
        plt.xlabel('Error')

        plt.tight_layout()
        plt.savefig(self.save_dir / f'prediction_comparison{title_suffix}.png')
        plt.close()

    def plot_training_history(self, history, title_suffix=''):
        """Plot training metrics history"""
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))

        ax[0].plot(history['train_loss'], label='Train Loss', color='blue')
        ax[0].plot(history['val_loss'], label='Validation Loss', color='red')
        ax[0].set_title(f'Loss History {title_suffix}')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].grid(True)

        if 'learning_rate' in history:
            ax_lr = ax[0].twinx()
            ax_lr.plot(history['learning_rate'], label='Learning Rate',
                       color='green', linestyle='--')
            ax_lr.set_ylabel('Learning Rate')
            ax_lr.legend(loc='upper right')

        metrics_to_plot = [k for k in history.keys()
                           if k not in ['train_loss', 'val_loss', 'learning_rate']]
        for metric in metrics_to_plot:
            ax[1].plot(history[metric], label=metric)
        ax[1].set_title('Training Metrics')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Value')
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        plt.savefig(self.save_dir / f'training_history{title_suffix}.png')
        plt.close()

    def plot_prediction_analysis(self, y_true, y_pred, title_suffix=''):
        """Plot prediction analysis for different error state groups"""
        error = y_pred - y_true

        component_groups = {
            'Attitude Errors': {
                0: 'Roll (φE)',
                1: 'Pitch (φN)',
                2: 'Yaw (φU)'
            },
            'Velocity Errors': {
                3: 'East (δυE)',
                4: 'North (δυN)',
                5: 'Up (δυU)'
            },
            'Position Errors': {
                6: 'Latitude (δL)',
                7: 'Longitude (δλ)',
                8: 'Height (δh)'
            },
            'Gyroscope Bias': {
                9: 'X-axis (εx)',
                10: 'Y-axis (εy)',
                11: 'Z-axis (εz)'
            },
            'Accelerometer Bias': {
                12: 'X-axis (∇x)',
                13: 'Y-axis (∇y)',
                14: 'Z-axis (∇z)'
            }
        }

        # Create subplot for each group
        for i, (group_name, components) in enumerate(component_groups.items()):
            plt.figure(figsize=(12, 6))

            # Plot error distribution for each component in group
            for comp_idx, comp_name in components.items():
                sns.kdeplot(error[:, comp_idx], label=comp_name)

            plt.title(f'{group_name} Distribution')
            plt.xlabel('Error')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)

            # Add RMSE and R² values as text
            text_str = "Metrics:\n"
            for comp_idx, comp_name in components.items():
                rmse = np.sqrt(mean_squared_error(y_true[:, comp_idx], y_pred[:, comp_idx]))
                r2 = r2_score(y_true[:, comp_idx], y_pred[:, comp_idx])
                text_str += f"{comp_name}:\nRMSE={rmse:.4f}, R²={r2:.4f}\n\n"

            plt.figtext(1.02, 0.5, text_str, fontsize=9, va='center')
            plt.tight_layout()
            plt.savefig(self.save_dir / f'error_distribution_{group_name}{title_suffix}.png',
                        bbox_inches='tight')
            plt.close()

        # Time series analysis plots
        time_steps = np.arange(len(y_true))

        for group_name, components in component_groups.items():
            plt.figure(figsize=(15, 8))

            for comp_idx, comp_name in components.items():
                plt.subplot(len(components), 1, list(components.keys()).index(comp_idx) + 1)

                plt.plot(time_steps, y_true[:, comp_idx], label='True', color='blue', alpha=0.6)
                plt.plot(time_steps, y_pred[:, comp_idx], label='Predicted',
                         color='red', alpha=0.6, linestyle='--')

                plt.fill_between(time_steps,
                                 y_true[:, comp_idx] - np.abs(error[:, comp_idx]),
                                 y_true[:, comp_idx] + np.abs(error[:, comp_idx]),
                                 color='gray', alpha=0.2, label='Error Band')

                plt.title(f'{comp_name} Over Time')
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)

            plt.tight_layout()
            plt.savefig(self.save_dir / f'timeseries_{group_name}{title_suffix}.png')
            plt.close()

    def plot_group_metrics(self, group_metrics, title_suffix=''):
        """Plot metrics for each logical group in the error state vector"""
        plt.figure(figsize=(15, 8))

        metrics_to_plot = ['rmse', 'mae', 'r2']
        groups = list(group_metrics.keys())

        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(1, len(metrics_to_plot), i + 1)
            values = [group_metrics[group][metric] for group in groups]

            plt.bar(groups, values)
            plt.title(f'{metric.upper()} by Component Group')
            plt.ylabel(metric.upper())
            plt.xticks(rotation=45)
            plt.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig(self.save_dir / f'group_metrics{title_suffix}.png')
        plt.close()

    def plot_spatial_errors(self, spatial_metrics, title_suffix=''):
        """Plot spatial errors as in Chen et al."""
        plt.figure(figsize=(12, 6))

        # 2D Error
        plt.subplot(1, 2, 1)
        values = [spatial_metrics['2d_error']['mean'], spatial_metrics['2d_error']['std']]
        plt.bar(['Mean', 'Std Dev'], values)
        plt.title('2D Position Error (Latitude/Longitude)')
        plt.ylabel('Error (m)')
        plt.grid(True, axis='y')

        # Z Error
        plt.subplot(1, 2, 2)
        values = [spatial_metrics['z_error']['mean'], spatial_metrics['z_error']['std']]
        plt.bar(['Mean', 'Std Dev'], values)
        plt.title('Z-Direction Error (Height)')
        plt.ylabel('Error (m)')
        plt.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig(self.save_dir / f'spatial_errors{title_suffix}.png')
        plt.close()


def evaluate_test_set(model, test_loader, device):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # Output is already final predictions

            all_targets.append(target.cpu().numpy())
            all_predictions.append(output.cpu().numpy())

    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_predictions)

    # Basic metrics
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred)
    }

    # Spatial metrics from Chen et al.
    spatial_metrics = calculate_spatial_errors(y_true, y_pred)
    metrics.update({
        '2d_error_mean': spatial_metrics['2d_error']['mean'],
        '2d_error_std': spatial_metrics['2d_error']['std'],
        'z_error_mean': spatial_metrics['z_error']['mean'],
        'z_error_std': spatial_metrics['z_error']['std']
    })

    # Group metrics
    group_metrics = calculate_group_metrics(y_true, y_pred)

    # Component metrics
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