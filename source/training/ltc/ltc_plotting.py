import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset, Subset

from training.ltc.ltc_training_k_fold import EnhancedNavigationLTC

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3
})


class LTCKFoldPlotter:
    """Class for generating scientific publication-quality plots from model evaluation results"""

    def __init__(self, save_dir=None):
        """Initialize the plot generator

        Args:
            save_dir: Directory to save plots
        """
        if save_dir is None:
            self.save_dir = Path("scientific_plots")
        else:
            self.save_dir = Path(save_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Define component and group names
        self.component_names = {
            0: 'Roll (φE)', 1: 'Pitch (φN)', 2: 'Yaw (φU)',
            3: 'East (δυE)', 4: 'North (δυN)', 5: 'Up (δυU)',
            6: 'Latitude (δL)', 7: 'Longitude (δλ)', 8: 'Height (δh)',
            9: 'X-axis (εx)', 10: 'Y-axis (εy)', 11: 'Z-axis (εz)',
            12: 'X-axis (∇x)', 13: 'Y-axis (∇y)', 14: 'Z-axis (∇z)'
        }

        self.group_names = {
            'attitude': 'Attitude',
            'velocity': 'Velocity',
            'position': 'Position',
            'gyro_bias': 'Gyro Bias',
            'accel_bias': 'Accel Bias'
        }

        # Define color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'tertiary': '#2ca02c',
            'error': '#d62728',
            'accent': '#9467bd',
            'grid': '#cccccc'
        }

        # Standard set of markers for plots
        self.markers = ['o', 's', 'D', '^', 'v']

    def evaluate_models(self, index_paths, model_paths, X_tensor, y, device):
        """
        Evaluate models directly from saved .pth files using saved fold indices

        Args:
            model_paths: List of paths to model files
            X: Full input data array
            y: Full target data array
            device: Torch device
        """
        # Create full dataset
        #X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.DoubleTensor(y).to(device)

        full_dataset = TensorDataset(X_tensor[0], y_tensor)

        results = []

        for  index_path in  index_paths:
            # Get fold number
            fold_num = int(Path(index_path).stem.split('_')[1])

            # Load test indices
            indices_path = Path(index_path).parent / f"fold_{fold_num}_test_indices.npy"
            print(indices_path)
            test_indices = np.load(indices_path)

            # Create test dataset
            test_dataset = Subset(full_dataset, test_indices)
            test_loader = DataLoader(test_dataset, batch_size=16)

            # Load model
            model = EnhancedNavigationLTC(
                input_size=128,
                hidden_size=128
            ).to(device)

            # Dictionary to store results
            all_results = {
                'metrics': [],
                'component_metrics': [],
                'group_metrics': [],
                'spatial_metrics': [],
                'predictions': []
            }

            # Evaluate each model
            for fold_idx, model_path in enumerate(model_paths):
                model.load_state_dict(torch.load(model_path))

                try:
                    # Evaluate model
                    logger.info(f"Evaluating model for fold {fold_idx + 1}")
                    metrics, component_metrics, group_metrics, spatial_metrics, predictions = self._evaluate_model(
                        model, test_loader, device
                    )

                    # Store results
                    all_results['metrics'].append(metrics)
                    all_results['component_metrics'].append(component_metrics)
                    all_results['group_metrics'].append(group_metrics)
                    all_results['spatial_metrics'].append(spatial_metrics)
                    all_results['predictions'].append(predictions)

                    logger.info(f"Fold {fold_idx + 1} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

                except Exception as e:
                    logger.error(f"Error evaluating model {model_path}: {e}")
                    raise e

            return all_results

    def evaluate_models_from_files(self, model_paths, test_data, device, model_class, batch_size=16):
        """
        Evaluate models directly from saved .pth files

        Args:
            model_paths: List of paths to .pth model files
            test_data: Tuple of (X_test, y_test) numpy arrays
            device: Torch device (cuda/cpu)
            model_class: Model class to instantiate
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with evaluation results for all models
        """
        logger.info(f"Evaluating {len(model_paths)} models from .pth files")

        # Extract test data
        X_test, y_test = test_data

        # Convert to torch tensors
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)

        # Create test dataset and loader
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Dictionary to store results
        all_results = {
            'metrics': [],
            'component_metrics': [],
            'group_metrics': [],
            'spatial_metrics': [],
            'predictions': []
        }

        # Evaluate each model
        for fold_idx, model_path in enumerate(model_paths):
            try:
                # Load model
                logger.info(f"Loading model from {model_path}")
                model = self._load_model(model_path, model_class, X_test.shape[-1], y_test.shape[-1], device)

                # Evaluate model
                logger.info(f"Evaluating model for fold {fold_idx + 1}")
                metrics, component_metrics, group_metrics, spatial_metrics, predictions = self._evaluate_model(
                    model, test_loader, device
                )

                # Store results
                all_results['metrics'].append(metrics)
                all_results['component_metrics'].append(component_metrics)
                all_results['group_metrics'].append(group_metrics)
                all_results['spatial_metrics'].append(spatial_metrics)
                all_results['predictions'].append(predictions)

                logger.info(f"Fold {fold_idx + 1} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating model {model_path}: {e}")
                raise e

        return all_results

    def _load_model(self, model_path, model_class, input_size, output_size, device):
        """
        Load a PyTorch model from a .pth file

        Args:
            model_path: Path to .pth model file
            model_class: Model class to instantiate
            input_size: Input size for model
            output_size: Output size for model
            device: Torch device

        Returns:
            Loaded model
        """
        # Create model instance
        model = model_class(
            input_size=input_size,
            output_size=output_size
        ).to(device)

        # Load state dict
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Set to evaluation mode
        model.eval()

        return model

    def _evaluate_model(self, model, test_loader, device):
        """
        Evaluate a model on test data

        Args:
            model: PyTorch model
            test_loader: DataLoader with test data
            device: Torch device

        Returns:
            Evaluation metrics
        """
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error,
            r2_score, explained_variance_score
        )

        # Set model to evaluation mode
        model.eval()

        # Lists to store all targets and predictions
        all_targets = []
        all_predictions = []

        # Disable gradient computation
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                # Forward pass
                model.double()
                output = model(data)

                # Store targets and predictions
                all_targets.append(target.cpu().numpy())
                all_predictions.append(output.cpu().numpy())

        # Concatenate all batches
        y_true = np.vstack(all_targets)
        y_pred = np.vstack(all_predictions)

        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred)
        }

        # Calculate spatial metrics
        spatial_metrics = self._calculate_spatial_errors(y_true, y_pred)

        # Calculate group metrics
        group_metrics = self._calculate_group_metrics(y_true, y_pred)

        # Calculate component metrics
        component_metrics = []
        for i in range(y_true.shape[1]):
            component_metrics.append({
                'component': self.component_names[i],
                'rmse': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
                'r2': r2_score(y_true[:, i], y_pred[:, i]),
                'mae': mean_absolute_error(y_true[:, i], y_pred[:, i]),
                'explained_variance': explained_variance_score(y_true[:, i], y_pred[:, i])
            })

        return metrics, component_metrics, group_metrics, spatial_metrics, (y_true, y_pred)

    def _calculate_spatial_errors(self, y_true, y_pred):
        """
        Calculate 2D and Z-Direction errors as in Chen et al.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Dictionary with spatial metrics
        """
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

    def _calculate_group_metrics(self, y_true, y_pred):
        """
        Calculate metrics for each logical group in the error state vector

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Dictionary with group metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

    def plot_group_metrics_boxplot(self, fold_results, metric='r2', title_suffix=''):
        """
        Create scientific boxplots of group metrics across folds

        Args:
            fold_results: Dictionary with results from all folds
            metric: Metric to plot ('r2', 'rmse', or 'mae')
            title_suffix: Suffix to add to plot title
        """
        # Extract group metrics from folds
        group_metrics = []

        for fold_idx, fold_group_metrics in enumerate(fold_results['group_metrics']):
            for group_name, metrics in fold_group_metrics.items():
                group_metrics.append({
                    'fold': fold_idx + 1,
                    'group': self.group_names.get(group_name, group_name),
                    'metric': metrics[metric],
                    'metric_name': metric
                })

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(group_metrics)

        # Set figure size based on number of groups
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create boxplot
        sns.boxplot(
            x='group',
            y='metric',
            data=df,
            ax=ax,
            palette='viridis',
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
        )

        # Add individual data points
        sns.stripplot(
            x='group',
            y='metric',
            data=df,
            ax=ax,
            color='black',
            size=4,
            jitter=True
        )

        # Customize plot
        metric_names = {'r2': 'R²', 'rmse': 'RMSE', 'mae': 'MAE'}
        ax.set_title(f'{metric_names.get(metric, metric.upper())} by Component Group{title_suffix}',
                     fontsize=16, pad=20)
        ax.set_xlabel('Component Group', fontsize=14, labelpad=10)
        ax.set_ylabel(metric_names.get(metric, metric.upper()), fontsize=14, labelpad=10)

        # Rotate x-tick labels
        plt.xticks(rotation=0)

        # Add mean values as text annotations
        for i, group in enumerate(df['group'].unique()):
            group_data = df[df['group'] == group]['metric']
            mean_val = group_data.mean()
            std_val = group_data.std()

            ax.annotate(f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}',
                        xy=(i, group_data.min() - 0.1 * (group_data.max() - group_data.min())),
                        ha='center',
                        va='top',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))

        # Add statistical significance indicators if more than one group
        if len(df['group'].unique()) > 1:
            # Perform statistical tests between groups
            groups = df['group'].unique()

            # We'll add significance bars only for selected comparisons
            # Here we compare each group with the next one
            for i in range(len(groups) - 1):
                group1 = groups[i]
                group2 = groups[i + 1]

                # Get data for the two groups
                data1 = df[df['group'] == group1]['metric']
                data2 = df[df['group'] == group2]['metric']

                # Perform t-test
                t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)

                # Determine significance
                if p_val < 0.001:
                    sig_symbol = '***'
                elif p_val < 0.01:
                    sig_symbol = '**'
                elif p_val < 0.05:
                    sig_symbol = '*'
                else:
                    sig_symbol = 'n.s.'

                # Position for the significance bar
                y_pos = max(data1.max(), data2.max()) * 1.1

                # Draw a line between the two groups
                ax.plot([i, i + 1], [y_pos, y_pos], 'k-', linewidth=1.5)

                # Add the significance symbol
                ax.text((i + i + 1) / 2, y_pos, sig_symbol, ha='center', va='bottom')

        # Add grid lines for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        # Improve figure style
        fig.tight_layout()

        # Save figure
        metric_str = metric.replace(' ', '_').lower()
        plt.savefig(self.save_dir / f'group_{metric_str}_scientific_boxplot{title_suffix}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_combined_spatial_errors(self, df_2d, df_z, title_suffix=''):
        """
        Create combined boxplot of 2D and Z errors with statistics above each boxplot
        """
        # Prepare data for combined plot
        df_2d['error_type'] = '2D Error'
        df_z['error_type'] = 'Z Error'
        combined_df = pd.concat([df_2d, df_z])

        # Create plot
        plt.figure(figsize=(12, 8))

        ax = sns.boxplot(x='error_type', y='error', data=combined_df, palette=['#1f77b4', '#2ca02c'])

        # Add individual points
        sns.stripplot(x='error_type', y='error', data=combined_df,
                      color='black', size=3, alpha=0.4, jitter=True)

        # Calculate statistics for each error type
        for i, error_type in enumerate(['2D Error', 'Z Error']):
            error_data = combined_df[combined_df['error_type'] == error_type]['error']

            stats_text = (
                f"Mean: {error_data.mean():.4f} m\n"
                f"Median: {error_data.median():.4f} m\n"
                f"Std Dev: {error_data.std():.4f} m"
            )

            # Position the stats box above each boxplot
            plt.annotate(stats_text,
                         xy=(i, error_data.max() * 1.1),
                         ha='center', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))

        # Perform t-test between 2D and Z errors
        t_stat, p_value = stats.ttest_ind(
            combined_df[combined_df['error_type'] == '2D Error']['error'],
            combined_df[combined_df['error_type'] == 'Z Error']['error'],
            equal_var=False
        )

        # Add significance indicator to plot title
        if p_value < 0.001:
            sig_str = "p < 0.0001\n***"
        elif p_value < 0.01:
            sig_str = f"p = {p_value:.4f}\n**"
        elif p_value < 0.05:
            sig_str = f"p = {p_value:.4f}\n*"
        else:
            sig_str = f"p = {p_value:.4f}"

        # Add statistical significance
        y_max = combined_df['error'].max() * 1.2
        ax.plot([0, 1], [y_max, y_max], 'k-', linewidth=1.5)
        ax.text(0.5, y_max, sig_str, ha='center', va='bottom')

        # Customize plot
        plt.title('Position Error Comparison', fontsize=16)
        plt.xlabel('Error Type', fontsize=14)
        plt.ylabel('Error (meters)', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Save figure
        plt.savefig(self.save_dir / f'combined_spatial_errors{title_suffix}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

    def plot_position_error_boxplot(self, fold_results, baseline_metrics=None, title_suffix=''):
        """
        Create scientific boxplot of position errors with baseline comparison

        Args:
            fold_results: Dictionary with results from all folds
            baseline_metrics: Dictionary with baseline metrics (optional)
            title_suffix: Suffix to add to plot title
        """
        # Extract position errors from folds
        position_data = {
            '2D Error': [],
            'Z Error': []
        }

        # For each fold's spatial metrics
        for fold_idx, spatial_metrics in enumerate(fold_results['spatial_metrics']):
            position_data['2D Error'].append({
                'fold': fold_idx + 1,
                'model': 'LTC-Enhanced',
                'error': spatial_metrics['2d_error']['mean']
            })

            position_data['Z Error'].append({
                'fold': fold_idx + 1,
                'model': 'LTC-Enhanced',
                'error': spatial_metrics['z_error']['mean']
            })

        # Add baseline metrics if provided
        if baseline_metrics:
            position_data['2D Error'].append({
                'fold': 0,
                'model': 'Standard KF',
                'error': baseline_metrics.get('2d_error_mean', 0)
            })

            position_data['Z Error'].append({
                'fold': 0,
                'model': 'Standard KF',
                'error': baseline_metrics.get('z_error_mean', 0)
            })

        # Convert to DataFrames
        df_2d = pd.DataFrame(position_data['2D Error'])
        df_z = pd.DataFrame(position_data['Z Error'])

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 2D error
        if baseline_metrics:
            # Plot with baseline comparison
            sns.barplot(x='model', y='error', data=df_2d, ax=ax1,
                        palette={'Standard KF': '#d62728', 'LTC-Enhanced': '#1f77b4'})

            # Add individual points for LTC
            sns.stripplot(x='model', y='error',
                          data=df_2d[df_2d['model'] == 'LTC-Enhanced'],
                          ax=ax1, color='black', size=7, jitter=True)

            # Calculate improvement percentage
            baseline_error = baseline_metrics.get('2d_error_mean', 0)
            ltc_error = df_2d[df_2d['model'] == 'LTC-Enhanced']['error'].mean()

            if baseline_error > 0:
                improvement = ((baseline_error - ltc_error) / baseline_error) * 100

                # Add improvement text
                max_val = df_2d['error'].max() * 1.1
                ax1.annotate(f"{improvement:.1f}% improvement",
                             xy=(1, max_val),
                             xytext=(0, 5),
                             textcoords="offset points",
                             ha='center', va='bottom',
                             color='green', fontsize=14,
                             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        else:
            # Plot without baseline
            sns.boxplot(y='error', data=df_2d, ax=ax1, color='#1f77b4')
            sns.stripplot(y='error', data=df_2d, ax=ax1, color='black', size=7, jitter=True)

        # Customize 2D error plot
        ax1.set_title('2D Position Error', fontsize=16)
        ax1.set_ylabel('Error (meters)', fontsize=14)
        if baseline_metrics:
            ax1.set_xlabel('Model', fontsize=14)
        else:
            ax1.set_xlabel('')
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Add statistics to 2D plot
        stats_2d = {
            'mean': df_2d[df_2d['model'] == 'LTC-Enhanced']['error'].mean(),
            'std': df_2d[df_2d['model'] == 'LTC-Enhanced']['error'].std()
        }

        stats_text_2d = f"Mean: {stats_2d['mean']:.4f} m\nStd: {stats_2d['std']:.4f} m"
        ax1.text(0.05, 0.95, stats_text_2d, transform=ax1.transAxes,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                 fontsize=12)

        # Plot Z error
        if baseline_metrics:
            # Plot with baseline comparison
            sns.barplot(x='model', y='error', data=df_z, ax=ax2,
                        palette={'Standard KF': '#d62728', 'LTC-Enhanced': '#2ca02c'})

            # Add individual points for LTC
            sns.stripplot(x='model', y='error',
                          data=df_z[df_z['model'] == 'LTC-Enhanced'],
                          ax=ax2, color='black', size=7, jitter=True)

            # Calculate improvement percentage
            baseline_error = baseline_metrics.get('z_error_mean', 0)
            ltc_error = df_z[df_z['model'] == 'LTC-Enhanced']['error'].mean()

            if baseline_error > 0:
                improvement = ((baseline_error - ltc_error) / baseline_error) * 100

                # Add improvement text
                max_val = df_z['error'].max() * 1.1
                ax2.annotate(f"{improvement:.1f}% improvement",
                             xy=(1, max_val),
                             xytext=(0, 5),
                             textcoords="offset points",
                             ha='center', va='bottom',
                             color='green', fontsize=14,
                             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        else:
            # Plot without baseline
            sns.boxplot(y='error', data=df_z, ax=ax2, color='#2ca02c')
            sns.stripplot(y='error', data=df_z, ax=ax2, color='black', size=7, jitter=True)

        # Customize Z error plot
        ax2.set_title('Z-Direction Error', fontsize=16)
        ax2.set_ylabel('Error (meters)', fontsize=14)
        if baseline_metrics:
            ax2.set_xlabel('Model', fontsize=14)
        else:
            ax2.set_xlabel('')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Add statistics to Z plot
        stats_z = {
            'mean': df_z[df_z['model'] == 'LTC-Enhanced']['error'].mean(),
            'std': df_z[df_z['model'] == 'LTC-Enhanced']['error'].std()
        }

        stats_text_z = f"Mean: {stats_z['mean']:.4f} m\nStd: {stats_z['std']:.4f} m"
        ax2.text(0.05, 0.95, stats_text_z, transform=ax2.transAxes,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                 fontsize=12)

        # Add overall title
        if baseline_metrics:
            fig.suptitle('Position Error Comparison: Standard KF vs. LTC-Enhanced', fontsize=18)
        else:
            fig.suptitle('Position Errors Across All Folds', fontsize=18)

        ax1.set_yscale('log')
        ax2.set_yscale('log')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

        # Save figure
        plt.savefig(self.save_dir / f'position_error_comparison{title_suffix}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

    def plot_multiples_comparison(self, fold_results, baseline_metrics, title_suffix=''):
        """
        Create small multiples plot showing fold performance with different scales

        Args:
            fold_results: Dictionary with results from all folds
            baseline_metrics: Dictionary with baseline metrics
            title_suffix: Suffix to add to plot title
        """
        # Create figure with grid for multiple plots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3)

        # Create a broken axis for the linear plot
        gs_linear = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], height_ratios=[1, 1], hspace=0.05)
        ax1_top = fig.add_subplot(gs_linear[0])
        ax1_bottom = fig.add_subplot(gs_linear[1])

        # Log scale plot (compresses the difference)
        ax2 = fig.add_subplot(gs[0, 1])

        # Just LTC folds (zoomed in to show fold differences)
        ax3 = fig.add_subplot(gs[0, 2])

        # Improvement percentage for each fold
        ax4 = fig.add_subplot(gs[1, 0])

        # CDF of errors for all models
        ax5 = fig.add_subplot(gs[1, 1:])

        # Prepare data
        data = []

        # Add standard KF data
        data.append({
            'model': 'Standard KF',
            'error_2d': baseline_metrics.get('2d_error_mean', 0),
            'error_z': baseline_metrics.get('z_error_mean', 0)
        })

        # Add each fold as a separate entry
        for fold_idx, spatial_metrics in enumerate(fold_results['spatial_metrics']):
            data.append({
                'model': f'Fold {fold_idx + 1}',
                'error_2d': spatial_metrics['2d_error']['mean'],
                'error_z': spatial_metrics['z_error']['mean'],
                'improvement_2d': 100 * (1 - spatial_metrics['2d_error']['mean'] /
                                         baseline_metrics.get('2d_error_mean', 1)),
                'improvement_z': 100 * (1 - spatial_metrics['z_error']['mean'] /
                                        baseline_metrics.get('z_error_mean', 1))
            })

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Helper function to create lighter color variants
        def lighten_color(color, amount=0.5):
            import matplotlib.colors as mc
            import colorsys
            try:
                c = mc.cnames[color]
            except:
                c = color
            c = colorsys.rgb_to_hls(*mc.to_rgb(c))
            return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

        # Define colors
        colors = ['#d62728'] + ['#1f77b4'] * (len(data) - 1)
        light_colors = [lighten_color(c) for c in colors]
        width = 0.35

        # 1. Linear Scale Plot (Broken Axis)
        # Top subplot - for standard KF
        standard_df = df[df['model'] == 'Standard KF']
        ax1_top.bar(0 - width / 2, standard_df['error_2d'], width, color='#d62728', label='2D Error')
        ax1_top.bar(0 + width / 2, standard_df['error_z'], width, color=lighten_color('#d62728'), label='Z Error')
        ax1_top.set_ylim(min(standard_df['error_z'].min() * 0.9, standard_df['error_2d'].min() * 0.9),
                         max(standard_df['error_2d'].max() * 1.1, standard_df['error_z'].max() * 1.1))
        ax1_top.set_xticks([0])
        ax1_top.set_xticklabels(['Standard KF'])

        # Bottom subplot - for LTC folds
        ltc_df = df[df['model'] != 'Standard KF']
        x_ltc = np.arange(len(ltc_df))
        ax1_bottom.bar(x_ltc - width / 2, ltc_df['error_2d'], width, color='#1f77b4', label='2D Error')
        ax1_bottom.bar(x_ltc + width / 2, ltc_df['error_z'], width, color='#2ca02c', label='Z Error')
        ax1_bottom.set_ylim(0, max(ltc_df['error_2d'].max(), ltc_df['error_z'].max()) * 1.2)
        ax1_bottom.set_xticks(x_ltc)
        ax1_bottom.set_xticklabels(ltc_df['model'], rotation=45, ha='right')

        # Add broken axis indicators
        d = .015  # Size of diagonal lines
        kwargs = dict(transform=ax1_top.transAxes, color='k', clip_on=False)
        ax1_top.plot((-d, +d), (-d, +d), **kwargs)
        ax1_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax1_bottom.transAxes)
        ax1_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax1_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        # Add only the value in the center of each bar
        ax1_top.text(0 - width / 2, standard_df['error_2d'].values[0] / 2,
                     f"{standard_df['error_2d'].values[0]:.3f} m",
                     ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        ax1_top.text(0 + width / 2, standard_df['error_z'].values[0] / 2,
                     f"{standard_df['error_z'].values[0]:.3f} m",
                     ha='center', va='center', fontsize=10, color='white', fontweight='bold')

        # Set labels
        ax1_bottom.set_xlabel('Model', fontsize=12)
        ax1_top.set_ylabel('Error (meters)', fontsize=12)
        ax1_bottom.set_ylabel('Error (meters)', fontsize=12)
        ax1_top.set_title('Position Errors (Linear Scale)')

        # Add a single legend
        handles, labels = ax1_top.get_legend_handles_labels()
        ax1_top.legend(handles, labels, loc='upper right')

        # 2. Log Scale Plot
        x = np.arange(len(df['model']))
        ax2.bar(x - width / 2, df['error_2d'], width, label='2D Error', color=[c for c in colors])
        ax2.bar(x + width / 2, df['error_z'], width, label='Z Error', color=[lighten_color(c) for c in colors])

        ax2.set_xticks(x)
        ax2.set_xticklabels(df['model'], rotation=45, ha='right')
        ax2.set_ylabel('Error (meters)', fontsize=12)
        ax2.set_yscale('log')
        ax2.set_title('Position Errors (Log Scale)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)


        # 3. Zoomed LTC Folds Only
        ax3.bar(x_ltc - width / 2, ltc_df['error_2d'], width, label='2D Error', color='#1f77b4')
        ax3.bar(x_ltc + width / 2, ltc_df['error_z'], width, label='Z Error', color='#2ca02c')

        ax3.set_xticks(x_ltc)
        ax3.set_xticklabels(ltc_df['model'], rotation=45, ha='right')
        ax3.set_ylabel('Error (meters)', fontsize=12)
        ax3.set_title('LTC Folds Position Errors (Zoomed)')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)

        # Add only the value in the center of each bar

        # 4. Improvement Percentage Plot
        ax4.bar(x_ltc - width / 2, ltc_df['improvement_2d'], width,
                label='2D Improvement', color='#1f77b4')
        ax4.bar(x_ltc + width / 2, ltc_df['improvement_z'], width,
                label='Z Improvement', color='#2ca02c')

        ax4.set_xticks(x_ltc)
        ax4.set_xticklabels(ltc_df['model'], rotation=45, ha='right')
        ax4.set_ylabel('Improvement (%)', fontsize=12)
        ax4.set_title('Error Reduction vs. Standard KF')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.7)

        # Set y-axis to highlight fold differences
        ax4.set_ylim(95, 100)

        # Add a note about the y-axis
        ax4.text(0.02, 0.05, "Note: y-axis starts at 95%", transform=ax4.transAxes,
                 style='italic', bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})

        # 5. CDF Plot - simplified
        if 'predictions' in fold_results and fold_results['predictions']:
            # Collect individual 2D and Z errors
            all_2d_errors = []
            all_z_errors = []

            for fold_idx, (y_true, y_pred) in enumerate(fold_results['predictions']):
                e_2d = np.sqrt((y_true[:, 6] - y_pred[:, 6]) ** 2 + (y_true[:, 7] - y_pred[:, 7]) ** 2)
                e_z = np.abs(y_true[:, 8] - y_pred[:, 8])

                all_2d_errors.extend(e_2d)
                all_z_errors.extend(e_z)

            # Generate CDF for 2D errors
            all_2d_errors.sort()
            p_2d = np.linspace(0, 1, len(all_2d_errors))
            ax5.plot(all_2d_errors, p_2d, label='2D Error (LTC)', color='#1f77b4')

            # Generate CDF for Z errors
            all_z_errors.sort()
            p_z = np.linspace(0, 1, len(all_z_errors))
            ax5.plot(all_z_errors, p_z, label='Z Error (LTC)', color='#2ca02c')

        # Set primary x-axis to focus on LTC errors
        ax5.set_xlim([0, max(ltc_df['error_2d'].max() * 1.2, ltc_df['error_z'].max() * 1.2)])

        # Create a secondary x-axis for the Standard KF errors
        ax5_secondary = ax5.twiny()

        # Set secondary x-axis to show full range including Standard KF
        max_kf_error = max(baseline_metrics.get('2d_error_mean', 0), baseline_metrics.get('z_error_mean', 0))
        ax5_secondary.set_xlim([0, max_kf_error * 1.1])
        ax5_secondary.set_xlabel('Standard KF Error (meters)', fontsize=12)

        # Add annotation boxes for Standard KF errors
        ax5.annotate(f"Standard KF 2D Error: {baseline_metrics.get('2d_error_mean', 0):.3f} m",
                     xy=(0.95, 0.05), xycoords='axes fraction',
                     ha='right', va='bottom',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.8))

        ax5.annotate(f"Standard KF Z Error: {baseline_metrics.get('z_error_mean', 0):.3f} m",
                     xy=(0.95, 0.15), xycoords='axes fraction',
                     ha='right', va='bottom',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.8))

        ax5.set_xlabel('Error (meters)', fontsize=12)
        ax5.set_ylabel('Cumulative Probability', fontsize=12)
        ax5.set_title('Cumulative Distribution of Position Errors')
        ax5.grid(True, linestyle='--', alpha=0.7)
        ax5.legend()

        # Add overall title
        fig.suptitle('Comprehensive Position Error Analysis: LTC-Enhanced vs. Standard KF',
                     fontsize=18)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure
        plt.savefig(self.save_dir / f'comprehensive_error_analysis{title_suffix}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

    def plot_position_error_comparison(self, fold_results, baseline_metrics, title_suffix=''):
        """
        Create an innovative split-panel visualization showing both the dramatic difference
        and the detailed LTC performance
        """
        fig = plt.figure(figsize=(12, 10))

        # Create a custom grid layout
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1])

        # Top panels - Main comparison with full scale
        ax1_main = fig.add_subplot(gs[0, 0])
        ax2_main = fig.add_subplot(gs[0, 1])

        # Bottom panels - Zoomed view of LTC only
        ax1_zoom = fig.add_subplot(gs[1, 0])
        ax2_zoom = fig.add_subplot(gs[1, 1])

        # Extract data
        ltc_2d_errors = []
        ltc_z_errors = []
        for spatial_metrics in fold_results['spatial_metrics']:
            ltc_2d_errors.append(spatial_metrics['2d_error']['mean'])
            ltc_z_errors.append(spatial_metrics['z_error']['mean'])

        # Calculate statistics
        ltc_2d_mean = np.mean(ltc_2d_errors)
        ltc_2d_std = np.std(ltc_2d_errors)
        ltc_z_mean = np.mean(ltc_z_errors)
        ltc_z_std = np.std(ltc_z_errors)

        # Get standard KF errors
        kf_2d_error = baseline_metrics.get('2d_error_mean', 0)
        kf_z_error = baseline_metrics.get('z_error_mean', 0)

        # Plot data in main panels - Full scale comparison
        models = ['LTC-Enhanced', 'Standard KF']

        # 2D Error - Main panel
        ax1_main.bar([0, 1], [ltc_2d_mean, kf_2d_error], color=['#1f77b4', '#d62728'])
        ax1_main.set_title('2D Position Error', fontsize=14)
        ax1_main.set_ylabel('Error (meters)', fontsize=12)
        ax1_main.set_xticks([0, 1])
        ax1_main.set_xticklabels(models)
        ax1_main.grid(axis='y', linestyle='--', alpha=0.7)

        # Add a connection line to zoom panel
        ax1_main.plot([0, 0], [0, ltc_2d_mean], 'k--', alpha=0.5)

        # Z Error - Main panel
        ax2_main.bar([0, 1], [ltc_z_mean, kf_z_error], color=['#2ca02c', '#d62728'])
        ax2_main.set_title('Z-Direction Error', fontsize=14)
        ax2_main.set_ylabel('Error (meters)', fontsize=12)
        ax2_main.set_xticks([0, 1])
        ax2_main.set_xticklabels(models)
        ax2_main.grid(axis='y', linestyle='--', alpha=0.7)

        # Add a connection line to zoom panel
        ax2_main.plot([0, 0], [0, ltc_z_mean], 'k--', alpha=0.5)

        # Add magnification indicator
        ax1_main.annotate('', xy=(0, ltc_2d_mean), xytext=(0, 0),
                          arrowprops=dict(arrowstyle='<->', color='black', alpha=0.5))
        ax2_main.annotate('', xy=(0, ltc_z_mean), xytext=(0, 0),
                          arrowprops=dict(arrowstyle='<->', color='black', alpha=0.5))

        # Add stats for standard KF
        ax1_main.text(1, kf_2d_error / 2, f"{kf_2d_error:.3f} m", ha='center', va='center',
                      color='white', fontweight='bold', fontsize=12)
        ax2_main.text(1, kf_z_error / 2, f"{kf_z_error:.3f} m", ha='center', va='center',
                      color='white', fontweight='bold', fontsize=12)

        # Zoomed panels - Only LTC data at appropriate scale

        # 2D Error - Zoomed panel with individual fold points
        x_pos = np.arange(len(ltc_2d_errors))
        ax1_zoom.bar(x_pos, ltc_2d_errors, color='#1f77b4', alpha=0.7)
        ax1_zoom.set_title('LTC-Enhanced 2D Error (Detail)', fontsize=12)
        ax1_zoom.set_ylabel('Error (meters)', fontsize=12)
        ax1_zoom.set_xticks(x_pos)
        ax1_zoom.set_xticklabels([f'Fold {i + 1}' for i in range(len(ltc_2d_errors))])
        ax1_zoom.grid(axis='y', linestyle='--', alpha=0.7)

        # Add horizontal line for mean
        ax1_zoom.axhline(y=ltc_2d_mean, color='#1f77b4', linestyle='-', alpha=0.5)

        # Z Error - Zoomed panel with individual fold points
        ax2_zoom.bar(x_pos, ltc_z_errors, color='#2ca02c', alpha=0.7)
        ax2_zoom.set_title('LTC-Enhanced Z Error (Detail)', fontsize=12)
        ax2_zoom.set_ylabel('Error (meters)', fontsize=12)
        ax2_zoom.set_xticks(x_pos)
        ax2_zoom.set_xticklabels([f'Fold {i + 1}' for i in range(len(ltc_z_errors))])
        ax2_zoom.grid(axis='y', linestyle='--', alpha=0.7)

        # Add horizontal line for mean
        ax2_zoom.axhline(y=ltc_z_mean, color='#2ca02c', linestyle='-', alpha=0.5)

        # Add stats boxes to main panels
        ax1_main.text(0, ltc_2d_mean * 1.1, f"Mean: {ltc_2d_mean:.4f} m\nStd: {ltc_2d_std:.4f} m",
                      ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.3",
                                                          fc='white', ec='blue', alpha=0.8))

        ax2_main.text(0, ltc_z_mean * 1.1, f"Mean: {ltc_z_mean:.4f} m\nStd: {ltc_z_std:.4f} m",
                      ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.3",
                                                          fc='white', ec='green', alpha=0.8))

        # For the individual bars in zoomed panels
        for i, (error_2d, error_z) in enumerate(zip(ltc_2d_errors, ltc_z_errors)):
            # For 2D errors
            if error_2d > 0.1:  # Only add text for bars tall enough to accommodate it
                ax1_zoom.text(i, error_2d / 2, f"{error_2d:.3f}",
                              ha='center', va='center', fontsize=10,
                              fontweight="bold", color="white")
            else:
                ax1_zoom.text(i, error_2d + 0.02, f"{error_2d:.3f}",
                              ha='center', va='bottom', fontsize=9)

            # For Z errors
            if error_z > 0.05:  # Only add text for bars tall enough to accommodate it
                ax2_zoom.text(i, error_z / 2, f"{error_z:.3f}",
                              ha='center', va='center', fontsize=10,
                              fontweight="bold", color="white")
            else:
                ax2_zoom.text(i, error_z + 0.01, f"{error_z:.3f}",
                              ha='center', va='bottom', fontsize=9)

        # Add scale comparison indicator
        scale_ratio_2d = kf_2d_error / ltc_2d_mean
        scale_ratio_z = kf_z_error / ltc_z_mean

        fig.text(0.5, 0.01,
                 f"Note: Standard KF error is {scale_ratio_2d:.1f}x larger for 2D and {scale_ratio_z:.1f}x larger for Z",
                 ha='center', fontsize=12, style='italic')

        # Add overall title
        plt.suptitle('Position Error Comparison: Standard KF vs. LTC-Enhanced', fontsize=16, y=0.98)

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

        # Save figure
        plt.savefig(self.save_dir / f'position_error_creative_comparison{title_suffix}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

    def plot_spatial_errors_scientific(self, fold_results, title_suffix=''):
        """
        Create scientific boxplot of 2D and Z-Direction errors across folds
        """
        # Extract individual error values from each fold's predictions
        spatial_data = {
            '2D Error': [],
            'Z Error': []
        }

        # Extract data from predictions
        if fold_results['predictions']:
            for fold_idx, (y_true, y_pred) in enumerate(fold_results['predictions']):
                # Calculate 2D error for each sample
                e_2d = np.sqrt((y_true[:, 6] - y_pred[:, 6]) ** 2 +
                               (y_true[:, 7] - y_pred[:, 7]) ** 2)

                # Calculate Z error for each sample
                e_z = np.abs(y_true[:, 8] - y_pred[:, 8])

                # Store each individual error with its fold number
                for error in e_2d:
                    spatial_data['2D Error'].append({'fold': fold_idx + 1, 'error': error})

                for error in e_z:
                    spatial_data['Z Error'].append({'fold': fold_idx + 1, 'error': error})

        # Convert to DataFrames
        df_2d = pd.DataFrame(spatial_data['2D Error'])
        df_z = pd.DataFrame(spatial_data['Z Error'])

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 2D Error plot
        sns.boxplot(y='error', x='fold', data=df_2d, ax=ax1, palette='Blues')
        sns.stripplot(y='error', x='fold', data=df_2d, ax=ax1, color='black', size=2, alpha=0.4, jitter=True)

        # Z Error plot
        sns.boxplot(y='error', x='fold', data=df_z, ax=ax2, palette='Greens')
        sns.stripplot(y='error', x='fold', data=df_z, ax=ax2, color='black', size=2, alpha=0.4, jitter=True)

        # Calculate statistics for each fold
        fold_stats_2d = df_2d.groupby('fold')['error'].agg(['mean', 'median', 'std', lambda x: np.percentile(x, 95)])
        fold_stats_2d = fold_stats_2d.rename(columns={'<lambda_0>': '95th_percentile'})

        fold_stats_z = df_z.groupby('fold')['error'].agg(['mean', 'median', 'std', lambda x: np.percentile(x, 95)])
        fold_stats_z = fold_stats_z.rename(columns={'<lambda_0>': '95th_percentile'})

        # Overall statistics
        overall_stats_2d = {
            'mean': df_2d['error'].mean(),
            'median': df_2d['error'].median(),
            'std': df_2d['error'].std(),
            '95th_percentile': np.percentile(df_2d['error'], 95)
        }

        overall_stats_z = {
            'mean': df_z['error'].mean(),
            'median': df_z['error'].median(),
            'std': df_z['error'].std(),
            '95th_percentile': np.percentile(df_z['error'], 95)
        }

        # Add statistic boxes in the upper right corners
        stats_text_2d = (
            f"Mean: {overall_stats_2d['mean']:.4f} m\n"
            f"Median: {overall_stats_2d['median']:.4f} m\n"
            f"Std Dev: {overall_stats_2d['std']:.4f} m\n"
            f"95th percentile: {overall_stats_2d['95th_percentile']:.4f} m"
        )

        stats_text_z = (
            f"Mean: {overall_stats_z['mean']:.4f} m\n"
            f"Median: {overall_stats_z['median']:.4f} m\n"
            f"Std Dev: {overall_stats_z['std']:.4f} m\n"
            f"95th percentile: {overall_stats_z['95th_percentile']:.4f} m"
        )

        # Add stats boxes to upper right corners
        ax1.text(0.95, 0.95, stats_text_2d, transform=ax1.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                 fontsize=10)

        ax2.text(0.95, 0.95, stats_text_z, transform=ax2.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                 fontsize=10)

        # Customize plots
        ax1.set_title('2D Position Error (X-Y)', fontsize=16)
        ax1.set_xlabel('Fold', fontsize=14)
        ax1.set_ylabel('Error (meters)', fontsize=14)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

        ax2.set_title('Z-Direction Error (Height)', fontsize=16)
        ax2.set_xlabel('Fold', fontsize=14)
        ax2.set_ylabel('Error (meters)', fontsize=14)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Perform t-test between 2D and Z errors
        t_stat, p_value = stats.ttest_ind(df_2d['error'], df_z['error'], equal_var=False)

        # Add significance annotation
        significance = ""
        if p_value < 0.001:
            significance = "p < 0.001 ***"
        elif p_value < 0.01:
            significance = f"p = {p_value:.3f} **"
        elif p_value < 0.05:
            significance = f"p = {p_value:.3f} *"
        else:
            significance = f"p = {p_value:.3f} (n.s.)"

        # Add the title with p-value
        fig.suptitle(f"Position Errors Across All Folds\n{significance}", fontsize=18)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

        # Save figure
        plt.savefig(self.save_dir / f'spatial_errors_scientific_boxplot{title_suffix}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()