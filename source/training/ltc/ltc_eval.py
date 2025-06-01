import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.model_selection import KFold
import logging
from pathlib import Path
from scipy import stats

# Import from local modules
from training.lstm.lstm_eval import calculate_spatial_errors
from training.ltc.ltc_training import EnhancedNavigationLTC, NavigationDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LTCEvaluator:
    """Minimalistic class for k-fold cross-validation and evaluation of LTC models using Kendall's tau"""

    def __init__(self, X, y, device, n_splits=5, save_dir=None):
        self.X = X
        self.y = y
        self.device = device
        self.n_splits = n_splits

        if save_dir is None:
            current_file_dir = Path(__file__).resolve().parent
            self.save_dir = current_file_dir.parent / "cv_results"
        else:
            self.save_dir = Path(save_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.component_names = {
            0: 'Roll (φE)', 1: 'Pitch (φN)', 2: 'Yaw (φU)',
            3: 'East (δυE)', 4: 'North (δυN)', 5: 'Up (δυU)',
            6: 'Latitude (δL)', 7: 'Longitude (δλ)', 8: 'Height (δh)',
            9: 'X-axis (εx)', 10: 'Y-axis (εy)', 11: 'Z-axis (εz)',
            12: 'X-axis (∇x)', 13: 'Y-axis (∇y)', 14: 'Z-axis (∇z)'
        }

        # Store results from all folds
        self.fold_results = {
            'metrics': [],
            'component_metrics': [],
            'group_metrics': [],
            'spatial_metrics': []
        }

    def create_model(self, input_size, hidden_size=128, output_size=15):
        """Create a new LTC model instance"""
        model = EnhancedNavigationLTC(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size
        ).to(self.device)
        return model

    def evaluate_fold(self, model, test_loader):
        """Evaluate model on a single fold"""
        model.eval()
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                all_targets.append(target.cpu().numpy())
                all_predictions.append(output.cpu().numpy())

        y_true = np.vstack(all_targets)
        y_pred = np.vstack(all_predictions)

        # Calculate overall Kendall's tau
        overall_tau, overall_p = stats.kendalltau(y_true.flatten(), y_pred.flatten())

        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'kendall_tau': overall_tau,
            'explained_variance': explained_variance_score(y_true, y_pred)
        }

        # Calculate spatial metrics
        spatial_metrics = calculate_spatial_errors(y_true, y_pred)

        # Calculate group metrics with Kendall's tau
        group_metrics = self.calculate_group_metrics_kendall(y_true, y_pred)

        # Calculate component-wise metrics
        component_metrics = []
        for i in range(y_true.shape[1]):
            tau, p_value = stats.kendalltau(y_true[:, i], y_pred[:, i])
            component_metrics.append({
                'component': self.component_names[i],
                'rmse': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
                'kendall_tau': tau,
                'mae': mean_absolute_error(y_true[:, i], y_pred[:, i])
            })

        return metrics, component_metrics, group_metrics, spatial_metrics

    def calculate_group_metrics_kendall(self, y_true, y_pred):
        """Calculate group metrics using Kendall's tau"""
        groups = {
            'attitude': slice(0, 3),
            'velocity': slice(3, 6),
            'position': slice(6, 9),
            'gyro_bias': slice(9, 12),
            'accel_bias': slice(12, 15)
        }

        group_metrics = {}
        for group_name, indices in groups.items():
            group_true = y_true[:, indices].flatten()
            group_pred = y_pred[:, indices].flatten()

            tau, p_value = stats.kendalltau(group_true, group_pred)
            rmse = np.sqrt(mean_squared_error(group_true, group_pred))
            mae = mean_absolute_error(group_true, group_pred)

            group_metrics[group_name] = {
                'kendall_tau': tau,
                'rmse': rmse,
                'mae': mae
            }

        return group_metrics

    def run_cross_validation(self, model_params=None, batch_size=16, load_existing=False):
        """Run k-fold cross-validation"""
        if model_params is None:
            model_params = {'hidden_size': 128}

        dataset = NavigationDataset(self.X, self.y, self.device)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(kf.split(range(len(dataset)))):
            logger.info(f"Evaluating fold {fold + 1}/{self.n_splits}")

            test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size)

            # Load model
            model_path = self.save_dir / f"fold_{fold + 1}_model.pth"
            if load_existing and model_path.exists():
                logger.info(f"Loading existing model from {model_path}")
                model = self.create_model(
                    input_size=self.X.shape[-1],
                    hidden_size=model_params['hidden_size'],
                    output_size=self.y.shape[-1]
                )
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                logger.error(f"Model not found at {model_path}. Please train models for all folds first.")
                return

            # Evaluate model
            metrics, component_metrics, group_metrics, spatial_metrics = self.evaluate_fold(model, test_loader)

            # Store results
            self.fold_results['metrics'].append(metrics)
            self.fold_results['component_metrics'].append(component_metrics)
            self.fold_results['group_metrics'].append(group_metrics)
            self.fold_results['spatial_metrics'].append(spatial_metrics)

            # Log results
            logger.info(f"Fold {fold + 1} Results:")
            logger.info(f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"Kendall's τ: {metrics['kendall_tau']:.4f}")
            logger.info(
                f"2D Error: {spatial_metrics['2d_error']['mean']:.4f} ± {spatial_metrics['2d_error']['std']:.4f} m")
            logger.info(
                f"Z Error: {spatial_metrics['z_error']['mean']:.4f} ± {spatial_metrics['z_error']['std']:.4f} m")

        # Generate summary results
        self.generate_summary_results()

    def generate_summary_results(self):
        """Generate summary results without plotting"""
        if not self.fold_results['metrics']:
            logger.error("No fold results to summarize. Run cross-validation first.")
            return

        # Overall metrics summary
        overall_metrics = pd.DataFrame(self.fold_results['metrics'])
        mean_metrics = overall_metrics.mean()
        std_metrics = overall_metrics.std()

        # Component metrics summary
        component_summary = {}
        for i in range(len(self.component_names)):
            component_name = self.component_names[i]
            tau_values = []
            rmse_values = []

            for fold_idx in range(self.n_splits):
                fold_components = self.fold_results['component_metrics'][fold_idx]
                for comp in fold_components:
                    if comp['component'] == component_name:
                        tau_values.append(comp['kendall_tau'])
                        rmse_values.append(comp['rmse'])
                        break

            component_summary[component_name] = {
                'kendall_tau_mean': np.mean(tau_values),
                'kendall_tau_std': np.std(tau_values),
                'rmse_mean': np.mean(rmse_values),
                'rmse_std': np.std(rmse_values)
            }

        # Group metrics summary
        group_summary = {}
        group_names = {'attitude': 'Attitude', 'velocity': 'Velocity', 'position': 'Position',
                       'gyro_bias': 'Gyro Bias', 'accel_bias': 'Accel Bias'}

        for group_name in group_names.keys():
            tau_values = []
            rmse_values = []

            for fold_idx in range(self.n_splits):
                fold_groups = self.fold_results['group_metrics'][fold_idx]
                if group_name in fold_groups:
                    tau_values.append(fold_groups[group_name]['kendall_tau'])
                    rmse_values.append(fold_groups[group_name]['rmse'])

            group_summary[group_names[group_name]] = {
                'kendall_tau_mean': np.mean(tau_values),
                'kendall_tau_std': np.std(tau_values),
                'rmse_mean': np.mean(rmse_values),
                'rmse_std': np.std(rmse_values)
            }

        # Spatial metrics summary
        spatial_summary = {
            '2D Error': {
                'mean': np.mean([sm['2d_error']['mean'] for sm in self.fold_results['spatial_metrics']]),
                'std': np.mean([sm['2d_error']['std'] for sm in self.fold_results['spatial_metrics']])
            },
            'Z Error': {
                'mean': np.mean([sm['z_error']['mean'] for sm in self.fold_results['spatial_metrics']]),
                'std': np.mean([sm['z_error']['std'] for sm in self.fold_results['spatial_metrics']])
            }
        }

        # Save results
        results = {
            'overall_metrics': {
                'mean': mean_metrics.to_dict(),
                'std': std_metrics.to_dict()
            },
            'component_metrics': component_summary,
            'group_metrics': group_summary,
            'spatial_metrics': spatial_summary
        }

        # Save to file
        import json
        with open(self.save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)

        logger.info("Evaluation complete. Results saved to evaluation_results.json")

        # Print summary
        print("\n=== LTC Evaluation Summary ===")
        print(f"Overall RMSE: {mean_metrics['rmse']:.4f} ± {std_metrics['rmse']:.4f}")
        print(f"Overall Kendall's τ: {mean_metrics['kendall_tau']:.4f} ± {std_metrics['kendall_tau']:.4f}")
        print(
            f"2D Position Error: {spatial_summary['2D Error']['mean']:.4f} ± {spatial_summary['2D Error']['std']:.4f} m")
        print(
            f"Z-Direction Error: {spatial_summary['Z Error']['mean']:.4f} ± {spatial_summary['Z Error']['std']:.4f} m")

        return results


def run_ltc_k_fold_evaluation(X, y, device, n_splits=5, load_existing=True):
    """
    Run k-fold cross-validation for LTC models

    Args:
        X: Input features array
        y: Target labels array
        device: Torch device
        n_splits: Number of folds
        load_existing: Whether to load existing models
    """
    evaluator = LTCEvaluator(X, y, device, n_splits=n_splits)
    evaluator.run_cross_validation(load_existing=load_existing)
    return evaluator