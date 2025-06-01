from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


class ModelComparisonVisualizer:
    """Core visualization engine for scientific visualization of model comparisons"""

    def __init__(self, save_dir=None):
        """Initialize visualizer with output directory"""
        if save_dir is None:
            current_file_dir = Path(__file__).resolve().parent
            self.save_dir = current_file_dir.parent / "thesis_visualizations"
        else:
            self.save_dir = Path(save_dir)

        # Create directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set default styling for scientific publication standards
        self.setup_plot_style()

    def setup_plot_style(self):
        """Configure plot style for scientific publication standards"""
        # Set consistent styles
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.2)

        # Define custom color palette for models
        self.model_colors = {
            'LTC': '#1f77b4',  # Blue
            'LSTM': '#ff7f0e',  # Orange
            'BiLSTM': '#2ca02c',  # Green
            'Standard KF': '#d62728',  # Red
        }

        # Define consistent marker styles
        self.model_markers = {
            'LTC': 'o',
            'LSTM': 's',
            'BiLSTM': '^',
            'Standard KF': 'x',
        }

    def radar_chart(self, models_data, metrics, title="Model Performance Comparison", filename="radar_chart.png"):
        """Create a radar chart comparing models across multiple metrics"""
        # Number of variables
        N = len(metrics)

        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Initialize the figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], metrics, size=12)

        # Draw the y labels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], size=10)
        plt.ylim(0, 1)

        # Plot each model
        for i, (model_name, values) in enumerate(models_data.items()):
            if model_name == "BiLSTM":  # Skip BiLSTM temporarily
                continue

            values = np.array(values)
            values = np.append(values, values[0])  # Close the loop

            ax.plot(angles, values, linewidth=2, linestyle='solid',
                    label=model_name, color=self.model_colors.get(model_name, f"C{i}"))
            ax.fill(angles, values, alpha=0.1,
                    color=self.model_colors.get(model_name, f"C{i}"))

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # Add title
        plt.title(title, size=15, y=1.1)

        # Save the figure
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def grouped_boxplots(self, data_dict, group_name, component_names, title=None, filename=None, y_label="Error"):
        """Create grouped boxplots for multiple components"""
        fig, axes = plt.subplots(1, len(component_names), figsize=(4 * len(component_names), 5), sharey=True)

        if len(component_names) == 1:
            axes = [axes]

        for i, component in enumerate(component_names):
            ax = axes[i]

            # Extract data for this component
            plot_data = []
            model_names = []

            for model_name, errors in data_dict.items():
                if model_name != "BiLSTM":  # Skip BiLSTM temporarily
                    plot_data.append(errors[component])
                    model_names.append(model_name)

            # Create boxplot
            sns.boxplot(data=plot_data, ax=ax, palette=[self.model_colors.get(m, f"C{i}") for m in model_names])

            # Add swarm plot
            for j, (model, data) in enumerate(zip(model_names, plot_data)):
                sns.swarmplot(y=data, ax=ax, color='black', alpha=0.5, size=3)

            # Set labels
            ax.set_title(component)
            ax.set_xticklabels(model_names, rotation=45)

            if i == 0:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel("")

            ax.grid(True, linestyle='--', alpha=0.7)

        plt.suptitle(title or f"{group_name} Error Distribution", fontsize=14)
        plt.tight_layout()

        # Save the figure
        if filename is None:
            filename = f"{group_name.lower().replace(' ', '_')}_boxplots.png"
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def component_heatmap(self, component_metrics, title="Component-wise Performance", filename="component_heatmap.png",
                          metric="r2"):
        """Create a heatmap of component-wise performance metrics"""
        # Extract unique components and models
        components = set()
        models = set()
        for model_name, metrics in component_metrics.items():
            models.add(model_name)
            for metric_data in metrics:
                components.add(metric_data['component'])

        components = sorted(list(components))
        models = sorted(list(models))

        # Filter out BiLSTM temporarily
        if "BiLSTM" in models:
            models.remove("BiLSTM")

        # Create data matrix
        data = np.zeros((len(models), len(components)))

        for i, model in enumerate(models):
            for j, component in enumerate(components):
                # Find metric for this component and model
                for metric_data in component_metrics[model]:
                    if metric_data['component'] == component:
                        data[i, j] = metric_data.get(metric, 0)
                        break

        # Create figure
        fig, ax = plt.subplots(figsize=(len(components) * 0.6 + 2, len(models) * 0.4 + 2))

        # Create heatmap
        im = ax.imshow(data, cmap='viridis')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(f"{metric.upper()} Score", rotation=-90, va="bottom")

        # Set ticks
        ax.set_xticks(np.arange(len(components)))
        ax.set_yticks(np.arange(len(models)))

        # Set tick labels
        ax.set_xticklabels(components, rotation=45, ha="right")
        ax.set_yticklabels(models)

        # Add labels to cells
        for i in range(len(models)):
            for j in range(len(components)):
                text = ax.text(j, i, f"{data[i, j]:.3f}",
                               ha="center", va="center", color="w")

        # Set title
        ax.set_title(title)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def bar_comparison_plot(self, data_dict, title, filename, y_label="Value", add_improvement=False):
        """Create a bar plot for component-wise metrics"""
        # Get components and models
        components = list(data_dict.keys())

        # Find all unique models across components
        all_models = set()
        for component in components:
            if isinstance(data_dict[component], dict):
                all_models.update(data_dict[component].keys())

        # Filter out BiLSTM temporarily
        models = [m for m in all_models if m != "BiLSTM"]

        if not models:
            return

        # Set up figure
        fig, ax = plt.subplots(figsize=(12, 6))

        n_components = len(components)
        n_models = len(models)
        bar_width = 0.8 / n_models

        # Plot bars
        for i, model in enumerate(models):
            x_positions = np.arange(n_components) + (i - n_models / 2 + 0.5) * bar_width

            values = []
            for component in components:
                if isinstance(data_dict[component], dict) and model in data_dict[component]:
                    values.append(data_dict[component][model])
                else:
                    values.append(0)

            ax.bar(x_positions, values, bar_width, label=model, color=self.model_colors.get(model, f"C{i}"))

        # Set labels
        ax.set_xlabel("Component")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_xticks(np.arange(n_components))
        ax.set_xticklabels(components, rotation=45, ha="right")
        ax.legend()

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Save figure
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def grouped_qq_plots(self, error_data, group_name, component_names, filename=None):
        """Create grouped Q-Q plots for multiple components"""
        # Get models (excluding BiLSTM)
        models = [m for m in error_data.keys() if m != "BiLSTM"]

        if not models:
            return

        # Create figure
        fig, axes = plt.subplots(len(models), len(component_names),
                                 figsize=(3 * len(component_names), 2.5 * len(models)),
                                 squeeze=False)

        # Create Q-Q plots
        for i, model in enumerate(models):
            for j, component in enumerate(component_names):
                # Get error data
                errors = error_data[model][component]

                # Create Q-Q plot
                stats.probplot(errors, plot=axes[i, j])

                # Set titles and labels
                if i == 0:
                    axes[i, j].set_title(component)
                if j == 0:
                    axes[i, j].set_ylabel(f"{model}\nQuantiles")

                # Add grid
                axes[i, j].grid(True, linestyle='--', alpha=0.7)

        # Set overall title
        plt.suptitle(f"{group_name} Q-Q Plots", fontsize=16)
        plt.tight_layout()

        # Save figure
        if filename is None:
            filename = f"{group_name.lower().replace(' ', '_')}_qq_plots.png"
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def cdf_plot(self, data_dict, title, filename, x_label="Error", x_max=None):
        """Create a CDF plot comparing error distributions"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot CDF for each model
        for i, (model, errors) in enumerate(data_dict.items()):
            if model == "BiLSTM":  # Skip BiLSTM temporarily
                continue

            # Sort errors for CDF calculation
            sorted_errors = np.sort(errors)
            # Calculate the CDF values
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

            # Plot the CDF
            ax.plot(sorted_errors, cdf, label=model,
                    color=self.model_colors.get(model, f"C{i}"),
                    linewidth=2)

            # Mark the 50%, 90%, and 95% percentiles
            p50 = np.percentile(errors, 50)
            p90 = np.percentile(errors, 90)
            p95 = np.percentile(errors, 95)

            # Add markers at those percentiles
            ax.plot(p50, 0.5, 'o', color=self.model_colors.get(model, f"C{i}"), markersize=8)
            ax.plot(p90, 0.9, 's', color=self.model_colors.get(model, f"C{i}"), markersize=8)
            ax.plot(p95, 0.95, '^', color=self.model_colors.get(model, f"C{i}"), markersize=8)

        # Add horizontal lines at key percentiles
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)

        # Add percentile labels
        ax.text(0, 0.5, "50%", va='center', ha='right', fontsize=10)
        ax.text(0, 0.9, "90%", va='center', ha='right', fontsize=10)
        ax.text(0, 0.95, "95%", va='center', ha='right', fontsize=10)

        # Set axis limits
        if x_max is not None:
            ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.05)

        # Add labels and title
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Cumulative Probability", fontsize=12)
        ax.set_title(title, fontsize=14)

        # Add legend
        ax.legend(loc='lower right')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Save figure
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def timeseries_plot(self, time_steps, ground_truth, model_predictions, component_name,
                        title, filename, y_label="Value"):
        """Create a time series plot of ground truth vs. model predictions"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot ground truth
        ax.plot(time_steps, ground_truth, 'k-', label='Ground Truth', linewidth=2)

        # Plot predictions for each model (excluding BiLSTM)
        for i, (model, preds) in enumerate(model_predictions.items()):
            if model == "BiLSTM":
                continue

            ax.plot(time_steps, preds, '--',
                    label=f"{model} Prediction",
                    color=self.model_colors.get(model, f"C{i}"),
                    linewidth=1.5, alpha=0.8)

        # Add labels and title
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f"{title} ({component_name})", fontsize=14)

        # Add legend
        ax.legend(loc='best')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Save figure
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()