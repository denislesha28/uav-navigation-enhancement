from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        }

        # Define consistent marker styles
        self.model_markers = {
            'LTC': 'o',
            'LSTM': 's',
            'BiLSTM': '^',
        }

    def radar_chart(self, models_data, metrics, title="Model Performance Comparison",
                    filename="radar_chart.png"):
        """
        Create a radar chart comparing models across multiple metrics

        Args:
            models_data: Dictionary with model names as keys and lists of metric values as values
            metrics: List of metric names corresponding to values in models_data
            title: Plot title
            filename: Output filename
        """
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

    def boxplot_with_significance(self, data_dict, metric_name, title, filename,
                                  y_label="Error", significance_test=stats.ttest_ind):
        """
        Create boxplot with individual data points and significance markers

        Args:
            data_dict: Dictionary with model names as keys and lists of values as values
            metric_name: Name of the metric being plotted
            title: Plot title
            filename: Output filename
            y_label: Y-axis label
            significance_test: Statistical test function to use
        """
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame({k: pd.Series(v) for k, v in data_dict.items()})

        # Calculate statistics for table
        stats_df = pd.DataFrame({
            'Mean': df.mean(),
            'Std': df.std(),
            'Median': df.median(),
            'Min': df.min(),
            'Max': df.max()
        })

        # Perform significance tests
        p_values = {}
        for i, model1 in enumerate(df.columns):
            for model2 in df.columns[i + 1:]:
                # Perform test
                stat, p = significance_test(df[model1].dropna(), df[model2].dropna())
                p_values[f"{model1} vs {model2}"] = p

        # Create plot
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])

        # Main boxplot
        ax_box = fig.add_subplot(gs[0, 0])
        sns.boxplot(data=df, ax=ax_box, palette=self.model_colors, showfliers=False)
        sns.stripplot(data=df, ax=ax_box, color='black', alpha=0.5, jitter=True, size=4)

        # Add significance annotations
        y_max = df.max().max()
        y_height = y_max * 1.1
        y_increment = y_max * 0.05

        for i, (pair, p_val) in enumerate(p_values.items()):
            models = pair.split(' vs ')
            x1 = df.columns.get_loc(models[0])
            x2 = df.columns.get_loc(models[1])

            # Calculate significance level
            if p_val < 0.001:
                sig_symbol = '***'
            elif p_val < 0.01:
                sig_symbol = '**'
            elif p_val < 0.05:
                sig_symbol = '*'
            else:
                sig_symbol = 'ns'

            # Draw the line and text
            y_pos = y_height + i * y_increment
            ax_box.plot([x1, x2], [y_pos, y_pos], 'k-')
            ax_box.text((x1 + x2) / 2, y_pos + y_increment / 2, sig_symbol,
                        ha='center', va='bottom')

        # Set labels
        ax_box.set_title(title, fontsize=14)
        ax_box.set_ylabel(y_label, fontsize=12)
        ax_box.set_xlabel('Model', fontsize=12)

        # Stats table
        ax_table = fig.add_subplot(gs[0, 1])
        ax_table.axis('off')
        table = ax_table.table(
            cellText=stats_df.round(4).values,
            rowLabels=stats_df.index,
            colLabels=stats_df.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # P-values table
        ax_pval = fig.add_subplot(gs[1, :])
        ax_pval.axis('off')
        p_df = pd.DataFrame({'p-value': p_values})
        pval_table = ax_pval.table(
            cellText=p_df.round(4).values,
            rowLabels=p_df.index,
            colLabels=p_df.columns,
            cellLoc='center',
            loc='center'
        )
        pval_table.auto_set_font_size(False)
        pval_table.set_fontsize(10)
        pval_table.scale(1, 1.5)

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def component_heatmap(self, component_metrics, models, components,
                          metric="r2", title="Component-wise Performance",
                          filename="component_heatmap.png"):
        """
        Create a heatmap showing model performance for each component

        Args:
            component_metrics: Dictionary with model names as keys and component metrics as values
            models: List of model names
            components: List of component names
            metric: Metric to display (e.g., 'r2', 'rmse')
            title: Plot title
            filename: Output filename
        """
        # Extract values for the specified metric
        data = []
        for model in models:
            model_values = []
            for component in components:
                # Find the metric for this component and model
                for comp_metric in component_metrics[model]:
                    if comp_metric['component'] == component:
                        model_values.append(comp_metric[metric])
                        break
            data.append(model_values)

        # Convert to numpy array
        data = np.array(data)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

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

    def bar_comparison_plot(self, data_dict, title, filename, y_label="Value",
                            add_improvement=True, baseline_model="Standard KF"):
        """
        Create a bar plot comparing models with improvement percentages

        Args:
            data_dict: Dictionary with component names as keys and model values as values
            title: Plot title
            filename: Output filename
            y_label: Y-axis label
            add_improvement: Whether to add improvement percentages
            baseline_model: Name of the model to use as baseline for improvement calculation
        """
        components = list(data_dict.keys())

        # Get all unique models across all components
        all_models = set()
        for comp_data in data_dict.values():
            all_models.update(comp_data.keys())
        models = list(all_models)

        # Check if baseline model exists in the data
        baseline_exists = baseline_model in models
        if add_improvement and not baseline_exists:
            # If baseline doesn't exist but we want improvements, use first model as baseline
            baseline_model = models[0] if models else None

        # Set up the figure and axes
        fig, ax = plt.subplots(figsize=(12, 6))

        # Number of groups and bar width
        n_components = len(components)
        n_models = len(models)

        if n_models == 0 or n_components == 0:
            # No data to plot
            plt.close()
            return

        bar_width = 0.8 / n_models if n_models > 0 else 0.8

        # Create bars
        for i, model in enumerate(models):
            positions = np.arange(n_components) + (i - n_models / 2 + 0.5) * bar_width
            values = []

            # Get value for each component, handling missing model keys
            for comp in components:
                if model in data_dict[comp]:
                    values.append(data_dict[comp][model])
                else:
                    # Use 0 or NaN for missing values
                    values.append(0)

            rects = ax.bar(positions, values, bar_width,
                           label=model, color=self.model_colors.get(model, f"C{i}"))

            # Add improvement percentages if requested
            if add_improvement and model != baseline_model and baseline_model is not None:
                baseline_values = []
                for comp in components:
                    if baseline_model in data_dict[comp]:
                        baseline_values.append(data_dict[comp][baseline_model])
                    else:
                        baseline_values.append(0)  # Default value if baseline data is missing

                for j, (rect, baseline) in enumerate(zip(rects, baseline_values)):
                    if baseline != 0:  # Avoid division by zero
                        # For metrics where lower is better (like error)
                        improvement = (baseline - values[j]) / abs(baseline) * 100
                        # Only annotate if improvement is significant
                        if abs(improvement) > 5:
                            height = rect.get_height()
                            # Place annotation above the bar
                            if height > 0:
                                y_pos = height * 1.05
                            else:
                                y_pos = 0.05

                            ax.annotate(f"{improvement:.1f}%",
                                        xy=(positions[j], y_pos),
                                        ha='center', va='bottom',
                                        color='green' if improvement > 0 else 'red',
                                        fontsize=8)

        # Add labels and title
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(np.arange(n_components))
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.legend()

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def cdf_plot(self, data_dict, title, filename, x_label="Error", x_max=None):
        """
        Create a CDF plot comparing error distributions

        Args:
            data_dict: Dictionary with model names as keys and error lists as values
            title: Plot title
            filename: Output filename
            x_label: X-axis label
            x_max: Maximum x value to display
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot CDF for each model
        for i, (model, errors) in enumerate(data_dict.items()):
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

            # Add annotations for main model
            if model == 'LTC':
                ax.annotate(f"{p50:.3f}", xy=(p50, 0.5), xytext=(5, 5),
                            textcoords='offset points', fontsize=8)
                ax.annotate(f"{p90:.3f}", xy=(p90, 0.9), xytext=(5, 5),
                            textcoords='offset points', fontsize=8)
                ax.annotate(f"{p95:.3f}", xy=(p95, 0.95), xytext=(5, 5),
                            textcoords='offset points', fontsize=8)

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

    # Define line styles and widths for clarity
    line_styles = {
        'Ground Truth': {'color': 'black', 'linewidth': 2.5, 'linestyle': '-'},
        'LTC': {'color': '#1f77b4', 'linewidth': 1.8, 'linestyle': '--'},
        'LSTM': {'color': '#ff7f0e', 'linewidth': 1.8, 'linestyle': '-.'},
        'Standard KF': {'color': '#d62728', 'linewidth': 1.8, 'linestyle': ':'}
    }

    # Plot ground truth with emphasized styling
    ax.plot(time_steps, ground_truth, **line_styles['Ground Truth'], label='Ground Truth')

    # Add shaded region for errors
    for model, preds in model_predictions.items():
        if model == "BiLSTM":  # Skip BiLSTM temporarily
            continue

        # Calculate absolute error
        error = np.abs(ground_truth - preds)

        # Plot prediction
        ax.plot(time_steps, preds, **line_styles.get(model, {'color': 'gray', 'linestyle': '--'}),
                label=f"{model}")

        # Add lightly shaded error region
        color = line_styles.get(model, {'color': 'gray'})['color']
        ax.fill_between(time_steps, ground_truth - error, ground_truth + error,
                        alpha=0.1, color=color)

    # Add labels with units
    unit = ""
    if "Position" in title:
        unit = " (m)"
    elif "Velocity" in title:
        unit = " (m/s)"
    elif "Attitude" in title:
        unit = " (rad)"
    elif "Bias" in title:
        unit = " (rad/s)" if "Gyroscope" in title else " (m/s²)"

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel(f"{y_label}{unit}", fontsize=12)
    ax.set_title(f"{title} ({component_name})", fontsize=14)

    # Improve legend for clarity
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='black')

    # Add vertical grid for better readability of time steps
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.grid(axis='x', linestyle='-', alpha=0.3)

    # Add text annotation explaining the plot
    model_rmse = {}
    for model, preds in model_predictions.items():
        if model == "BiLSTM":
            continue
        model_rmse[model] = np.sqrt(np.mean((ground_truth - preds) ** 2))

    info_text = "RMSE Comparison:\n"
    for model, rmse in model_rmse.items():
        info_text += f"{model}: {rmse:.4f}{unit}\n"

    # Place text box in the upper left corner
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Save figure
    plt.tight_layout()
    plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()