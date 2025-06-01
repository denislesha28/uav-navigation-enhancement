import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from model_comparison_visualizer import ModelComparisonVisualizer


def create_component_wise_analysis(model_results, y_true, y_pred_dict, save_dir=None):
    """Create component-wise analysis plots for model comparison"""
    visualizer = ModelComparisonVisualizer(save_dir)

    # Define component groups
    component_groups = {
        'Attitude': [0, 1, 2],
        'Velocity': [3, 4, 5],
        'Position': [6, 7, 8],
        'Gyroscope Bias': [9, 10, 11],
        'Accelerometer Bias': [12, 13, 14]
    }

    component_names = [
        'Roll (φE)', 'Pitch (φN)', 'Yaw (φU)',
        'East (δυE)', 'North (δυN)', 'Up (δυU)',
        'Latitude (δL)', 'Longitude (δλ)', 'Height (δh)',
        'X-axis (εx)', 'Y-axis (εy)', 'Z-axis (εz)',
        'X-axis (∇x)', 'Y-axis (∇y)', 'Z-axis (∇z)'
    ]

    # 1. Create error distribution boxplots for each component group
    for group_name, indices in component_groups.items():
        group_components = [component_names[i] for i in indices]
        create_grouped_boxplots(visualizer, y_true, y_pred_dict, indices, group_components, group_name)
        create_scientific_distribution_plot(visualizer, y_true, y_pred_dict, indices, group_components, group_name)

    # 2. Create Q-Q plots for error distribution analysis (grouped by component group)
    create_grouped_qq_plots(visualizer, y_true, y_pred_dict, component_names, component_groups)

    # 3. Create time series plots for selected components
    create_timeseries_plots(visualizer, y_true, y_pred_dict, component_names, component_groups)

    # 4. Create component-wise R² comparison plots
    create_r2_comparison_plots(visualizer, model_results, component_groups, component_names)


def create_grouped_boxplots(visualizer, y_true, y_pred_dict, indices, component_names, group_name):
    """Create grouped boxplots for a component group"""
    # Prepare data for boxplots
    error_data = {}

    for model, y_pred in y_pred_dict.items():
        error_data[model] = {}
        for i, idx in enumerate(indices):
            component = component_names[i]
            error = y_true[:, idx] - y_pred[:, idx]
            error_data[model][component] = error

    # Create grouped boxplots
    visualizer.grouped_boxplots(
        error_data,
        group_name,
        component_names,
        title=f"{group_name} Error Distribution",
        filename=f"{group_name.lower().replace(' ', '_')}_boxplots.png"
    )


def create_grouped_qq_plots(visualizer, y_true, y_pred_dict, component_names, component_groups):
    """Create grouped Q-Q plots for error distribution analysis"""
    # Prepare error data for Q-Q plots
    error_data = {}

    for model, y_pred in y_pred_dict.items():
        error_data[model] = {}
        for i, component in enumerate(component_names):
            error = y_true[:, i] - y_pred[:, i]
            error_data[model][component] = error

    # Create Q-Q plots for each component group
    for group_name, indices in component_groups.items():
        group_components = [component_names[i] for i in indices]
        visualizer.grouped_qq_plots(
            error_data,
            group_name,
            group_components,
            filename=f"{group_name.lower().replace(' ', '_')}_qq_plots.png"
        )


def create_timeseries_plots(visualizer, y_true, y_pred_dict, component_names, component_groups):
    """Create time series plots for selected components"""
    # Create time steps (sample indices)
    time_steps = np.arange(len(y_true))

    # Select one component from each group for time series analysis
    selected_components = {
        'Attitude': 0,  # Roll
        'Velocity': 3,  # East velocity
        'Position': 8,  # Height
        'Gyroscope Bias': 9,  # X-axis gyro bias
        'Accelerometer Bias': 12  # X-axis accel bias
    }

    for group_name, idx in selected_components.items():
        # Extract ground truth
        ground_truth = y_true[:, idx]

        # Extract predictions for each model
        model_predictions = {}
        for model, y_pred in y_pred_dict.items():
            model_predictions[model] = y_pred[:, idx]

        # Create time series plot
        visualizer.timeseries_plot(
            time_steps,
            ground_truth,
            model_predictions,
            component_names[idx],
            title=f"{group_name} Time Series Analysis",
            filename=f"{group_name.lower().replace(' ', '_')}_timeseries.png",
            y_label="Value"
        )


def create_r2_comparison_plots(visualizer, model_results, component_groups, component_names):
    """Create R² comparison plots for component groups"""
    # Extract R² values for each component and model
    for group_name, indices in component_groups.items():
        # Create data for this group
        r2_data = {}
        group_components = [component_names[i] for i in indices]

        for component in group_components:
            r2_data[component] = {}

            for model, results in model_results.items():
                # Find R² for this component and model
                for comp_metric in results['component_metrics']:
                    if comp_metric['component'] == component and 'r2' in comp_metric:
                        r2_data[component][model] = comp_metric['r2']
                        break

        # Create bar plot
        visualizer.bar_comparison_plot(
            r2_data,
            title=f"{group_name} R² Comparison",
            filename=f"{group_name.lower().replace(' ', '_')}_r2_comparison.png",
            y_label="R²"
        )


def create_scientific_distribution_plot(visualizer, y_true, y_pred_dict, indices, component_names, group_name):
    """Create scientific distribution plots with statistics for a component group"""
    # Prepare data for plotting
    error_data = {}

    for model, y_pred in y_pred_dict.items():
        if model == "BiLSTM":  # Skip BiLSTM temporarily
            continue

        error_data[model] = {}
        for i, idx in enumerate(indices):
            component = component_names[i]
            error = y_true[:, idx] - y_pred[:, idx]
            error_data[model][component] = error

    # Create figure
    fig, axes = plt.subplots(1, len(component_names),
                             figsize=(5 * len(component_names), 5),
                             sharey=True)  # Share y-axis for consistent scale

    if len(component_names) == 1:
        axes = [axes]

    # Define unit information based on component group
    unit = ""
    if group_name == "Position":
        unit = "m"
    elif group_name == "Velocity":
        unit = "m/s"
    elif group_name == "Attitude":
        unit = "rad"
    elif "Bias" in group_name:
        unit = "rad/s" if "Gyroscope" in group_name else "m/s²"

    # Plot each component
    for i, component in enumerate(component_names):
        ax = axes[i] if isinstance(axes, np.ndarray) else axes

        # Extract data for this component
        plot_data = []
        model_names = []

        for model, errors in error_data.items():
            if component in errors:
                plot_data.append(errors[component])
                model_names.append(model)

        # Create violin plot with embedded boxplot
        parts = ax.violinplot(plot_data, showmeans=False, showmedians=False, showextrema=False)

        # Make violins more transparent
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(0.3)

        # Add boxplot inside violin
        bp = ax.boxplot(plot_data, positions=range(1, len(plot_data) + 1),
                        widths=0.3, patch_artist=True,
                        showfliers=False, showcaps=True,
                        medianprops={'color': 'black', 'linewidth': 1.5})

        # Customize boxplot colors
        for j, box in enumerate(bp['boxes']):
            color = visualizer.model_colors.get(model_names[j], f"C{j}")
            box.set(facecolor=color, alpha=0.7)

        # Add individual data points
        for j, data in enumerate(plot_data):
            # Add swarm of points
            y = data
            x = np.random.normal(j + 1, 0.05, size=len(y))
            ax.scatter(x, y, s=3, c='black', alpha=0.4)

        # Set labels
        ax.set_title(component)

        # Only add y-label with units to the leftmost plot
        if i == 0:
            ax.set_ylabel(f"Error ({unit})")

        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add RMSE values at the bottom of the plot where they won't overlap
        for j, (model, data) in enumerate(zip(model_names, plot_data)):
            rmse = np.sqrt(np.mean(data ** 2))
            # Place the RMSE value below the x-axis labels
            ax.annotate(f"RMSE: {rmse:.3f}",
                        xy=(j + 1, ax.get_ylim()[0] - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                        xycoords='data',
                        ha='center', fontsize=8)

    # Add overall title
    fig.suptitle(f"{group_name} Error Distribution", fontsize=16, y=0.98)

    # Add extra space at the bottom for the RMSE annotations
    plt.subplots_adjust(bottom=0.2)

    # Save figure
    filename = f"{group_name.lower().replace(' ', '_')}_scientific_distribution.png"
    plt.savefig(visualizer.save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()