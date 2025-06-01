import numpy as np
import pandas as pd
from model_comparison_visualizer import ModelComparisonVisualizer


def create_component_wise_analysis(model_results, y_true, y_pred_dict, save_dir=None):
    """
    Create component-wise analysis plots for model comparison
    """
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
        create_error_distribution_boxplots(
            visualizer, y_true, y_pred_dict,
            indices, component_names, group_name
        )

    # 2. Create error histograms as in scientific working checklist
    create_error_histograms(
        visualizer, y_true, y_pred_dict,
        component_names, component_groups
    )

    # 3. Create time series plots for selected components
    create_timeseries_plots(
        visualizer, y_true, y_pred_dict,
        component_names, component_groups
    )

    # 4. Create component-wise R² comparison plots
    create_r2_comparison_plots(
        visualizer, model_results, component_groups, component_names
    )

    # 5. Create Q-Q plots for error distribution analysis
    create_qq_plots(
        visualizer, y_true, y_pred_dict,
        component_names
    )


def create_error_distribution_boxplots(visualizer, y_true, y_pred_dict,
                                       indices, component_names, group_name):
    """Create condensed error distribution boxplots for a group of components"""
    # Create a figure with subplots for all components in this group
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_components = len(indices)
    fig, axes = plt.subplots(1, n_components, figsize=(12, 5), sharey=True)

    if n_components == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        component = component_names[idx]
        ax = axes[i]

        # Extract error data for each model
        error_data = []
        model_names = []

        for model, y_pred in y_pred_dict.items():
            if model != "BiLSTM":  # Skip BiLSTM temporarily
                error = y_true[:, idx] - y_pred[:, idx]
                error_data.append(error)
                model_names.append(model)

        # Create boxplot on this axis
        sns.boxplot(data=error_data, ax=ax, palette=[visualizer.model_colors.get(m, f"C{i}") for m in model_names])

        # Add strip plot for individual points
        positions = range(len(model_names))
        for j, (model, errors) in enumerate(zip(model_names, error_data)):
            sns.stripplot(data=errors, ax=ax, color='black', alpha=0.3,
                          jitter=True, size=2, zorder=1, at=j)

        # Set title and adjust appearance
        ax.set_title(component)
        if i == 0:
            ax.set_ylabel("Error")
        ax.set_xticklabels(model_names, rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)

    # Add overall title
    fig.suptitle(f"{group_name} Error Distribution", fontsize=14)
    plt.tight_layout()

    # Save figure
    plt.savefig(visualizer.save_dir / f"{group_name.lower().replace(' ', '_')}_error_boxplot.png",
                dpi=300, bbox_inches='tight')
    plt.close()

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
            time_steps, ground_truth, model_predictions,
            component_names[idx],
            title=f"{group_name} Time Series Analysis",
            filename=f"{group_name.lower().replace(' ', '_')}_timeseries.png",
            y_label="Value"
        )


def create_r2_comparison_plots(visualizer, model_results, component_groups, component_names):
    """Create R² comparison plots for component groups"""
    # Extract R² values for each component and model
    r2_data = {}
    for i, component in enumerate(component_names):
        r2_data[component] = {}
        for model, results in model_results.items():
            # Find R² for this component and model
            for comp_metric in results['component_metrics']:
                if comp_metric['component'] == component:
                    r2_data[component][model] = comp_metric['r2']
                    break

    # Create grouped bar plots for each component group
    for group_name, indices in component_groups.items():
        # Extract components for this group
        group_components = [component_names[i] for i in indices]

        # Extract R² data for these components
        group_r2_data = {comp: r2_data[comp] for comp in group_components}

        # Create bar plot
        visualizer.bar_comparison_plot(
            group_r2_data,
            title=f"{group_name} R² Comparison",
            filename=f"{group_name.lower().replace(' ', '_')}_r2_comparison.png",
            y_label="R²",
            add_improvement=False
        )


def create_qq_plots(visualizer, y_true, y_pred_dict, component_names):
    """Create grouped Q-Q plots for error distribution analysis"""
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    # Define component groups
    component_groups = {
        'Attitude': [0, 1, 2],
        'Velocity': [3, 4, 5],
        'Position': [6, 7, 8],
        'Gyroscope Bias': [9, 10, 11],
        'Accelerometer Bias': [12, 13, 14]
    }

    # Create one figure per group
    for group_name, indices in component_groups.items():
        components = [component_names[idx] for idx in indices]
        n_components = len(components)
        n_models = len([m for m in y_pred_dict.keys() if m != "BiLSTM"])  # Skip BiLSTM temporarily

        # Create a grid of subplots - one row per model, one column per component
        fig, axes = plt.subplots(n_models, n_components, figsize=(n_components * 3, n_models * 2.5))

        # If only one model, ensure axes is still 2D
        if n_models == 1:
            axes = axes.reshape(1, -1)

        # For each model and component, create Q-Q plot
        row_idx = 0
        for model, y_pred in y_pred_dict.items():
            if model != "BiLSTM":  # Skip BiLSTM temporarily
                for col_idx, idx in enumerate(indices):
                    ax = axes[row_idx, col_idx]
                    component = component_names[idx]

                    # Calculate error
                    error = y_true[:, idx] - y_pred[:, idx]

                    # Create Q-Q plot
                    stats.probplot(error, plot=ax)

                    # Set title only for top row
                    if row_idx == 0:
                        ax.set_title(component)

                    # Set model name for leftmost column
                    if col_idx == 0:
                        ax.set_ylabel(f"{model}\nQuantiles")

                    # Improve layout
                    ax.grid(True, linestyle='--', alpha=0.7)

                row_idx += 1

        plt.suptitle(f"{group_name} Q-Q Plots", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            visualizer.save_dir / f"{group_name.lower().replace(' ', '_')}_qq_plots.png",
            dpi=300, bbox_inches='tight')
        plt.close()


def create_error_histograms(visualizer, y_true, y_pred_dict, component_names, component_groups):
    """Create histograms with swarm plots as in the scientific working checklist"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    for group_name, indices in component_groups.items():
        group_components = [component_names[i] for i in indices]
        n_components = len(group_components)

        # Create figure
        fig, axes = plt.subplots(1, n_components, figsize=(n_components * 3.5, 4), sharey=True)

        if n_components == 1:
            axes = [axes]

        # For each component
        for i, (idx, component) in enumerate(zip(indices, group_components)):
            ax = axes[i]

            # Extract error data for each model
            error_data = {}
            for model, y_pred in y_pred_dict.items():
                if model != "BiLSTM":  # Skip BiLSTM temporarily
                    error = y_true[:, idx] - y_pred[:, idx]
                    error_data[model] = error

            # Create violin plot
            sns.violinplot(data=error_data, ax=ax, palette=visualizer.model_colors,
                           inner=None, cut=0, alpha=0.3)

            # Add box plot
            sns.boxplot(data=error_data, ax=ax, palette=visualizer.model_colors,
                        width=0.3, showfliers=False)

            # Add swarm plot
            for j, (model, errors) in enumerate(error_data.items()):
                sns.swarmplot(y=errors, ax=ax, color='black', alpha=0.5, size=3, at=j)

            # Set title and labels
            ax.set_title(component)
            if i == 0:
                ax.set_ylabel("Error")
            ax.grid(True, linestyle='--', alpha=0.7)

        # Add overall title
        fig.suptitle(f"{group_name} Error Distribution", fontsize=14)
        plt.tight_layout()

        # Save figure
        plt.savefig(visualizer.save_dir / f"{group_name.lower().replace(' ', '_')}_error_histogram.png",
                    dpi=300, bbox_inches='tight')
        plt.close()