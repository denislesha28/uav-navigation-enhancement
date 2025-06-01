import numpy as np
import pandas as pd
from pathlib import Path
from model_comparison_visualizer import ModelComparisonVisualizer


def create_performance_metrics_plots(model_results, save_dir=None):
    """
    Create performance metrics plots for model comparison

    Args:
        model_results: Dictionary with model names as keys and result dictionaries as values
        save_dir: Directory to save plots to
    """
    visualizer = ModelComparisonVisualizer(save_dir)

    # 1. Create radar chart of normalized metrics
    create_radar_chart(visualizer, model_results)

    # 2. Create component-wise R² heatmap
    # create_r2_heatmap(visualizer, model_results)

    # 3. Create RMSE comparison plot
    create_rmse_comparison(visualizer, model_results)

    # 4. Create explained variance plot
    create_explained_variance_plot(visualizer, model_results)


def create_radar_chart(visualizer, model_results):
    """Create radar chart for model comparison"""
    # Define metrics to plot
    metrics = ['Position Accuracy', 'Velocity Accuracy', 'Attitude Accuracy',
               'Computational Efficiency', 'Generalization', 'Robustness']

    # Normalize values for each metric (higher is better)
    normalized_results = {}
    for model, results in model_results.items():
        # Extract or calculate values for each metric
        # Note: This is an example calculation, adjust based on your actual metrics
        position_acc = 1 - min(results['spatial_metrics']['2d_error']['mean'] / 10.0, 1.0)
        velocity_acc = results['group_metrics']['velocity']['r2'] if 'velocity' in results['group_metrics'] else 0.5
        attitude_acc = results['group_metrics']['attitude']['r2'] if 'attitude' in results['group_metrics'] else 0.5

        # Computational efficiency and other metrics might need to be calculated
        # or obtained from your training logs
        comp_efficiency = results.get('computational_efficiency', 0.5)
        generalization = results.get('generalization', 0.5)
        robustness = results.get('robustness', 0.5)

        normalized_results[model] = [
            position_acc, velocity_acc, attitude_acc,
            comp_efficiency, generalization, robustness
        ]

    # Create radar chart
    visualizer.radar_chart(
        normalized_results,
        metrics,
        title="Model Performance Comparison",
        filename="model_performance_radar.png"
    )


def create_r2_heatmap(visualizer, model_results):
    """Create component-wise R² heatmap"""
    # Extract components from first model
    components = []
    for comp in model_results[list(model_results.keys())[0]]['component_metrics']:
        components.append(comp['component'])

    # Create visualizer
    visualizer.component_heatmap(
        {model: results['component_metrics'] for model, results in model_results.items()},
        list(model_results.keys()),
        components,
        metric="r2",
        title="Component-wise R² Scores",
        filename="component_r2_heatmap.png"
    )

    # Create a highlighted version showing best model for each component
    create_best_model_heatmap(visualizer, model_results, components)


def create_best_model_heatmap(visualizer, model_results, components):
    """Create a heatmap highlighting the best model for each component"""
    models = list(model_results.keys())

    # Initialize matrix with zeros
    data = np.zeros((len(models), len(components)))

    # Fill matrix with 1s where model is best for component
    for j, component in enumerate(components):
        best_r2 = -float('inf')
        best_model_idx = 0

        for i, model in enumerate(models):
            # Find R² for this component and model
            for comp_metric in model_results[model]['component_metrics']:
                if comp_metric['component'] == component:
                    r2 = comp_metric['r2']
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_idx = i
                    break

        # Mark best model
        data[best_model_idx, j] = 1

    # Create figure
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    cmap = sns.color_palette("viridis", as_cmap=True)
    mask = data == 0  # Mask for non-best models

    sns.heatmap(data, mask=mask, cmap=cmap, ax=ax, cbar=False)

    # Add text showing which model is best and its R² value
    for i, model in enumerate(models):
        for j, component in enumerate(components):
            if data[i, j] == 1:
                # Find R² for this component and model
                for comp_metric in model_results[model]['component_metrics']:
                    if comp_metric['component'] == component:
                        r2 = comp_metric['r2']
                        break

                # Add text
                ax.text(j + 0.5, i + 0.5, f"{r2:.3f}",
                        ha="center", va="center", fontsize=10)

    # Set ticks
    ax.set_xticks(np.arange(len(components)) + 0.5)
    ax.set_yticks(np.arange(len(models)) + 0.5)

    # Set tick labels
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_yticklabels(models)

    # Set title
    ax.set_title("Best Model for Each Component (R²)")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(visualizer.save_dir / "best_model_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_rmse_comparison(visualizer, model_results):
    """Create RMSE comparison plot for different components"""
    # Group RMSE by component group
    component_groups = {
        'Attitude': ['Roll (φE)', 'Pitch (φN)', 'Yaw (φU)'],
        'Velocity': ['East (δυE)', 'North (δυN)', 'Up (δυU)'],
        'Position': ['Latitude (δL)', 'Longitude (δλ)', 'Height (δh)'],
        'Gyroscope Bias': ['X-axis (εx)', 'Y-axis (εy)', 'Z-axis (εz)'],
        'Accelerometer Bias': ['X-axis (∇x)', 'Y-axis (∇y)', 'Z-axis (∇z)']
    }

    for group_name, components in component_groups.items():
        # Create data dict for this group
        data_dict = {}

        for component in components:
            data_dict[component] = {}

            for model, results in model_results.items():
                # Find RMSE for this component and model
                for comp_metric in results['component_metrics']:
                    if comp_metric['component'] == component:
                        data_dict[component][model] = comp_metric['rmse']
                        break

        # Create bar plot
        visualizer.bar_comparison_plot(
            data_dict,
            title=f"{group_name} RMSE Comparison",
            filename=f"{group_name.lower().replace(' ', '_')}_rmse_comparison.png",
            y_label="RMSE",
            add_improvement=True
        )


def create_explained_variance_plot(visualizer, model_results):
    # Extract group metrics
    groups = []
    for model, results in model_results.items():
        for group in results['group_metrics'].keys():
            if group not in groups:
                groups.append(group)

    # Create data dict
    data_dict = {}
    for group in groups:
        data_dict[group] = {}
        for model, results in model_results.items():
            if group in results['group_metrics']:
                # Check if 'explained_variance' key exists
                if 'explained_variance' in results['group_metrics'][group]:
                    data_dict[group][model] = results['group_metrics'][group]['explained_variance']
                else:
                    # Use R² as fallback if explained_variance is not available
                    data_dict[group][model] = results['group_metrics'][group].get('r2', 0.0)

    # Check if we have data to plot
    if not data_dict or all(len(model_data) == 0 for model_data in data_dict.values()):
        print("No explained variance data available, skipping plot")
        return

    # Create bar plot
    visualizer.bar_comparison_plot(
        data_dict,
        title="Component Group Performance (R²)",  # Changed title to reflect what we're actually showing
        filename="group_performance_comparison.png",
        y_label="Score",
        add_improvement=True,
        baseline_model=list(model_results.keys())[0]  # Use first model as baseline
    )
