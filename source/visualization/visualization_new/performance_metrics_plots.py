import numpy as np
import pandas as pd
from pathlib import Path
from model_comparison_visualizer import ModelComparisonVisualizer


def create_performance_metrics_plots(model_results, save_dir=None):
    """Create performance metrics plots for model comparison"""
    visualizer = ModelComparisonVisualizer(save_dir)

    # 1. Create radar chart of normalized metrics
    create_radar_chart(visualizer, model_results)

    # 2. Create component-wise R² heatmap
    create_r2_heatmap(visualizer, model_results)

    # 3. Create RMSE comparison plot
    create_rmse_comparison(visualizer, model_results)

    # 4. Create group metrics comparison
    create_group_metrics_comparison(visualizer, model_results)


def create_radar_chart(visualizer, model_results):
    """Create radar chart for model comparison"""
    # Define metrics to plot
    metrics = ['Position Accuracy', 'Velocity Accuracy', 'Attitude Accuracy',
               'Computational Efficiency', 'Generalization', 'Robustness']

    # Normalize values for each metric (higher is better)
    normalized_results = {}
    for model, results in model_results.items():
        # Extract or calculate values for each metric
        position_acc = 1 - min(results['spatial_metrics']['2d_error']['mean'] / 10.0, 1.0)

        # Extract other metrics from results (with fallbacks)
        velocity_acc = 0.5
        attitude_acc = 0.5
        comp_efficiency = 0.5
        generalization = 0.5
        robustness = 0.5

        if 'group_metrics' in results:
            if 'velocity' in results['group_metrics']:
                velocity_acc = results['group_metrics']['velocity'].get('r2', 0.5)
            if 'attitude' in results['group_metrics']:
                attitude_acc = results['group_metrics']['attitude'].get('r2', 0.5)

        # Computational metrics (if available)
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
    # Extract all components from first model
    first_model = next(iter(model_results.values()))

    # Create visualizer
    visualizer.component_heatmap(
        {model: results['component_metrics'] for model, results in model_results.items()},
        title="Component-wise R² Scores",
        filename="component_r2_heatmap.png",
        metric="r2"
    )


def create_rmse_comparison(visualizer, model_results):
    """Create RMSE comparison plot for different components"""
    # Define component groups
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
                        data_dict[component][model] = comp_metric.get('rmse', 0)
                        break

        # Create bar plot
        visualizer.bar_comparison_plot(
            data_dict,
            title=f"{group_name} RMSE Comparison",
            filename=f"{group_name.lower().replace(' ', '_')}_rmse_comparison.png",
            y_label="RMSE"
        )


def create_group_metrics_comparison(visualizer, model_results):
    """Create comparison of metrics across component groups"""
    # Extract group metrics
    group_metrics = ['attitude', 'velocity', 'position', 'gyroscope_bias', 'accelerometer_bias']
    metric_types = ['r2', 'rmse', 'explained_variance']

    for metric in metric_types:
        data_dict = {}

        for group in group_metrics:
            data_dict[group] = {}

            for model, results in model_results.items():
                if 'group_metrics' in results and group in results['group_metrics']:
                    data_dict[group][model] = results['group_metrics'][group].get(metric, 0)

        # Create bar plot
        visualizer.bar_comparison_plot(
            data_dict,
            title=f"Component Group {metric.upper()} Comparison",
            filename=f"group_{metric}_comparison.png",
            y_label=metric.upper()
        )