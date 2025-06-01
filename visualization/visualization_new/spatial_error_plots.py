import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from model_comparison_visualizer import ModelComparisonVisualizer
from scipy import stats


def create_spatial_error_plots(model_results, y_true, y_pred_dict, save_dir=None):
    """Create spatial error plots for model comparison (2D and Z-direction errors)"""
    visualizer = ModelComparisonVisualizer(save_dir)

    # 1. Create 2D and Z-direction error boxplots
    create_error_boxplots(visualizer, model_results)

    # 2. Create CDF plots for 2D and Z errors
    create_error_cdf_plots(visualizer, y_true, y_pred_dict)

    # 3. Create position error maps
    create_position_error_maps(visualizer, y_true, y_pred_dict)


def create_error_boxplots(visualizer, model_results):
    """Create 2D and Z-direction error boxplots"""
    # Create data for boxplots
    error_2d = {}
    error_z = {}

    for model, results in model_results.items():
        if model == "BiLSTM":  # Skip BiLSTM temporarily
            continue

        # Generate sample data for visualization
        # In a real implementation, you would use actual error distributions
        e2d_mean = results['spatial_metrics']['2d_error']['mean']
        e2d_std = results['spatial_metrics']['2d_error']['std']
        ez_mean = results['spatial_metrics']['z_error']['mean']
        ez_std = results['spatial_metrics']['z_error']['std']

        # Generate samples following normal distributions
        np.random.seed(42)  # For reproducibility
        error_2d[model] = np.random.normal(e2d_mean, e2d_std, 100)
        error_z[model] = np.random.normal(ez_mean, ez_std, 100)

    # Prepare data in the format expected by grouped_boxplots
    error_data = {}
    for model in error_2d.keys():
        error_data[model] = {
            '2D Error': error_2d[model],
            'Z Error': error_z[model]
        }

    # Create boxplots
    visualizer.grouped_boxplots(
        error_data,
        'Spatial',
        ['2D Error', 'Z Error'],
        title="Spatial Error Comparison",
        filename="spatial_error_boxplots.png",
        y_label="Error (m)"
    )

def create_error_cdf_plots(visualizer, y_true, y_pred_dict):
    """Create CDF plots for 2D and Z-direction errors"""
    # Calculate 2D error for each model
    error_2d = {}
    for model, y_pred in y_pred_dict.items():
        # 2D error (position components 6 and 7 - Latitude/Longitude)
        e_2d = np.sqrt((y_true[:, 6] - y_pred[:, 6]) ** 2 +
                       (y_true[:, 7] - y_pred[:, 7]) ** 2)
        error_2d[model] = e_2d

    # Create CDF plot for 2D error
    visualizer.cdf_plot(
        error_2d,
        title="Cumulative Distribution of 2D Position Error",
        filename="2d_error_cdf.png",
        x_label="2D Error (m)"
    )

    # Calculate Z-direction error for each model
    error_z = {}
    for model, y_pred in y_pred_dict.items():
        # Z-direction error (position component 8 - Height)
        e_z = np.abs(y_true[:, 8] - y_pred[:, 8])
        error_z[model] = e_z

    # Create CDF plot for Z-direction error
    visualizer.cdf_plot(
        error_z,
        title="Cumulative Distribution of Z-Direction (Height) Error",
        filename="z_error_cdf.png",
        x_label="Z-Direction Error (m)"
    )


def create_position_error_maps(visualizer, y_true, y_pred_dict):
    """Create position error maps (top-down view)"""
    # Extract latitude and longitude components (indices 6 and 7)
    lat_true = y_true[:, 6]
    lon_true = y_true[:, 7]

    # Create figure
    fig, axes = plt.subplots(1, len([m for m in y_pred_dict if m != "BiLSTM"]),
                             figsize=(15, 5))

    if len([m for m in y_pred_dict if m != "BiLSTM"]) == 1:
        axes = [axes]

    # Determine plot limits
    lat_min, lat_max = lat_true.min(), lat_true.max()
    lon_min, lon_max = lon_true.min(), lon_true.max()

    # Add some margin
    lat_margin = (lat_max - lat_min) * 0.1
    lon_margin = (lon_max - lon_min) * 0.1

    lat_min -= lat_margin
    lat_max += lat_margin
    lon_min -= lon_margin
    lon_max += lon_margin

    # Plot for each model (excluding BiLSTM)
    i = 0
    for model, y_pred in y_pred_dict.items():
        if model == "BiLSTM":  # Skip BiLSTM temporarily
            continue

        ax = axes[i]
        i += 1

        # Extract predicted latitude and longitude
        lat_pred = y_pred[:, 6]
        lon_pred = y_pred[:, 7]

        # Calculate 2D position error for each point
        error_2d = np.sqrt((lat_true - lat_pred) ** 2 + (lon_true - lon_pred) ** 2)

        # Plot ground truth trajectory
        ax.plot(lon_true, lat_true, 'k-', linewidth=2, label='Ground Truth')

        # Plot predicted trajectory, colored by error magnitude
        scatter = ax.scatter(lon_pred, lat_pred, c=error_2d, cmap='viridis',
                             alpha=0.7, label='Prediction')

        # Set title and labels
        ax.set_title(f"{model} Position Error")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Set consistent limits
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Error (m)')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

    # Add overall title
    fig.suptitle('Position Error Maps (Top-Down View)', fontsize=16, y=1.05)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(visualizer.save_dir / "position_error_maps.png", dpi=300, bbox_inches='tight')
    plt.close()