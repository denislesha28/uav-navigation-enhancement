import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


class TrajectoryVisualizer:
    def __init__(self, ground_truth, kf_estimate, lnn_enhanced_estimate, timestamps=None):
        """
        Visualize and compare flight trajectories.

        Args:
            ground_truth: nx3 array of ground truth positions
            kf_estimate: nx3 array of Kalman filter estimated positions
            lnn_enhanced_estimate: nx3 array of LNN-enhanced positions
            timestamps: Optional time values for error plotting
        """
        self.ground_truth = ground_truth
        self.kf_estimate = kf_estimate
        self.lnn_enhanced_estimate = lnn_enhanced_estimate
        self.timestamps = timestamps

        # Calculate errors
        self.kf_error = np.linalg.norm(ground_truth - kf_estimate, axis=1)
        self.lnn_error = np.linalg.norm(ground_truth - lnn_enhanced_estimate, axis=1)

        # Calculate improvement
        self.improvement = self.kf_error - self.lnn_error
        self.percent_improvement = 100 * (self.kf_error - self.lnn_error) / self.kf_error

        # Statistics
        self.kf_rmse = np.sqrt(np.mean(np.square(self.kf_error)))
        self.lnn_rmse = np.sqrt(np.mean(np.square(self.lnn_error)))

    def plot_3d_comparison(self, title="UAV Trajectory Comparison"):
        """Create a 3D plot comparing all trajectories"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the trajectories
        ax.plot(self.ground_truth[:, 0], self.ground_truth[:, 1], self.ground_truth[:, 2],
                'k-', linewidth=2, label='Ground Truth')
        ax.plot(self.kf_estimate[:, 0], self.kf_estimate[:, 1], self.kf_estimate[:, 2],
                'g-', linewidth=1.5, label='Kalman Filter')
        ax.plot(self.lnn_enhanced_estimate[:, 0], self.lnn_enhanced_estimate[:, 1], self.lnn_enhanced_estimate[:, 2],
                'b-', linewidth=1.5, label='Enhanced Navigation')

        # Add markers for start/end points
        ax.scatter(self.ground_truth[0, 0], self.ground_truth[0, 1], self.ground_truth[0, 2],
                   c='k', marker='o', s=100, label='Start')
        ax.scatter(self.ground_truth[-1, 0], self.ground_truth[-1, 1], self.ground_truth[-1, 2],
                   c='r', marker='o', s=100, label='End')

        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)

        # Add legend
        ax.legend()

        # Add error statistics as text
        error_text = (f"KF RMSE: {self.kf_rmse:.2f}m\n"
                      f"LNN RMSE: {self.lnn_rmse:.2f}m\n"
                      f"Improvement: {((self.kf_rmse - self.lnn_rmse) / self.kf_rmse * 100):.2f}%")
        ax.text2D(0.05, 0.05, error_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))

        return fig, ax

    def plot_error_comparison(self):
        """Plot position errors over time"""
        if self.timestamps is None:
            x = np.arange(len(self.kf_error))
        else:
            x = self.timestamps

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Error plot
        ax1.plot(x, self.kf_error, 'g-', label='Kalman Filter Error')
        ax1.plot(x, self.lnn_error, 'b-', label='LNN Enhanced Error')
        ax1.set_ylabel('Position Error (m)')
        ax1.set_title('Position Error Comparison')
        ax1.legend()
        ax1.grid(True)

        # Improvement plot
        ax2.plot(x, self.improvement, 'r-', label='Absolute Improvement')
        ax2.plot(x, self.percent_improvement, 'm-', label='Percent Improvement')
        ax2.set_xlabel('Time' if self.timestamps is None else 'Time (s)')
        ax2.set_ylabel('Improvement (m / %)')
        ax2.set_title('LNN Enhancement Impact')
        ax2.legend()
        ax2.grid(True)

        # Add overall statistics
        stats_text = (f"Mean KF Error: {np.mean(self.kf_error):.2f}m\n"
                      f"Mean LNN Error: {np.mean(self.lnn_error):.2f}m\n"
                      f"Mean Improvement: {np.mean(self.improvement):.2f}m ({np.mean(self.percent_improvement):.2f}%)")
        ax1.text(0.05, 0.05, stats_text, transform=ax1.transAxes,
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        plt.tight_layout()
        return fig, (ax1, ax2)

    def create_animation(self, interval=50, save_path=None):
        """Create an animated visualization of the trajectories"""
        fig = plt.figure(figsize=(15, 8))

        # Set up 3D trajectory plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('UAV Trajectory')

        # Set up error plot
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Position Error')
        ax2.grid(True)

        # Find axis limits for trajectory plot
        x_min, x_max = np.min(self.ground_truth[:, 0]), np.max(self.ground_truth[:, 0])
        y_min, y_max = np.min(self.ground_truth[:, 1]), np.max(self.ground_truth[:, 1])
        z_min, z_max = np.min(self.ground_truth[:, 2]), np.max(self.ground_truth[:, 2])

        # Add some padding
        x_pad = (x_max - x_min) * 0.1
        y_pad = (y_max - y_min) * 0.1
        z_pad = (z_max - z_min) * 0.1

        ax1.set_xlim(x_min - x_pad, x_max + x_pad)
        ax1.set_ylim(y_min - y_pad, y_max + y_pad)
        ax1.set_zlim(z_min - z_pad, z_max + z_pad)

        # Initialize plots
        gt_line, = ax1.plot([], [], [], 'k-', label='Ground Truth')
        kf_line, = ax1.plot([], [], [], 'g-', label='Kalman Filter')
        lnn_line, = ax1.plot([], [], [], 'b-', label='LNN Enhanced')

        gt_point = ax1.scatter([], [], [], c='k', marker='o', s=50)
        kf_point = ax1.scatter([], [], [], c='g', marker='o', s=50)
        lnn_point = ax1.scatter([], [], [], c='b', marker='o', s=50)

        kf_error_line, = ax2.plot([], [], 'g-', label='KF Error')
        lnn_error_line, = ax2.plot([], [], 'b-', label='LNN Error')

        # Add legend
        ax1.legend()
        ax2.legend()

        # Current stats text
        stats_text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes,
                              fontsize=10, bbox=dict(facecolor='white', alpha=0.7),
                              verticalalignment='top')

        def init():
            gt_line.set_data([], [])
            gt_line.set_3d_properties([])
            kf_line.set_data([], [])
            kf_line.set_3d_properties([])
            lnn_line.set_data([], [])
            lnn_line.set_3d_properties([])

            gt_point._offsets3d = (np.array([]), np.array([]), np.array([]))
            kf_point._offsets3d = (np.array([]), np.array([]), np.array([]))
            lnn_point._offsets3d = (np.array([]), np.array([]), np.array([]))

            kf_error_line.set_data([], [])
            lnn_error_line.set_data([], [])

            stats_text.set_text('')

            return (gt_line, kf_line, lnn_line, gt_point, kf_point, lnn_point,
                    kf_error_line, lnn_error_line, stats_text)

        def update(frame):
            # Update trajectory history
            history_len = 100  # Show this many points of history
            start_idx = max(0, frame - history_len)

            gt_line.set_data(self.ground_truth[start_idx:frame + 1, 0], self.ground_truth[start_idx:frame + 1, 1])
            gt_line.set_3d_properties(self.ground_truth[start_idx:frame + 1, 2])

            kf_line.set_data(self.kf_estimate[start_idx:frame + 1, 0], self.kf_estimate[start_idx:frame + 1, 1])
            kf_line.set_3d_properties(self.kf_estimate[start_idx:frame + 1, 2])

            lnn_line.set_data(self.lnn_enhanced_estimate[start_idx:frame + 1, 0],
                              self.lnn_enhanced_estimate[start_idx:frame + 1, 1])
            lnn_line.set_3d_properties(self.lnn_enhanced_estimate[start_idx:frame + 1, 2])

            # Update current position points
            gt_point._offsets3d = (
            [self.ground_truth[frame, 0]], [self.ground_truth[frame, 1]], [self.ground_truth[frame, 2]])
            kf_point._offsets3d = (
            [self.kf_estimate[frame, 0]], [self.kf_estimate[frame, 1]], [self.kf_estimate[frame, 2]])
            lnn_point._offsets3d = ([self.lnn_enhanced_estimate[frame, 0]], [self.lnn_enhanced_estimate[frame, 1]],
                                    [self.lnn_enhanced_estimate[frame, 2]])

            # Update error plot
            kf_error_line.set_data(np.arange(frame + 1), self.kf_error[:frame + 1])
            lnn_error_line.set_data(np.arange(frame + 1), self.lnn_error[:frame + 1])

            # Adjust error plot limits if needed
            if frame > 10:
                ax2.set_xlim(0, frame + 10)
                max_error = max(np.max(self.kf_error[:frame + 1]), np.max(self.lnn_error[:frame + 1]))
                ax2.set_ylim(0, max_error * 1.1)

            # Update statistics text
            current_kf_error = self.kf_error[frame]
            current_lnn_error = self.lnn_error[frame]
            current_improvement = current_kf_error - current_lnn_error
            current_percent = (current_improvement / current_kf_error * 100) if current_kf_error > 0 else 0

            stats_text.set_text(f"Current Frame: {frame}\n"
                                f"KF Error: {current_kf_error:.2f}m\n"
                                f"LNN Error: {current_lnn_error:.2f}m\n"
                                f"Improvement: {current_improvement:.2f}m ({current_percent:.2f}%)")

            return (gt_line, kf_line, lnn_line, gt_point, kf_point, lnn_point,
                    kf_error_line, lnn_error_line, stats_text)

        ani = FuncAnimation(fig, update, frames=len(self.ground_truth),
                            init_func=init, blit=True, interval=interval)

        if save_path:
            ani.save(save_path, writer='ffmpeg', fps=30, dpi=200)

        plt.tight_layout()
        return ani