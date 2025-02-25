import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import logging
import numpy as np
from collections import defaultdict
from sklearn.metrics import r2_score, explained_variance_score

from training.lstm.lstm_eval import PerformanceVisualizer, evaluate_test_set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SensorFusionLSTM(nn.Module):
    def __init__(self,
                 input_size=128,  # CNN output
                 hidden_size=64,  # Reduced to match input scaling
                 num_layers=2,
                 output_size=15,  # Error state vector size
                 dropout=0.1):  # Original dropout maintained
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out)
        return predictions


class NavigationDataset(Dataset):
    def __init__(self, data, target, device):
        self.data = torch.as_tensor(data, dtype=torch.float32).to(device)
        self.labels = torch.as_tensor(target, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MetricsTracker:
    def __init__(self):
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'mae': [],
            'rmse': [],
            'r2': [],
            'explained_variance': []
        }

    def update(self, outputs, targets):
        outputs_np = outputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        mae = self.mae(outputs, targets).item()
        mse = self.mse(outputs, targets).item()
        rmse = np.sqrt(mse)

        r2 = r2_score(targets_np.flatten(), outputs_np.flatten())

        exp_var = explained_variance_score(targets_np.flatten(), outputs_np.flatten())

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'explained_variance': exp_var
        }

def train_lstm(X, y, device):
    logger.info(f"Input shape: {X.shape}, Output shape: {y.shape}")
    logger.info(f"Input dtype: {X.dtype}, Output dtype: {y.dtype}")

    params = {
        'epochs': 200,
        'batch_size': 16,
        'learning_rate': 0.0001,
        'hidden_size': 64,
        'num_layers': 2,
        'early_stopping_patience': 20
    }

    model = SensorFusionLSTM(
        input_size=X.shape[-1],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        output_size=y.shape[-1]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True
    )

    dataset = NavigationDataset(X, y, device)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params['batch_size'])
    test_loader = DataLoader(test_set, batch_size=params['batch_size'])

    best_val_loss = float('inf')
    early_stopping_counter = 0
    metrics_tracker = MetricsTracker()
    visualizer = PerformanceVisualizer()

    logger.info("Starting training loop...")
    for epoch in range(params['epochs']):
        model.train()
        epoch_metrics = defaultdict(float)
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            optimizer.zero_grad(set_to_none=True)

            try:
                output = model(data)
                target_expanded = target.unsqueeze(1).expand(-1, output.size(1), -1)
                loss = criterion(output, target_expanded)

                batch_metrics = metrics_tracker.update(output[:, -1, :], target)
                for k, v in batch_metrics.items():
                    epoch_metrics[k] += v

                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            except RuntimeError as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                logger.error(f"Data shape: {data.shape}, Target shape: {target.shape}")
                raise e

        avg_train_loss = epoch_loss / batch_count
        metrics_tracker.history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_metrics = defaultdict(float)
        val_batch_count = 0

        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                target_expanded = target.unsqueeze(1).expand(-1, output.size(1), -1)
                loss = criterion(output, target_expanded)

                batch_metrics = metrics_tracker.update(output[:, -1, :], target)
                for k, v in batch_metrics.items():
                    val_metrics[k] += v

                val_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        metrics_tracker.history['val_loss'].append(avg_val_loss)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        for metric_name in ['mae', 'rmse', 'r2', 'explained_variance']:
            train_metric = epoch_metrics[metric_name] / batch_count
            val_metric = val_metrics[metric_name] / val_batch_count
            metrics_tracker.history[metric_name].append(train_metric)
            logger.info(f'{metric_name.upper()}: Train = {train_metric:.4f}, Val = {val_metric:.4f}')

        logger.info(f'Epoch {epoch}:')
        logger.info(f'Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
        logger.info(f'Learning Rate = {current_lr:.6f}')

        if epoch % 10 == 0:
            visualizer.plot_training_history(metrics_tracker.history, title_suffix=f'_epoch_{epoch}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'models/best_lstm_model.pth')
            logger.info(f"New best model saved with validation loss: {avg_val_loss:.6f}")
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= params['early_stopping_patience']:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Final training loss: {metrics_tracker.history['train_loss'][-1]:.6f}")

    logger.info("Evaluating on test set...")
    test_metrics, component_metrics, (y_true, y_pred) = evaluate_test_set(
        model, test_loader, device)

    logger.info("\nTest Set Metrics:")
    for metric_name, value in test_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")

    logger.info("\nComponent-wise Metrics:")
    for comp in component_metrics:
        logger.info(f"Component {comp['component']}: RMSE={comp['rmse']:.4f}, R²={comp['r2']:.4f}")

    visualizer.plot_training_history(metrics_tracker.history, title_suffix='_final')
    visualizer.plot_prediction_analysis(y_true, y_pred, title_suffix='_final')

    return model, metrics_tracker.history, test_metrics