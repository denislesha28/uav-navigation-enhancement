import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
# Import LTC components from ncps
from ncps.torch import LTC
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedNavigationLTC(nn.Module):
    def __init__(self,
                 input_size=128,
                 hidden_size=128,
                 output_size=15,
                 dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LTC layer for continuous-time dynamics
        self.ltc = LTC(
            input_size=input_size,
            units=hidden_size,
            return_sequences=True,
            batch_first=True,
            mixed_memory=False,
            ode_unfolds=4
        )

        # No attention mechanism - LTCs model temporal dynamics differently
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Process with LTC - unpack the tuple
        ltc_out, _ = self.ltc(x)

        # Now you can index ltc_out as a tensor
        last_hidden = ltc_out[:, -1, :]

        # Apply normalization and dropout
        out = self.layer_norm(last_hidden)
        out = self.dropout(out)

        # Final prediction
        predictions = self.fc(out)

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
        r2 = r2_score(targets_np, outputs_np)
        exp_var = explained_variance_score(targets_np, outputs_np)

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'explained_variance': exp_var
        }


def train_ltc_k_folds(X, y, n_folds, device):
    logger.info(f"Input shape: {X.shape}, Output shape: {y.shape}")
    logger.info(f"Input dtype: {X.dtype}, Output dtype: {y.dtype}")

    params = {
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'hidden_size': 128,
        'weight_decay': 1e-5,  # Add weight decay for regularization
        'early_stopping_patience': 15
    }

    model = EnhancedNavigationLTC(
        input_size=X.shape[-1],
        hidden_size=params['hidden_size'],
        output_size=y.shape[-1]
    ).to(device)

    # 2. Adjust optimizer parameters if needed
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    dataset = NavigationDataset(X, y, device)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        save_test_fold_indexing(fold, test_idx)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=test_subsampler)

        metrics_tracker = MetricsTracker()

        best_val_loss = float('inf')
        early_stopping_counter = 0

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
                    criterion = nn.MSELoss()

                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=60, eta_min=1e-5
                    )
                    loss = criterion(output, target)

                    batch_metrics = metrics_tracker.update(output, target)
                    for k, v in batch_metrics.items():
                        epoch_metrics[k] += v

                    loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
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
                    loss = criterion(output, target)

                    batch_metrics = metrics_tracker.update(output, target)
                    for k, v in batch_metrics.items():
                        val_metrics[k] += v

                    val_loss += loss.item()
                    val_batch_count += 1

            avg_val_loss = val_loss / val_batch_count
            metrics_tracker.history['val_loss'].append(avg_val_loss)

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            for metric_name in ['mae', 'rmse', 'r2', 'explained_variance']:
                train_metric = epoch_metrics[metric_name] / batch_count
                val_metric = val_metrics[metric_name] / val_batch_count
                metrics_tracker.history[metric_name].append(train_metric)
                logger.info(f'{metric_name.upper()}: Train = {train_metric:.4f}, Val = {val_metric:.4f}')

            logger.info(f'Epoch {epoch}:')
            logger.info(f'Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
            logger.info(f'Learning Rate = {current_lr:.6f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                save_model(model, "best_ltc_" + str(fold))
                logger.info(f"New best model saved with validation loss: {avg_val_loss:.6f}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= params['early_stopping_patience']:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

    logger.info("Training completed!")


def save_model(model, model_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "..", "models")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, model_name + '.pth'))


def save_test_fold_indexing(fold, test_idx):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "..", "models")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'fold_{fold + 1}_test_indices.npy'), test_idx)


