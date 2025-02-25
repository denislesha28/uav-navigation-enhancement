import logging

import torch
from ncps.torch import LTC
from ncps.wirings import AutoNCP
from torch import optim, Generator, no_grad, save
from torch.nn import MSELoss
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch_generator = Generator().manual_seed(42)


class LTCDataset(Dataset):
    def __init__(self, data, target, device, transform=None):
        self.data = data.to(device)
        self.labels = torch.from_numpy(target).to(device)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


def train_ltc(X, y, device):
    training_params = {
        'epochs': 200,
        'batch_size': 8,
        'learning_rate': 0.0003,
        'early_stopping_patience': 15,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15
    }

    logger.info("Initializing model and parameters...")
    ncp_wiring = AutoNCP(20, 15)
    ltc = LTC(128, ncp_wiring, batch_first=True)
    ltc.to(device=device)
    ltc = ltc.to(torch.float32)  # Ensure consistent dtype

    optimizer = optim.Adam(ltc.parameters(), lr=training_params['learning_rate'])
    criterion = MSELoss()

    logger.info("Preparing datasets and dataloaders...")
    dataset = LTCDataset(X, y, device)
    train, test, val = random_split(dataset, [0.7, 0.15, 0.15], generator=torch_generator)

    train_loader = DataLoader(train, batch_size=training_params['batch_size'], pin_memory=False, shuffle=False)
    test_loader = DataLoader(test, batch_size=training_params['batch_size'], pin_memory=False, shuffle=False)
    val_loader = DataLoader(val, batch_size=training_params['batch_size'], pin_memory=False, shuffle=False)

    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_losses = []
    val_losses = []

    logger.info("Starting training...")
    for epoch in tqdm(range(training_params['epochs']), desc="Training Progress"):
        train_loss = train_epoch(ltc, train_loader, optimizer, criterion, device)
        val_loss = validate(ltc, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(
            f"Epoch [{epoch + 1}/{training_params['epochs']}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            save(ltc.state_dict(), 'models/best_model.pth')
            logger.info(f"New best model saved with validation loss: {val_loss:.6f}")
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= training_params['early_stopping_patience']:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Final training loss: {train_losses[-1]:.6f}")

    return train_losses, val_losses


def train_epoch(ltc, train_loader, optimizer, criterion, device):
    ltc.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, leave=False, desc="Training batches")
    for x, y in pbar:
        optimizer.zero_grad()

        # Initialize hidden state based on batch input size
        hidden = torch.zeros(x.size(0), ltc._wiring.units, device=device)

        x = x.to(torch.float32)
        y = y.to(torch.float32)

        output, _ = ltc(x, hidden)
        loss = criterion(output, y.unsqueeze(1).expand(-1, output.size(1), -1))
        loss.backward(retain_graph=True)

        # Add gradient clipping
        # torch.nn.utils.clip_grad_norm_(ltc.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'batch_loss': f'{loss.item():.6f}'})

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def validate(ltc, val_loader, criterion):
    ltc.eval()
    total_loss = 0.0

    with no_grad():
        for x, y in tqdm(val_loader, leave=False, desc="Validation"):
            # Initialize hidden state based on current batch
            hidden = torch.zeros(x.size(0), ltc._wiring.units, device=x.device)

            x = x.to(torch.float32)
            y = y.to(torch.float32)

            output, _ = ltc(x, hidden)
            loss = criterion(output, y.unsqueeze(1).expand(-1, output.size(1), -1))
            total_loss += loss.item()

    return total_loss / len(val_loader)