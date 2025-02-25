import numpy as np
import torch

from torch import nn

torch.manual_seed(42)
np.random.seed(42)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y):
        if isinstance(yhat, tuple):
            yhat = yhat[0]

        print(f"Before reshape - yhat shape: {yhat.shape}, y shape: {y.shape}")

        yhat = yhat[:, -1, :]
        y = y.to(yhat.device).to(dtype=yhat.dtype)

        print(f"After reshape - yhat shape: {yhat.shape}, y shape: {y.shape}")

        loss = torch.mean((yhat - y) ** 2)
        loss = torch.sqrt(loss + self.eps)

        return loss


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=9):
        super().__init__()

        self.bn_input = nn.BatchNorm1d(input_channels)

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.01)

        self.conv2 = nn.Conv1d(64, 126, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(126)
        self.act2 = nn.LeakyReLU(0.01)

        self.conv3 = nn.Conv1d(126, 128, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.LeakyReLU(0.01)

        self.skip_proj = nn.Conv1d(126, 128, kernel_size=1)

        self.bn_out = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.1)

        self._initialize_weights()
        self.double()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Kaiming initialization works well with LeakyReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.bn_input(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout(x)

        identity = self.skip_proj(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = x + identity

        x = self.bn_out(x)

        # Reshape for LNN input [batch, time, features]
        return x.permute(0, 2, 1)

    def debug_forward(self, x):
        """Debug forward pass with detailed statistics"""

        def print_stats(name, tensor):
            stats = {
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item(),
                'has_nan': torch.isnan(tensor).any().item()
            }
            print(f"\n{name}:")
            for k, v in stats.items():
                print(f"{k}: {v:.6f}")
            return stats

        print("\nStarting forward pass debug...")

        stats = {}

        stats['input'] = print_stats("Input", x)
        x = self.bn_input(x)
        stats['after_input_bn'] = print_stats("After input BN", x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout(x)
        stats['after_block1'] = print_stats("After Block 1", x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout(x)
        stats['after_block2'] = print_stats("After Block 2", x)

        identity = self.skip_proj(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = x + identity
        stats['after_block3'] = print_stats("After Block 3", x)

        x = self.bn_out(x)
        stats['final_output'] = print_stats("Final Output", x)

        return x.permute(0, 2, 1), stats

    # def cnn_extract_features(self, X, y):
    #     dataset = VectorDataset(X, y)
    #
    #     data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    #
    #     optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
    #     criterion = self.RMSELoss()
    #
    #     for epoch in range(EPOCHS):
    #         self.train(True)
    #         self.train_epoch(data_loader, criterion, optimizer, epoch)
    #         self.eval()
    #
    #     print('Finished Training')
    #
    # def train_epoch(self, data_loader, criterion, optimizer, epoch):
    #     running_loss = 0.0
    #
    #     for i, data in enumerate(data_loader, 0):
    #         inputs, labels = data
    #
    #         output = self(inputs)
    #         loss = criterion(output, labels)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:
    #             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
    #             running_loss = 0.0


def validate_cnn_features(input_data, output_tuple):
    """
    Validate CNN feature extraction with detailed analysis

    Args:
        input_data: Input tensor
        output_tuple: Tuple of (features, stats) from CNN
    """
    output_features, stats = output_tuple if isinstance(output_tuple, tuple) else (output_tuple, None)

    print("\nCNN Feature Extraction Validation:")
    print("-" * 50)

    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output_features.shape}")

    print("\nQuality Checks:")
    input_nans = torch.isnan(input_data).any()
    output_nans = torch.isnan(output_features).any()
    print(f"Input contains NaN: {input_nans}")
    print(f"Output contains NaN: {output_nans}")

    if not output_nans:
        stats = {
            'input_mean': torch.mean(input_data).item(),
            'input_std': torch.std(input_data).item(),
            'input_min': torch.min(input_data).item(),
            'input_max': torch.max(input_data).item(),
            'output_mean': torch.mean(output_features).item(),
            'output_std': torch.std(output_features).item(),
            'output_min': torch.min(output_features).item(),
            'output_max': torch.max(output_features).item()
        }

        print("\nInput Statistics:")
        print(f"Mean: {stats['input_mean']:.6f}")
        print(f"Std:  {stats['input_std']:.6f}")
        print(f"Min:  {stats['input_min']:.6f}")
        print(f"Max:  {stats['input_max']:.6f}")

        print("\nOutput Statistics:")
        print(f"Mean: {stats['output_mean']:.6f}")
        print(f"Std:  {stats['output_std']:.6f}")
        print(f"Min:  {stats['output_min']:.6f}")
        print(f"Max:  {stats['output_max']:.6f}")

        if len(output_features.shape) == 3:
            feature_means = torch.mean(output_features, dim=(0, 1))
            feature_stds = torch.std(output_features, dim=(0, 1))

            print("\nPer-Feature Statistics:")
            print(f"Feature means range: [{torch.min(feature_means):.4f}, {torch.max(feature_means):.4f}]")
            print(f"Feature stds range:  [{torch.min(feature_stds):.4f}, {torch.max(feature_stds):.4f}]")

            dead_features = torch.where(feature_stds < 1e-6)[0]
            if len(dead_features) > 0:
                print(f"\nWarning: Found {len(dead_features)} potentially dead features")

    return not (input_nans or output_nans)


def visualize_feature_maps(features, sample_idx=0):
    """
    Visualize feature maps for a single sample
    """
    import matplotlib.pyplot as plt
    features = (features[0])
    channels_to_plot = np.random.choice((features[0]).shape[1], 4, replace=False)

    plt.figure(figsize=(15, 3))
    for i, channel in enumerate(channels_to_plot):
        plt.subplot(1, 4, i + 1)
        plt.plot(features[sample_idx, channel, :].detach().numpy())
        plt.title(f'Channel {channel}')
    plt.tight_layout()
    plt.show()
