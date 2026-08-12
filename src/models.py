"""CNN architectures for EEG seizure detection.

Two approaches on the same 4-second CHB-MIT window:

- ``CNN1D`` convolves along time across all 23 electrode channels.
- ``CNN2D`` treats an STFT spectrogram of the window as a single-channel image.
"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1D CNN over raw multi-channel EEG.

    Input:  (batch, 23, 1024) — 23 channels, 4 s at 256 Hz
    Output: (batch, 2) logits
    """

    def __init__(self, in_channels=23, num_classes=2):
        super().__init__()

        self.net = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 4
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # Collapse the time axis so the head is independent of window length.
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x)


class CNN2D(nn.Module):
    """2D CNN over spectrogram images derived from an EEG window.

    Input:  (batch, 1, freq_bins, time_bins)
    Output: (batch, 2) logits
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.5)

        # Global average pooling keeps the head independent of spectrogram size.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        return self.fc(x)
