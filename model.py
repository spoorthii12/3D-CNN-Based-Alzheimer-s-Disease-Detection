import torch
import torch.nn as nn

class CNN3DTabular(nn.Module):
    def __init__(self, num_tabular_features):
        super(CNN3DTabular, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # → (8, 64, 64, 64)
            nn.Dropout3d(0.3),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # → (16, 32, 32, 32)
            nn.Dropout3d(0.3),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # → (32, 16, 16, 16)
            nn.Dropout3d(0.3),
        )
        self.flatten = nn.Flatten()
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16 * 16 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(128, 1)  # Binary logit
        )

    def forward(self, x, tabular):
        x = self.cnn(x)
        x = self.flatten(x)
        tabular = self.tabular_branch(tabular)
        combined = torch.cat((x, tabular), dim=1)
        return self.classifier(combined)