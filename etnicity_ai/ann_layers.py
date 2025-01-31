import torch.nn as nn

class ANN_cycle(nn.Module):
    def __init__(self, in_features, out_features, dropout_val=0.3) -> None:
        super(ANN_cycle, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),  # Normalize activations
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_val)
        )
        
    def forward(self, x):
        return self.block(x)


class ANN_Blocks(nn.Module):
    def __init__(self, in_features, num_classes=7) -> None:
        super(ANN_Blocks, self).__init__()
        self.block = nn.Sequential(
            ANN_cycle(in_features, 512, 0.3),
            ANN_cycle(512, 256, 0.3),
            ANN_cycle(256, 128, 0.2),  # Reduce dropout
            ANN_cycle(128, 64, 0.2),
            ANN_cycle(64, 32, 0.1),
            ANN_cycle(32, 16, 0.1),
            nn.Linear(16, num_classes)  # Add output layer
        )

    def forward(self, x):
        return self.block(x)
