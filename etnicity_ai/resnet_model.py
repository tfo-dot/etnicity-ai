import torch.nn as nn
from etnicity_ai import ann_layers

class FairFaceResNet(nn.Module):
    def __init__(self, resnet) -> None:
        super(FairFaceResNet, self).__init__()

        # Extract ResNet backbone (excluding final FC layer)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))

        # Freeze early layers (first 6 blocks) for better transfer learning
        for param in list(self.resnet.children())[:6]:
            param.requires_grad = False  

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(2048, 512),  # First dense layer
            nn.BatchNorm1d(512),  # Normalize activations
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 128),  # Second dense layer
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 7)  # Output layer for 7 classes
        )

    def forward(self, x):
        out = self.resnet(x)
        out = out.view(out.size(0), -1)
        race = self.fc(out)
        return {'race_pred': race}
