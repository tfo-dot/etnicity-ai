import torch.nn as nn

from etnicity_ai import ann_layers

class FairFaceResNet(nn.Module):
    def __init__(self, resnet) -> None:
        super(FairFaceResNet, self).__init__()

        self.resnet = resnet
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        for param in self.resnet.parameters():
            param.requires_grad = True
        
        self.fc = nn.Sequential(nn.Flatten(), 
                                 ann_layers.ANN_Blocks(in_features=2048),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(16, 7))

    def forward(self, x):
        out = self.resnet(x)
        out = out.view(out.size(0), -1)

        race = self.fc(out)

        return {'race_pred' : race}