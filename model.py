# model.py
import torch
import torch.nn as nn

class DigitGeneratorNet(nn.Module):
    def __init__(self, latent_dim=64, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        x = torch.cat((z, label_input), dim=1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)