import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(FeatureExtractor, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 30000),
            nn.Linear(30000, 300), nn.Tanh(),
            nn.Linear(300, 300), nn.Tanh(),
            nn.Linear(300, 128), nn.Tanh()
        )

    def forward(self, x):
       return self.feature_extractor(x)

