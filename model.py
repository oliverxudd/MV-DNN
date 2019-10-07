import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):

    def __init__(self, input_dim):
        """
        :param neg_sample_num: number of negative sample in one set of input data.
        :param input_dim: dimension of input data
        """
        self.featuer_extractor = nn.Sequential(
            nn.Linear(input_dim, 30000),
            nn.Linear(30000, 300), nn.Tanh(),
            nn.Linear(300, 300), nn.Tanh(),
            nn.Linear(300, 128), nn.Tanh(),
        )


    def forward(self, input):
       return self.featuer_extractor(input)

