import torch.nn as nn


class Video_Feat_Encoder_NoPosembed(nn.Module):
    """ Transform the video feat encoder"""

    def __init__(self, origin_dim, embed_dim, seq_len=215):
        super().__init__() 
        self.embedder = nn.Sequential(nn.Linear(origin_dim, embed_dim))

    def forward(self, x):
        x = self.embedder(x)       

        return x

