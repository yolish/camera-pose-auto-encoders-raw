import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transposenet.transformer_encoder import Transformer

default_config = {
        "hidden_dim": 256,
        "nhead": 8,
        "num_encoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "gelu",
        "normalize_before": True,
        "return_intermediate_dec": False
    }
class PoseTransformer(nn.Module):
    def __init__(self, encoder_dim):
        super(PoseTransformer, self).__init__()
        self.transformer = Transformer(default_config)

    def forward(self, query_latent, ref_latent):
        pass


class ResidualPoseRegressor(nn.Module):

    def __init__(self, encoder_dim):
        """
        """
        super(ResidualPoseRegressor, self).__init__()

        # Efficient net
        encoder_dim = encoder_dim * 2
        self.x_regressor = nn.Sequential(nn.Linear(encoder_dim, encoder_dim*2), nn.ReLU(),
                                       nn.Linear(encoder_dim*2,encoder_dim*2),
                                       nn.ReLU(),
                                       nn.Linear(encoder_dim*2,encoder_dim),
                                       nn.ReLU(),
                                       nn.Linear(encoder_dim, 3)
                                       )
        self.q_regressor = nn.Sequential(nn.Linear(encoder_dim, encoder_dim * 2), nn.ReLU(),
                                         nn.Linear(encoder_dim * 2, encoder_dim * 2),
                                         nn.ReLU(),
                                         nn.Linear(encoder_dim * 2, encoder_dim),
                                         nn.ReLU(),
                                         nn.Linear(encoder_dim, 4)
                                         )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query_latent_x, ref_latent_x, query_latent_q, ref_latent_q):
        latent_query = torch.cat((query_latent_x, query_latent_q), dim=1)
        latent_ref = torch.cat((ref_latent_x, ref_latent_q), dim=1)
        latent = torch.cat((latent_query, latent_ref), dim=1)
        delta_x = self.x_regressor(latent)
        delta_q = self.q_regressor(latent)

        return delta_x, delta_q



