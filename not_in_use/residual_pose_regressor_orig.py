import torch
import torch.nn as nn
import torch.nn.functional as F


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
        latent_x = torch.cat((query_latent_x, ref_latent_x), dim=1)
        delta_x = self.x_regressor(latent_x)
        latent_q = torch.cat((query_latent_q, ref_latent_q), dim=1)
        delta_q = self.q_regressor(latent_q)
        return delta_x, delta_q

class ResidualPoseRegressor(nn.Module):

    def __init__(self, config):
        """
        """
        super(ResidualPoseRegressor, self).__init__()

        encoder_dim = config.get("hidden_dim")
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



