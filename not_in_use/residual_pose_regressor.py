import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transposenet.MSTransPoseNet import PoseRegressor


class PosePriorTransformer(nn.Module):
    def __init__(self, config):
        super(PosePriorTransformer, self).__init__()
        transformer_dim = config.get("hidden_dim")
        self.num_neighbors = config.get("num_neighbors")
        ppt_config = config.get("ppt")
        x_encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim,
                                               nhead=config.get("nheads"),
                                               dim_feedforward=config.get("dim_feedforward"),
                                               dropout=config.get("dropout"),
                                               activation="gelu")

        self.x_transformer_encoder = nn.TransformerEncoder(x_encoder_layer,
                                                      num_layers=ppt_config.get("num_t_encoder_layers"),
                                                      norm=nn.LayerNorm(transformer_dim))

        q_encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim,
                                                     nhead=config.get("nheads"),
                                                     dim_feedforward=config.get("dim_feedforward"),
                                                     dropout=config.get("dropout"),
                                                     activation="gelu")

        self.q_transformer_encoder = nn.TransformerEncoder(q_encoder_layer,
                                                           num_layers=ppt_config.get("num_rot_encoder_layers"),
                                                           norm=nn.LayerNorm(transformer_dim))
        self.x_token = nn.Parameter(torch.zeros((1, transformer_dim)), requires_grad=True)
        self.q_token = nn.Parameter(torch.zeros((1, transformer_dim)), requires_grad=True)
        self._reset_parameters()

        self.position_embed_x = nn.Parameter(torch.randn(self.num_neighbors+2, 1, transformer_dim))
        self.position_embed_q = nn.Parameter(torch.randn(self.num_neighbors+2, 1, transformer_dim))
        self.x_reg = PoseRegressor(transformer_dim, 3)
        self.q_reg = PoseRegressor(transformer_dim, 4)
        self.transformer_dim = transformer_dim


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def regress_proprerty(self, query_latent, ref_latent, target_type, dim):
        batch_size = query_latent.shape[0]
        if target_type == "x":
            token_fn = self.x_token
            transformer_fn = self.x_transformer_encoder
            reg_fn = self.x_reg
            position_embed_fn = self.position_embed_x
        else:
            token_fn = self.q_token
            transformer_fn = self.q_transformer_encoder
            reg_fn = self.q_reg
            position_embed_fn = self.position_embed_q

        out = torch.zeros(batch_size, dim).to(query_latent.device).to(query_latent.dtype)

        for i in range(batch_size):
            # N: batch size, C: channel, S: Sequence lenth
            # S x C
            src = torch.cat((query_latent[i, :].unsqueeze(0), ref_latent[i*self.num_neighbors:(i+1)*self.num_neighbors, :]), dim=0)
            # S x N x C
            src = src.unsqueeze(1)
            # Append the token
            token = token_fn.unsqueeze(1)
            src = torch.cat([src, token])
            # Add the position embedding
            src += position_embed_fn
            target = transformer_fn(src)[0]
            out[i, :] = reg_fn(target)
        return out

    def forward(self, query_latent_x, ref_latent_x, query_latent_q, ref_latent_q):
        x = self.regress_proprerty(query_latent_x, ref_latent_x, "x", 3)
        q = self.regress_proprerty(query_latent_q, ref_latent_q, "q", 4)
        return {"pose":torch.cat((x,q), dim=1)}


class TestTimePosePriorTransformer(PosePriorTransformer):
    def __init__(self, config):
        super(TestTimePosePriorTransformer, self).__init__(config)
        self.x_reg = PoseRegressor(self.transformer_dim, self.num_neighbors)
        self.q_reg = PoseRegressor(self.transformer_dim, self.num_neighbors)

    def forward(self, query_latent_x, ref_latent_x, query_latent_q, ref_latent_q, ref_poses):
        batch_size = query_latent_x.shape[0]
        x_weights = torch.nn.functional.softmax(self.regress_proprerty(query_latent_x, ref_latent_x, "x", self.num_neighbors), dim=1)
        q_weights = torch.nn.functional.softmax(self.regress_proprerty(query_latent_q, ref_latent_q, "q", self.num_neighbors), dim=1)
        x = torch.zeros((batch_size, 3)).to(query_latent_x.device).to(query_latent_x.dtype)
        est_latent_x = torch.zeros((batch_size, query_latent_x.shape[1])).to(query_latent_x.device).to(query_latent_x.dtype)
        q = torch.zeros((batch_size, 4)).to(query_latent_x.device).to(query_latent_x.dtype)
        est_latent_q = torch.zeros_like(est_latent_x)
        for i in range(batch_size):
            est_latent_x[i] = torch.sum(x_weights[i,:].unsqueeze(1) * ref_latent_x[i*self.num_neighbors:(i+1)*self.num_neighbors, :], dim=0)
            est_latent_q[i] = torch.sum(q_weights[i,:].unsqueeze(1) * ref_latent_q[i*self.num_neighbors:(i+1)*self.num_neighbors, :], dim=0)
            x[i] = torch.sum(x_weights[i, :].unsqueeze(1) * ref_poses[i*self.num_neighbors:(i+1)*self.num_neighbors, :3], dim=0)
            q[i] = torch.sum(q_weights[i, :].unsqueeze(1) * ref_poses[i*self.num_neighbors:(i+1)*self.num_neighbors, 3:], dim=0)
        return {"pose":torch.cat((x,q), dim=1), "est_latent_x":est_latent_x, "est_latent_q":est_latent_q}




