import torch
import torch.nn as nn
import torch.nn.functional as F
from models.posenet.PoseNet import PoseNet


class RPoseNet(PoseNet):
    """
    A class to represent a classic pose regressor (PoseNet) with an efficient-net backbone
    PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization,
    Kendall et al., 2015
    """
    def __init__(self, config, backbone_path):
        """
        Constructor
        :param backbone_path: backbone path to a resnet backbone
        """
        super(RPoseNet, self).__init__(config, backbone_path)


        # Regressor layers
        self.x_latent_fc = nn.Linear(self.backbone_dim*2, self.latent_dim)
        self.q_latent_fc = nn.Linear(self.backbone_dim*2, self.latent_dim)
        self.x_reg = nn.Linear(self.latent_dim, 3)
        self.q_reg = nn.Linear( self.latent_dim, 4)

        self.dropout = nn.Dropout(p=0.1)
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward_body(self, query, ref):

        if self.backbone_type == "efficientnet":
            query_vec = self.backbone.extract_features(query)
            ref_vec = self.backbone.extract_features(ref)

        else:
            query_vec = self.backbone(query)
            ref_vec = self.backbone(ref)

        query_vec = self.avg_pooling_2d(query_vec)
        query_vec = query_vec.flatten(start_dim=1)
        ref_vec = self.avg_pooling_2d(ref_vec)
        ref_vec = ref_vec.flatten(start_dim=1)

        latent = torch.cat((query_vec, ref_vec), dim=1)

        latent_q = F.relu(self.x_latent_fc(latent))
        latent_x = F.relu(self.q_latent_fc(latent))
        return latent_x, latent_q

    def forward_heads(self, latent_x, latent_q):
        rel_x = self.x_reg(self.dropout(latent_x))
        rel_q = self.q_reg(self.dropout(latent_q))
        return rel_x, rel_q

    def forward(self, query, ref, ref_pose):
        latent_x, latent_q = self.forward_body(query, ref)
        rel_x, rel_q = self.forward_heads(latent_x, latent_q)
        x = ref_pose[:, :3] + rel_x
        q = qmult(rel_q, qinv(ref_pose[:, 3:]))
        return {"pose": torch.cat((x,q), dim=1)}


def batch_dot(v1, v2):
    """
    Dot product along the dim=1
    :param v1: (torch.tensor) Nxd tensor
    :param v2: (torch.tensor) Nxd tensor
    :return: N x 1
    """
    out = torch.mul(v1, v2)
    out = torch.sum(out, dim=1, keepdim=True)
    return out

def qmult(quat_1, quat_2):
    """
    Perform quaternions multiplication
    :param quat_1: (torch.tensor) Nx4 tensor
    :param quat_2: (torch.tensor) Nx4 tensor
    :return: quaternion product
    """
    # Extracting real and virtual parts of the quaternions
    q1s, q1v = quat_1[:, :1], quat_1[:, 1:]
    q2s, q2v = quat_2[:, :1], quat_2[:, 1:]

    qs = q1s*q2s - batch_dot(q1v, q2v)
    qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) + torch.cross(q1v, q2v, dim=1)
    q = torch.cat((qs, qv), dim=1)

    return q


def qinv(q):
    """
    Inverts a unit quaternion
    :param q: (torch.tensor) Nx4 tensor (unit quaternion)
    :return: Nx4 tensor (inverse quaternion)
    """
    q_inv = torch.cat((q[:, :1], -q[:, 1:]), dim=1)
    return q_inv
