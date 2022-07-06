import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_regressors import get_model
from models.pose_encoder import PoseEncoder, MultiSCenePoseEncoder
import torch.nn as nn


class PoseOptim(torch.nn.Module):
    def __init__(self, num_neighbors, dim):
        """
        :param config: (dict) configuration to determine behavior
        """
        super(PoseOptim, self).__init__()
        dim = dim*2
        self.regressor = torch.nn.Sequential(nn.Linear(num_neighbors*dim, num_neighbors*64),
                                             nn.ReLU(),
                                             nn.Linear(num_neighbors * 64, num_neighbors * 32),
                                             nn.ReLU(),
                                             nn.Linear(num_neighbors * 32, num_neighbors * 16),
                                             nn.ReLU(),
                                             nn.Linear(num_neighbors * 16, num_neighbors)
                                             )


    def forward(self, ref_latent):
        weights = self.regressor(ref_latent.view(1, -1))
        weights = torch.nn.functional.softmax(weights, dim=1).squeeze(0).unsqueeze(1)
        return torch.sum(
            weights * ref_latent,
            dim=0).unsqueeze(0), weights

    def reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

def get_ref_pose(query_poses, db_poses, start_index, k):
    ref_poses = np.zeros((query_poses.shape[0], k, 7))
    for i, p in enumerate(query_poses):
        dist_x = np.linalg.norm(p[:3] - db_poses[:, :3], axis=1)
        dist_x = dist_x / np.max(dist_x)
        dist_q = np.linalg.norm(p[3:] - db_poses[:, 3:], axis=1)
        dist_q = dist_q / np.max(dist_q)
        sorted = np.argsort(dist_x + dist_q)
        ref_poses[i, 0:k, :] = db_poses[sorted[start_index:(k+start_index)]]
    return ref_poses


def weighted_avg_quaterions(Q, w):
    # Implementation copied from https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py
    # Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
    # The quaternions are arranged as (w,x,y,z), with w being the scalar
    # The result will be the average quaternion of the input. Note that the signs
    # of the output quaternion can be reversed, since q and -q describe the same orientation
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros((4,4))
    weightSum = 0

    for i in range(0,M):
        q = Q[i,:]
        A = w[i] * np.outer(q,q) + A
        weightSum += w[i]

    # scale
    A = (1.0/weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0])


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name",
                            help="name of model to create (e.g. posenet, transposenet")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("config_file", help="path to configuration file", default="7scenes-config.json")
    arg_parser.add_argument("checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("encoder_checkpoint_path", help="path to a trained pose encoder")
    arg_parser.add_argument("ref_poses_file", help="path to file with train poses")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start test time optimization for APR with prior")
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Load the apr model
    apr = get_model(args.model_name, args.backbone_path, config).to(device)
    apr.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
    logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))
    apr.eval()

    is_single_scene = config.get("single_scene")
    if is_single_scene:
        pose_encoder = PoseEncoder(config.get("hidden_dim")).to(device)
    else:
        pose_encoder = MultiSCenePoseEncoder(config.get("hidden_dim")).to(device)
    pose_encoder.load_state_dict(torch.load(args.encoder_checkpoint_path, map_location=device_id))
    logging.info("Initializing encoder from checkpoint: {}".format(args.encoder_checkpoint_path))
    pose_encoder.eval()

    # Test time optimization
    num_neighbors = int(config.get("num_neighbors"))
    refine_orientation = config.get("refine_orientation")

    pose_optim = PoseOptim(num_neighbors, config.get("hidden_dim")).to(device)
    # Set the losses
    loss = torch.nn.MSELoss().to(device)

    # Set the optimizer and scheduler
    lr = 1e-2  # config.get('lr')
    optim = torch.optim.AdamW(pose_optim.parameters(), lr=lr)
    test_optim_iterations = config.get("test_optim_iterations")

    # Set the dataset and data loader
    transform = utils.test_transforms.get('baseline')
    ref_poses = CameraPoseDataset(args.dataset_path, args.ref_poses_file, None).poses
    dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)
    loader_params = {'batch_size': 1,
                     'shuffle': False,
                     'num_workers': config.get('n_workers')}
    dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

    stats = np.zeros((len(dataloader.dataset), 3))


    for i, minibatch in enumerate(dataloader, 0):
        for k, v in minibatch.items():
            minibatch[k] = v.to(device)
        gt_scene = minibatch.get('scene')
        minibatch['scene'] = None # avoid using ground-truth scene during prediction

        gt_pose = minibatch.get('pose').to(dtype=torch.float32)

        # Forward pass to predict the pose
        with torch.no_grad():
            res = apr(minibatch)
            est_pose = res.get('pose')
            tic = time.time()
            latent_x = res.get("latent_x")
            latent_q = res.get("latent_q")
            if not is_single_scene:
                scene_dist = res.get('scene_log_distr')
                scene = torch.argmax(scene_dist, dim=1).to(dtype=torch.float32).unsqueeze(1).repeat(1,
                                                                                                    num_neighbors).view(
                    -1, 1)

            # get closest poses
            ref_pose = get_ref_pose(est_pose.cpu().numpy(), dataset.poses, 0, num_neighbors)

            ref_pose = torch.Tensor(ref_pose).to(device).reshape(1 * num_neighbors, 7)
            if is_single_scene:
                ref_latent_x, ref_latent_q = pose_encoder(ref_pose)
            else:
                ref_latent_x, ref_latent_q = pose_encoder(ref_pose, scene)



        query_latent = torch.cat((latent_x, latent_q), dim=1)
        ref_latent = torch.cat((ref_latent_x, ref_latent_q), dim=1)
        pose_optim.train()
        for _ in range(test_optim_iterations):

            optim.zero_grad()
            est_latent, weights = pose_optim(ref_latent)
            criterion = loss(query_latent, est_latent)
            # Back prop
            criterion.backward()
            optim.step()

        # Evaluate error
        pose_optim.eval()
        with torch.no_grad():
            refined_est_pose = torch.sum(weights * ref_pose, dim=0).unsqueeze(0)
            if refine_orientation is None: # use apr estimation for orientation - default for our paper
                refined_est_pose[:, 3:] = est_pose[:, 3:]
            elif refine_orientation == "affine":
                refined_est_pose[:, 3:] = refined_est_pose[:, 3:] / torch.norm(refined_est_pose[:, 3:])
            elif refine_orientation == "eigen":
                refined_est_pose[:, 3:] = torch.from_numpy(weighted_avg_quaterions(ref_pose[:, 3:].cpu().numpy(), weights.cpu().numpy())).to(device)
                refined_est_pose[:, 3:] = refined_est_pose[:, 3:] / torch.norm(refined_est_pose[:, 3:])
                refined_est_pose[:, 3:] = ref_pose[torch.argmax(weights, dim=0)[0]][3:]
            elif refine_orientation == "closest":
                refined_est_pose[:, 3:] = ref_pose[0, 3:]
            elif refine_orientation == "max":
                refined_est_pose[:, 3:] = ref_pose[torch.argmax(weights, dim=0)[0], 3:]
            else:
                raise NotImplementedError("Specified orientation refinement is not supported")

        toc = time.time()
        posit_err, orient_err = utils.pose_err(refined_est_pose, gt_pose)
        pose_optim.reset_params()

        # Collect statistics
        stats[i, 0] = posit_err.item()
        stats[i, 1] = orient_err.item()
        stats[i, 2] = (toc - tic)*1000

        logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
            stats[i, 0],  stats[i, 1],  stats[i, 2]))

    # Record overall statistics
    logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
    logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]), np.nanmedian(stats[:, 1])))
    logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))





