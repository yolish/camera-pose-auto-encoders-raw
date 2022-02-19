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
        self.regressor_x = torch.nn.Sequential(nn.Linear(num_neighbors*dim, num_neighbors*64),
                                             nn.GELU(),
                                             nn.Linear(num_neighbors * 64, num_neighbors * 32),
                                             nn.GELU(),
                                             nn.Linear(num_neighbors * 32, num_neighbors * 16),
                                             nn.GELU(),
                                             nn.Linear(num_neighbors * 16, num_neighbors)
                                             )

        self.regressor_q = torch.nn.Sequential(nn.Linear(num_neighbors * dim, num_neighbors * 64),
                                               nn.GELU(),
                                               nn.Linear(num_neighbors * 64, num_neighbors * 32),
                                               nn.GELU(),
                                               nn.Linear(num_neighbors * 32, num_neighbors * 16),
                                               nn.GELU(),
                                               nn.Linear(num_neighbors * 16, num_neighbors)
                                               )


    def forward(self, ref_latent_x, ref_latent_q):
        weights_x = self.regressor_x(ref_latent_x.view(1, -1))
        weights_x = torch.nn.functional.softmax(weights_x, dim=1).squeeze(0).unsqueeze(1)
        latent_x = torch.sum(weights_x * ref_latent_x, dim=0).unsqueeze(0)
        weights_q = self.regressor_q(ref_latent_q.view(1, -1))
        weights_q = torch.nn.functional.softmax(weights_q, dim=1).squeeze(0).unsqueeze(1)
        latent_q = torch.sum(weights_q * ref_latent_q, dim=0).unsqueeze(0)
        return latent_x, latent_q, weights_x, weights_q


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

    pose_optim = PoseOptim(num_neighbors, config.get("hidden_dim")).to(device)

    # Set the losses
    loss = torch.nn.MSELoss().to(device)

    # Set the optimizer and scheduler
    lr = 1e-2  # config.get('lr')
    optim = torch.optim.SGD(pose_optim.parameters(), lr=lr)

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
            est_pose = res.get('pose').cpu().numpy()
            tic = time.time()
            latent_x = res.get("latent_x")
            latent_q = res.get("latent_q")
            if not is_single_scene:
                scene_dist = res.get('scene_log_distr')
                scene = torch.argmax(scene_dist, dim=1).to(dtype=torch.float32).unsqueeze(1).repeat(1,
                                                                                                    num_neighbors).view(
                    -1, 1)

            # get closest pose
            ref_pose = get_ref_pose(est_pose, dataset.poses, 0, num_neighbors)
            ref_pose = torch.Tensor(ref_pose).to(device).reshape(1 * num_neighbors, 7)
            if is_single_scene:
                ref_latent_x, ref_latent_q = pose_encoder(ref_pose)
            else:
                ref_latent_x, ref_latent_q = pose_encoder(ref_pose, scene)

        tic = time.time()
        pose_optim.train()
        for _ in range(3):

            optim.zero_grad()
            est_latent_x, est_latent_q, weights_x, weights_q = pose_optim(ref_latent_x, ref_latent_q)
            criterion = loss(latent_x, est_latent_x) + loss(latent_q, est_latent_q)
            # Back prop
            criterion.backward()
            optim.step()

        # Evaluate error
        pose_optim.eval()
        with torch.no_grad():
            est_x = torch.sum(weights_x * ref_pose[:, :3], dim=0).unsqueeze(0)
            est_q = torch.sum(weights_q * ref_pose[:, 3:], dim=0).unsqueeze(0)
            refined_est_pose = torch.cat((est_x, est_q), dim=1)
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





