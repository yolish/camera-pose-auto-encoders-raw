import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_losses import CameraPoseLoss
from models.pose_regressors import get_model
from os.path import join
from models.pose_encoder import PoseEncoder
from models.residual_pose_regressor import ResidualPoseRegressor


def generate_ref_pose(gt_poses, db_poses, k=5):
    ref_poses = np.zeros((gt_poses.shape[0], 7))
    for i, p in enumerate(gt_poses):
        dist_x = np.linalg.norm(p[:3] - db_poses[:, :3], axis=1)
        dist_x = dist_x / np.max(dist_x)
        dist_q = np.linalg.norm(p[3:] - db_poses[:, 3:], axis=1)
        dist_q = dist_q / np.max(dist_q)
        sorted = np.argsort(dist_x + dist_q)
        closest_idx = np.random.choice(sorted[1:(k+1)], size=1)[0]
        ref_poses[i, :] = db_poses[closest_idx, :]
    return ref_poses


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name",
                            help="name of model to create (e.g. posenet, transposenet")
    arg_parser.add_argument("mode", help="train or eval")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("config_file", help="path to configuration file", default="7scenes-config.json")
    arg_parser.add_argument("checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("encoder_checkpoint_path", help="path to a trained pose encoder")
    arg_parser.add_argument("--rpr_checkpoint_path", help="path to rpr component")
    arg_parser.add_argument("--ref_poses_file", help="path to file with train poses")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {} with {}".format(args.model_name, args.mode))
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

    pose_encoder = PoseEncoder(config.get("hidden_dim")).to(device)
    pose_encoder.load_state_dict(torch.load(args.encoder_checkpoint_path, map_location=device_id))
    logging.info("Initializing encoder from checkpoint: {}".format(args.encoder_checkpoint_path))
    pose_encoder.eval()

    rpr = ResidualPoseRegressor(config.get("hidden_dim")).to(device)
    if args.rpr_checkpoint_path is not None:
        rpr.load_state_dict(torch.load(args.rpr_checkpoint_path, map_location=device_id))
        logging.info("Initializing encoder from checkpoint: {}".format(args.rpr_checkpoint_path))


    if args.mode == 'train':
        # Set to train mode
        rpr.train()

        # Set the losses
        pose_loss = CameraPoseLoss(config).to(device)

        # Set the optimizer and scheduler
        params = list(rpr.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform, False)
        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                gt_scene = minibatch.get('scene').to(device)
                minibatch['scene'] = None
                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                # Zero the gradients
                optim.zero_grad()

                with torch.no_grad():
                    res = apr(minibatch)
                    query_est_pose = res.get('pose')
                    query_latent_x = res.get("latent_x")
                    query_latent_q = res.get("latent_q")

                    # randomly compute a target pose
                    ref_pose = generate_ref_pose(gt_pose.cpu().numpy(), dataset.poses)
                    ref_pose = torch.Tensor(ref_pose).to(device)
                    ref_latent_x, ref_latent_q = pose_encoder(ref_pose)

                delta_x, delta_q = rpr(query_latent_x, ref_latent_x, query_latent_q, ref_latent_q)
                delta = torch.cat((delta_x, delta_q), dim=1)
                refined_est_pose = ref_pose + delta

                criterion = pose_loss(refined_est_pose, gt_pose)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(refined_est_pose.detach(), gt_pose.detach())
                    logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                                 "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item()))
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(rpr.state_dict(), checkpoint_prefix + '_rpr_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(rpr.state_dict(), checkpoint_prefix + '_rpr_final.pth'.format(epoch))


    else: # Test

        # Set to eval mode
        apr.eval()
        pose_encoder.eval()
        rpr.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        ref_poses = CameraPoseDataset(args.dataset_path, args.ref_poses_file, None).poses
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)


        stats = np.zeros((len(dataloader.dataset), 3))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_scene = minibatch.get('scene')
                minibatch['scene'] = None # avoid using ground-truth scene during prediction

                gt_pose = minibatch.get('pose').to(dtype=torch.float32)

                # Forward pass to predict the pose
                tic = time.time()
                res = apr(minibatch)
                est_pose = res.get('pose').squeeze(0).cpu().numpy()

                latent_x = res.get("latent_x")
                latent_q = res.get("latent_q")

                # get closest pose
                dist_x = np.linalg.norm(est_pose[:3]-ref_poses[:, :3], axis=1)
                dist_x = dist_x/np.max(dist_x)
                dist_q = np.linalg.norm(est_pose[3:]-ref_poses[:, 3:], axis=1)
                dist_q = dist_q/np.max(dist_q)
                '''
                closest_idx = np.argmin(dist_x+dist_q)
                ref_pose = torch.Tensor(ref_poses[closest_idx]).to(device).unsqueeze(0)

                latent_x = res.get("latent_x")
                latent_q = res.get("latent_q")

                ref_latent_x, ref_latent_q = pose_encoder(ref_pose)
                delta_x, delta_q = rpr(latent_x, ref_latent_x, latent_q, ref_latent_q)

                delta = torch.cat((delta_x, delta_q), dim=1)
                refined_est_pose = ref_pose + delta
                '''
                k = 1
                refined_est_pose = None
                closest =  np.argsort(dist_x+dist_q)[:k]
                for closest_idx in closest:
                    ref_pose = torch.Tensor(ref_poses[closest_idx]).to(device).unsqueeze(0)

                    ref_latent_x, ref_latent_q = pose_encoder(ref_pose)
                    delta_x, delta_q = rpr(latent_x, ref_latent_x, latent_q, ref_latent_q)

                    delta = torch.cat((delta_x, delta_q), dim=1)
                    if refined_est_pose is None:
                        refined_est_pose = ref_pose + delta
                    else:
                        refined_est_pose += ref_pose + delta
                refined_est_pose /= k
                toc = time.time()

                # Evaluate error
                posit_err, orient_err = utils.pose_err(refined_est_pose, gt_pose)

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





