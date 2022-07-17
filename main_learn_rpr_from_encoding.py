import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.CameraPoseDatasetExt import CameraPoseDatasetExt
from models.pose_losses import CameraPoseLoss
from models.pose_regressors import get_model
from os.path import join
from main_reconstruct_img import Decoder
from models.pose_encoder import PoseEncoder, MultiSCenePoseEncoder
from models.posenet.RPoseNet import RPoseNet



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name",
                            help="name of model to create (e.g. posenet, transposenet")
    arg_parser.add_argument("mode", help="train or eval")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("config_file", help="path to configuration file", default="7scenes-config.json")
    arg_parser.add_argument("apr_checkpoint_path",
                            help="path to a pre-trained apr model (should match the model indicated in model_name")
    arg_parser.add_argument("encoder_checkpoint_path",
                            help="path to a pre-trained encoder model (should match the APR model")

    arg_parser.add_argument("decoder_checkpoint_path",
                            help="path to a pre-trained decoder model (should match the encoder model")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained RPR model (should match the model indicated in model_name")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")
    arg_parser.add_argument("--ref_poses_file", help="path to file with train poses")

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

    apr = get_model(args.model_name, args.backbone_path, config).to(device)
    apr.load_state_dict(torch.load(args.apr_checkpoint_path, map_location=device_id))
    logging.info("Initializing from checkpoint: {}".format(args.apr_checkpoint_path))
    apr.eval()

    is_single_scene = config.get("single_scene")
    if is_single_scene:
        pose_encoder = PoseEncoder(config.get("hidden_dim")).to(device)
    else:
        pose_encoder = MultiSCenePoseEncoder(config.get("hidden_dim")).to(device)
    pose_encoder.load_state_dict(torch.load(args.encoder_checkpoint_path, map_location=device_id))
    logging.info("Initializing encoder from checkpoint: {}".format(args.encoder_checkpoint_path))
    pose_encoder.eval()

    img_size = config.get("img_size")
    decoder = Decoder(config.get("hidden_dim"), img_size).to(device)
    decoder.load_state_dict(torch.load(args.decoder_checkpoint_path, map_location=device_id))
    logging.info("Initializing encoder from checkpoint: {}".format(args.decoder_checkpoint_path))
    decoder.eval()

    # Create the RPR model
    config["backbone_type"] = "efficientnet"
    model = RPoseNet(config, args.backbone_path).to(device)
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    decoder_transform = utils.get_base_transform(img_size)
    normalize_transform = utils.get_normalize_transform()

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)

        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))


        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDatasetExt(args.dataset_path, args.labels_file, decoder_transform, transform, False)
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
                minibatch['scene'] = None  # avoid using ground-truth scene during prediction
                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                with torch.no_grad():
                    # estimate pose and scene
                    res = apr(minibatch)
                    est_pose = res.get('pose').cpu().numpy()
                    scene_dist = res.get('scene_log_distr')
                    scene = torch.argmax(scene_dist, dim=1).to(dtype=torch.float32).unsqueeze(1)

                    # get closest pose
                    closest_pose = np.zeros((batch_size, 7))

                    for i, p in enumerate(est_pose):
                        dist_x = np.linalg.norm(p[:3] - dataset.poses[:, :3], axis=1)
                        dist_x = dist_x / np.max(dist_x)
                        dist_q = np.linalg.norm(p[3:] - dataset.poses[:, 3:], axis=1)
                        dist_q = dist_q / np.max(dist_q)
                        sorted = np.argsort(dist_x + dist_q)
                        closest_pose[i, :] = (dataset.poses[sorted[1]])
                    closest_pose = torch.Tensor(closest_pose).to(device).to(torch.float32)

                    # Encode the pose
                    if not is_single_scene:
                        latent_x, latent_q = pose_encoder(gt_pose, scene)
                    else:
                        latent_x, latent_q = pose_encoder(gt_pose)

                    # Reconstruct the image
                    rec_img = decoder(torch.cat((latent_x, latent_q), dim=1))

                    small_img = minibatch.get("small_img")
                    # Normalize both images
                    for j in range(batch_size):
                        rec_img[j] = normalize_transform(rec_img[j])
                        small_img[j] = normalize_transform(small_img[j])

                # Use RPR to estimate the pose
                # Zero the gradients
                optim.zero_grad()

                res = model(small_img, rec_img, closest_pose)
                est_pose = res.get('pose')
                # Pose loss
                criterion = pose_loss(est_pose, gt_pose)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_pose.detach(), gt_pose.detach())
                    logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                                 "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item()))
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_rpr_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_rpr_final.pth'.format(epoch))


    else: # Test

        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDatasetExt(args.dataset_path, args.labels_file, decoder_transform, transform, False)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats_encoder = np.zeros(len(dataloader.dataset))
        stats_retrieval = np.zeros(len(dataloader.dataset))
        stats_decoder = np.zeros(len(dataloader.dataset))
        stats = np.zeros((len(dataloader.dataset), 3))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_scene = minibatch.get('scene')
                minibatch['scene'] = None # avoid using ground-truth scene during prediction

                gt_pose = minibatch.get('pose').to(dtype=torch.float32)

                # Forward pass to predict the pose


                # estimate pose and scene
                res = apr(minibatch)

                est_pose = res.get('pose')
                q_from_apr = est_pose[:, 3:]
                scene_dist = res.get('scene_log_distr')
                scene = torch.argmax(scene_dist, dim=1).to(dtype=torch.float32).unsqueeze(1)

                # get closest pose
                closest_pose = np.zeros((1, 7))
                tic = time.time()
                for j, p in enumerate(est_pose.cpu().numpy()):
                    dist_x = np.linalg.norm(p[:3] - dataset.poses[:, :3], axis=1)
                    dist_x = dist_x / np.max(dist_x)
                    dist_q = np.linalg.norm(p[3:] - dataset.poses[:, 3:], axis=1)
                    dist_q = dist_q / np.max(dist_q)
                    sorted = np.argsort(dist_x + dist_q)
                    closest_pose[j, :] = (dataset.poses[sorted[0]])
                stats_retrieval[i] = (time.time() - tic)*1000
                closest_pose = torch.Tensor(closest_pose).to(device).to(torch.float32)

                # Encode the pose
                tic = time.time()
                if not is_single_scene:
                    latent_x, latent_q = pose_encoder(gt_pose, scene)
                else:
                    latent_x, latent_q = pose_encoder(gt_pose)
                torch.cuda.synchronize()
                stats_encoder[i] = (time.time() - tic)*1000

                tic = time.time()
                # Reconstruct the image
                rec_img = decoder(torch.cat((latent_x, latent_q), dim=1))
                torch.cuda.synchronize()
                stats_decoder[i] = (time.time() - tic)*1000

                small_img = minibatch.get("small_img")

                # Normalize both images
                rec_img[0] = normalize_transform(rec_img[0])
                small_img[0] = normalize_transform(small_img[0])

                # Use RPR to estimate the pose
                tic = time.time()
                res = model(small_img, rec_img, closest_pose)
                toc = time.time()
                torch.cuda.synchronize()
                est_pose = res.get('pose')
                est_pose[:, 3:] = q_from_apr


                # Evaluate error
                posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

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
        logging.info("Mean retrieval inference time:{:.2f}[ms]".format(np.mean(stats_retrieval)))
        logging.info("Mean encoder inference time:{:.2f}[ms]".format(np.mean(stats_encoder)))
        logging.info("Mean decoder inference time:{:.2f}[ms]".format(np.mean(stats_decoder)))






