# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Adversarial Generation of Hierarchical Gaussians for 3D Generative Model"

Code adapted from
"Alias-Free Generative Adversarial Networks".
and
"Efficient Geometry-aware 3D Generative Adversarial Networks."
and
"GSGAN"
"""
import click
import dnnlib
from metrics import metric_main
import numpy as np
from train_helper import init_dataset_kwargs, launch_training, parse_comma_separated_list


@click.command()
# Required.
@click.option("--data",             help="Training data",                           type=str,   required=True)
# Optional features.
@click.option("--cam_sample_mode",  help="found in custom dist",                    type=str,   default="smile_pose_rebalancing")
@click.option("--outdir",           help="Where to save the results",               type=str,   default="./training-results")
@click.option("--cond",             help="Train conditional model",                 type=bool,  default=True)
@click.option("--mirror",           help="Enable dataset x-flips",                  type=bool,  default=True)
@click.option("--freezed",          help="Freeze first layers of D",                type=int,   default=0)
# Misc hyperparameters.
@click.option("--gpus",             help="Number of GPUs to use",                   type=int,   default=1)
@click.option("--gamma",            help="R1 regularization weight",                type=float, default=1.0)
@click.option("--batch",            help="Total batch size",                        type=int,   default=32)
@click.option("--batch-gpu",        help="Limit batch size per GPU",                type=int,   default=8)
@click.option("--cbase",            help="Capacity multiplier",                     type=int,   default=32768)
@click.option("--cmax",             help="Max. feature maps",                       type=int,   default=512)
@click.option("--glr",              help="G learning rate",                         type=float, default=0.0025)
@click.option("--dlr",              help="D learning rate",                         type=float, default=0.002)
@click.option("--map-depth",        help="Mapping network depth ",                  type=int,   default=2)
@click.option("--mbstd-group",      help="Minibatch std group size",                type=int,   default=4)
# Misc settings.
@click.option("--metrics",          help="Quality metrics",                         type=parse_comma_separated_list, default="fid20k_full")
@click.option("--kimg",             help="Total training duration",                 type=int,   default=15000)
@click.option("--tick",             help="How often to print progress",             type=int,   default=1)
@click.option("--img_snap",         help="How often to save snapshots",             type=int,   default=50)
@click.option("--network_snap",     help="How often to save network pkl",           type=int,   default=500)
@click.option("--seed",             help="Random seed",                             type=int,   default=0)
@click.option("--nobench",          help="Disable cuDNN benchmarking",              type=bool,  default=False)
@click.option("--workers",          help="DataLoader worker processes",             type=int,   default=3)
@click.option("--blur_fade_kimg",   help="Blur over how many",                      type=int,   default=200)
@click.option("--disc_c_noise",     help="Strength of disc pose cond reg, in std.", type=float, default=0)
@click.option("--resume_blur",      help="Enable to blur even on resume",           type=bool,  default=False)
@click.option("--g_num_fp16_res",   help="Number of fp16 layers in generator",      type=int,   default=0)
@click.option("--d_num_fp16_res",   help="Number of fp16 layers in discriminator",  type=int,   default=4)
# Gaussian Splatting Config
@click.option("--gaussian_num_pts", help="the number of init gaussian.",            type=int,   default=512)
@click.option("--start_pe",         help="Positional encoding",                     type=bool,  default=True)
@click.option("--center_dists",     help="coeff of center dist.",                   type=float, default=1.0)
@click.option("--knn_dists",        help="loss scale for knn dists.",               type=float, default=20.0)
@click.option("--knn_num_ks",       help="number of cluster center.",               type=int,   default=64)
@click.option("--use_multivew_reg", help="compute grad for multiple views",         type=bool,  default=True)
@click.option("--num_multiview",    help="number of renderings per training step",  type=int,   default=4)
# Optional job description
@click.option("--desc",             help="String to include in result dir name",    type=str,   default="cgs_gan")
@click.option("--job_id",           help="slurm job id",                            type=str,   default="")
# Resume Training
@click.option("--resume",           help="Resume from given network pickle",        type=str)
@click.option("--resume_kimg",      help="Resume k images",                         type=int)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    c = dnnlib.EasyDict()

    # Generator
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.G_kwargs.class_name = "training.cgs_generator.CGSGenerator"
    c.G_kwargs.mapping_kwargs.num_layers = opts.map_depth
    c.G_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = opts.cmax
    c.G_kwargs.fused_modconv_default = "inference_only"

    # Discriminator
    c.D_kwargs = dnnlib.EasyDict(block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.D_kwargs.class_name = "training.discriminator.Discriminator"
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.D_kwargs.channel_base = opts.cbase
    c.D_kwargs.channel_max = opts.cmax
    c.D_kwargs.disc_c_noise = opts.disc_c_noise

    # Optimizer
    c.G_opt_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=[0, 0.99], eps=1e-8)
    c.G_opt_kwargs.lr = opts.glr
    c.D_opt_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=[0, 0.99], eps=1e-8)
    c.D_opt_kwargs.lr = opts.dlr

    # Training Data
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(class_name="training.dataset.ImageFolderDataset", data=opts.data, cam_sample_mode=opts.cam_sample_mode)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException("--cond=True requires labels specified in dataset.json")
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror
    c.data_loader_kwargs.num_workers = opts.workers

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = opts.img_snap
    c.network_snapshot_ticks = opts.network_snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException("--batch must be a multiple of --gpus")
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException("--batch must be a multiple of --gpus times --batch-gpu")
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException("--batch-gpu cannot be smaller than --mbstd")
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException("\n".join(["--metrics can only contain the following values:"] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    n_transformer_mapping = {256: 5, 512: 6, 1024: 7, 2048: 8}

    # Configuration about the model architecture
    c.G_kwargs.rendering_kwargs = {
        "image_resolution": c.training_set_kwargs.resolution,
        "custom_options":{
            "scale_init": np.log((1 / np.sqrt(opts.gaussian_num_pts)) * 0.5),
            "scale_threshold": np.log((1 / np.sqrt(opts.gaussian_num_pts)) * 0.5),
            "scale_end": np.log((1 / c.training_set_kwargs.resolution) * 0.5),
            "xyz_output_scale": 0.1,                            # Additional scale for coarsest xyz
            "res_end": c.training_set_kwargs.resolution // 4,
            "num_pts": opts.gaussian_num_pts,                   # Number of initial gaussians
            "use_start_pe": opts.start_pe,                      # Use positional encoding on learnable constant
            "n_transformer": n_transformer_mapping[c.training_set_kwargs.resolution]
        }
    }

    # Loss
    c.loss_kwargs = dnnlib.EasyDict(class_name="training.loss.StyleGAN2Loss")
    c.loss_kwargs.r1_gamma = opts.gamma
    c.loss_kwargs.loss_custom_options = {
        "knn_dists": opts.knn_dists,                    # coeff. of KNN distance
        "knn_num_ks": opts.knn_num_ks,                  # the number of KNN for calculating loss
        "center_dists": opts.center_dists,              # coeff. of center distance
        "is_resume": True if opts.resume is not None else False,
        "use_multivew_reg": opts.use_multivew_reg,
        "num_multiview": opts.num_multiview
    }
    c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
    c.loss_kwargs.blur_fade_kimg = c.batch_size * opts.blur_fade_kimg / 32 # Fade out the blur during the first N kimg.
    c.loss_kwargs.resolution = c.training_set_kwargs.resolution

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.resume_kimg = opts.resume_kimg
        c.ema_rampup = None # Disable EMA rampup.
        if not opts.resume_blur:
            c.loss_kwargs.blur_init_sigma = 0

    # Performance-related toggles.
    c.D_kwargs.num_fp16_res = opts.d_num_fp16_res
    c.D_kwargs.conv_clamp = 256 if opts.d_num_fp16_res > 0 else None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = opts.desc
    desc += f"_gpus-{c.num_gpus:d}"
    if opts.job_id != "":
        desc += f"_jobid-{opts.job_id}"

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir)


if __name__ == "__main__":
    main()