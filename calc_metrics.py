# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
import click
import json
import tempfile
import copy
import torch

import dnnlib
import load_network
from metrics import metric_main
from metrics import metric_utils
from torch_utils import custom_ops
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from train_helper import parse_comma_separated_list


#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    # Configure torch.
    device = torch.device('cuda', rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    # Print network summary.
    G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)
    if rank == 0 and args.verbose:
        z = torch.rand([1, G.z_dim], device=device)
        cam = torch.tensor([[
             0.942, 0.0342, 0.333, -0.836,
             0.039, -0.999, -0.009, 0.017,
             0.332, 0.0225, -0.942, 2.566,
             0.0, 0.0, 0.0, 1.0,
             4.2647, 0.0, 0.5, #4.2647, 0.0, 0.5,
             0.0, 4.2647, 0.5, #0.0, 4.2647, 0.5,
             0.0, 0.0, 1.0
        ]], device=device) # do not use random or zero to avoid gpu memory blow up
        c = torch.tile(cam, [1, 1])
        misc.print_module_summary(G, [z, c])

    G_kwargs = {"cam_interp_ratio": 1.0}


    # Calculate each metric.
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        # result_dict = metric_main.calc_metric(metric=metric, G=G, dataset_kwargs=args.dataset_kwargs,
        #     num_gpus=args.num_gpus, rank=rank, device=device, progress=progress, save_images=args.save_images, cache=args.cache)
        result_dict = metric_main.calc_metric(metric=metric, G=G, G_kwargs=G_kwargs, dataset_kwargs=args.dataset_kwargs,
            num_gpus=args.num_gpus, rank=rank, device=device, progress=progress, save_images=args.save_images, cache=args.cache)
        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir, snapshot_pkl=args.network_pkl)
        if rank == 0 and args.verbose:
            print()

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')


@click.command()
@click.pass_context
@click.option('network_pkl', '--network', help='Network pickle filename or URL', metavar='PATH', required=True)
@click.option('--metrics', help='Quality metrics', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--data', help='Dataset to evaluate against  [default: look up]', metavar='[ZIP|DIR]')
@click.option('--mirror', help='Enable dataset x-flips  [default: look up]', type=bool, metavar='BOOL', default=True)
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL', show_default=True)
@click.option('--save_images', help='Save images at temp/real and temp/fake', type=bool, default=False, metavar='BOOL', show_default=True)
@click.option('--cache', help='caching real dataset', type=bool, default=True, metavar='BOOL', show_default=True)
def calc_metrics(ctx, network_pkl, metrics, data, mirror, gpus, verbose, save_images, cache):
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, network_pkl=network_pkl, verbose=verbose, save_images=save_images, cache=cache)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')

    # Load network.
    if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
        ctx.fail('--network must point to a file or URL')
    if args.verbose:
        print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=args.verbose) as f:
        network_dict = load_network.load_network_pkl(f)
        args.G = network_dict['G_ema'] # subclass of torch.nn.Module

    # Initialize dataset options.
    if data is not None:
        if "FFHQC" in data and "FFHQC" not in data:
            cam_mode = "ffhq_default"
        elif "FFHQC" in data:
            cam_mode = "smile_pose_rebalancing"
        else:
            raise ValueError(f'Unknown dataset "{data}"')

        args.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, camera_sample_mode=cam_mode, rand_background=False)
    elif network_dict['training_set_kwargs'] is not None:
        args.dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
    else:
        ctx.fail('Could not look up dataset options; please specify --data')

    # Finalize dataset options.
    args.dataset_kwargs.resolution = args.G.resolution
    args.dataset_kwargs.use_labels = (args.G.c_dim != 0)
    if mirror is not None:
        args.dataset_kwargs.xflip = mirror

    # Print dataset options.
    if args.verbose:
        print('Dataset options:')
        print(json.dumps(args.dataset_kwargs, indent=2))

    # Locate run dir.
    args.run_dir = None
    if os.path.isfile(network_pkl):
        pkl_dir = os.path.dirname(network_pkl)
        if os.path.isfile(os.path.join(pkl_dir, 'training_options.json')):
            args.run_dir = pkl_dir

    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_metrics() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
