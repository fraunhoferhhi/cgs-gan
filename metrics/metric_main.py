# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main API for computing and reporting quality metrics."""

import os
import time
import json
import torch
import dnnlib

from . import metric_utils
from . import frechet_inception_distance
from . import frechet_inception_distance_3d

#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

#----------------------------------------------------------------------------

def calc_metric(metric, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

#----------------------------------------------------------------------------

def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')


@register_metric
def fid2k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=2000)
    return dict(fid2k_full=fid)


@register_metric
def fid20k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=20000)
    return dict(fid20k_full=fid)


@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)


# 3D consistency metrics
@register_metric
def fid3d_2k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance_3d.compute_fid_3d(opts, max_real=None, num_gen=2000)
    return dict(fid2k_full=fid)

@register_metric
def fid3d_20k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance_3d.compute_fid_3d(opts, max_real=None, num_gen=20000)
    return dict(fid20k_full=fid)

@register_metric
def fid3d_50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance_3d.compute_fid_3d(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)
