import os
import time
import copy
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from torch_utils.logger import CustomLogger
import wandb
import load_network
from metrics import metric_main
from training.training_utils import save_image_grid, setup_snapshot_image_grid


def training_loop(
        run_dir='.',                # Output directory.
        training_set_kwargs={},     # Options for training set.
        data_loader_kwargs={},      # Options for torch.utils.data.DataLoader.
        G_kwargs={},                # Options for generator network.
        D_kwargs={},                # Options for discriminator network.
        G_opt_kwargs={},            # Options for generator optimizer.
        D_opt_kwargs={},            # Options for discriminator optimizer.
        loss_kwargs={},             # Options for loss function.
        metrics=[],                 # Metrics to evaluate during training.
        random_seed=0,              # Global random seed.
        num_gpus=1,                 # Number of GPUs participating in the training.
        rank=0,                     # Rank of the current process in [0, num_gpus[.
        batch_size=4,               # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
        batch_gpu=4,                # Number of samples processed at a time by one GPU.
        ema_kimg=10,                # Half-life of the exponential moving average (EMA) of generator weights.
        ema_rampup=0.05,            # EMA ramp-up coefficient. None = no rampup.
        G_reg_interval=None,        # How often to perform regularization for G? None = disable lazy regularization.
        D_reg_interval=16,          # How often to perform regularization for D? None = disable lazy regularization.
        total_kimg=25000,           # Total length of the training, measured in thousands of real images.
        kimg_per_tick=4,            # Progress snapshot interval.
        image_snapshot_ticks=50,    # How often to save image snapshots? None = disable.
        network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
        resume_pkl=None,            # Network pickle to resume training from.
        resume_kimg=0,              # First kimg to report when resuming training.
        cudnn_benchmark=True,       # Enable torch.backends.cudnn.benchmark?
        abort_fn=None,              # Callback function for determining whether to abort training. Must return consistent results across ranks.
        progress_fn=None,           # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = False  # Avoids errors with the augmentation pipe.


    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size // num_gpus, **data_loader_kwargs))
    eval_training_set = dict(training_set_kwargs)
    eval_training_set["rand_background"] = False
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.0
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = load_network.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.rand([batch_gpu, G.z_dim], device=device)
        cam = torch.tensor([[
            0.942, 0.0342, 0.333, -0.836,
            0.039, -0.999, -0.009, 0.017,
            0.332, 0.0225, -0.942, 2.566,
            0.0, 0.0, 0.0, 1.0,
            4.2647, 0.0, 0.5,
            0.0, 4.2647, 0.5,
            0.0, 0.0, 1.0
        ]], device=device)  # do not use random or zero to avoid gpu memory blow up
        c = torch.tile(cam, [batch_gpu, 1])
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])
        torch.cuda.empty_cache()

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, **loss_kwargs)  # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            if name == "G":
                model_params, gaussian_params, gaussian_params_names = [], [], []
                for n, p in module.named_parameters():
                    if n == '_xyz':
                        gaussian_params.append(p)
                        gaussian_params_names.append("xyz")
                    elif n == '_scale':
                        gaussian_params.append(p)
                        gaussian_params_names.append("scaling")
                    elif n == '_rotation':
                        gaussian_params.append(p)
                        gaussian_params_names.append("rotation")
                    else:
                        model_params.append(p)

                assert len(list(module.parameters())) == len(list(module.named_parameters())), "param: {} \t named_param:{}".format(
                    len(list(module.parameters())), len(list(module.named_parameters())))
                params_groups = []
                params_groups.append({"params": model_params})
                for p, n in zip(gaussian_params, gaussian_params_names):
                    params_groups.append({"params": p, "name": n})

                opt = dnnlib.util.construct_class_by_name(params_groups, **opt_kwargs)  # subclass of torch.optim.Optimizer
            else:
                opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer

            phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]

        else:  # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]

            if name == "G":
                model_params, gaussian_params, gaussian_params_names = [], [], []
                for n, p in module.named_parameters():
                    if n == '_xyz':
                        gaussian_params.append(p)
                        gaussian_params_names.append("xyz")  # name?
                    elif n == '_scale':
                        gaussian_params.append(p)
                        gaussian_params_names.append("scaling")  # name?
                    elif n == '_rotation':
                        gaussian_params.append(p)
                        gaussian_params_names.append("rotation")  # name?
                    else:
                        model_params.append(p)

                assert len(list(module.parameters())) == len(list(module.named_parameters())), "param: {} \t named_param:{}".format(
                    len(list(module.parameters())), len(list(module.named_parameters())))

                params_groups = []
                params_groups.append({"params": model_params})
                for p, n in zip(gaussian_params, gaussian_params_names):
                    params_groups.append({"params": [p], "name": n})

                opt = dnnlib.util.construct_class_by_name(params_groups, **opt_kwargs)  # subclass of torch.optim.Optimizer
            else:
                opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + 'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name + 'reg', module=module, opt=opt, interval=reg_interval)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    if (resume_pkl is not None) and (rank == 0):
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = load_network.load_network_pkl(f)
        print("loading optimizer states")
        for phase in phases:
            phase.opt.load_state_dict(resume_data[f"{phase.name}_opt"])
            phase.interval = resume_data[f"{phase.name}_interval"]

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set, gw=5, gh=5)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0, 255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)

    # Initialize logs.
    stats_metrics = {}
    logger = CustomLogger()
    wandb_logger = None
    if rank == 0:
        print('Initializing logs...', rank)
        config = dict(
            training_set_kwargs=training_set_kwargs,
            data_loader_kwargs=data_loader_kwargs,
            G_kwargs=G_kwargs,
            D_kwargs=D_kwargs,
            G_opt_kwargs=G_opt_kwargs,
            D_opt_kwargs=D_opt_kwargs,
            loss_kwargs=loss_kwargs
        )
        name = run_dir.split("/")[-1]
        wandb_logger = wandb.init(project="CGS GAN", dir=run_dir, name=name, config=config)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg,
                                          logger=logger)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))

            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

            G_ema.resolution = G.resolution
            G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
            # everything after this line is performed once per tick

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {cur_tick:<5d}"]
        fields += [f"kimg {cur_nimg / 1e3:<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(tick_end_time - start_time):<12s}"]
        fields += [f"sec/tick {tick_end_time - tick_start_time:<7.1f}"]
        fields += [f"sec/kimg {(tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3:<7.2f}"]
        fields += [f"maintenance {maintenance_time:<6.1f}"]
        fields += [f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2 ** 30:<6.2f}"]
        fields += [f"gpumem {torch.cuda.max_memory_allocated(device) / 2 ** 30:<6.2f}"]
        fields += [f"reserved {torch.cuda.max_memory_reserved(device) / 2 ** 30:<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        hours = (tick_end_time - start_time) / (60 * 60)
        logger.add("Timing", "total_hours", hours)
        logger.add("Timing", "total_days", hours / 24)

        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (done or cur_tick % image_snapshot_ticks == 0):
            out = [G_ema(z=z, c=c, noise_mode='const', random_bg=False)["image"] for z, c in zip(grid_z, grid_c)]
            images = torch.cat(out).cpu().detach().numpy()
            torch.cuda.empty_cache()
            save_image_grid(
                images, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}.png'), drange=[-1, 1],
                grid_size=grid_size, wandb_logger=wandb_logger
            )

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if done or cur_tick % network_snapshot_ticks == 0:
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
                if module is not None:
                    if num_gpus > 1:
                        pass
                    module.load_state_dict(copy.deepcopy(module.state_dict()))

                snapshot_data[name] = module
                del module  # conserve memory
            for phase in phases:
                snapshot_data[f"{phase.name}_opt"] = phase.opt.state_dict()
                snapshot_data[f"{phase.name}_interval"] = phase.interval
                snapshot_data[f"{phase.name}_module"] = phase.module

            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0) and cur_tick > 0:
            if rank == 0:
                print(run_dir)
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=G_ema, dataset_kwargs=eval_training_set, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir)
                stats_metrics.update(result_dict.results)
        del snapshot_data  # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            logger.add('Timing', phase.name, value)
        if rank == 0:
            wandb_logger.log(logger.content, step=cur_nimg)
            wandb_logger.log(stats_metrics, step=cur_nimg)
        logger.reset()

        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

# ----------------------------------------------------------------------------
