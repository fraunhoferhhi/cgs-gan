# CGS GAN

This repository is the official implementation of [CGS-GAN 3D Consistent Gaussian Splatting GANs for High Resolution Human Head Synthesis](). 

![alt text](assets/out_small.jpg "Teaser")

## Requirements

Install the conda environment and load the Gaussian splatting renderer:

Change the line `pytorch-cuda=11.8` in environment.yml to the installed cuda version on your machine. Otherwise diff-gaussian-rasterization fails to install. 
Any version >=11.8 should work.
```sh
conda env create -f environment.yml
conda activate cgsgan

git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
```

## Training

To enable logging with wandb, use `wandb login` before the first time you launch the training.
```sh
wandb login
```

To train our models run:

```sh
# Train with FFHQC
# 256
python train.py --data=path/FFHQC/256

# 512
python train.py --data=path/FFHQC/512

# 1024
python train.py --data=path/FFHQC/1024

# 2048
python train.py --data=path/FFHQC/2048

# Train with vanilla FFHQ
python train.py --data=path/FFHQ/512  --cam_sample_mode=ffhq_default
```

Further optional parameters are:
- `--gamma = 1.0` (R1 regularization weight)
- `--gpus = 4` (training with multiple GPUs)
- `--batch_gpu = 8` (split the training batchsize of 32 into smaller batches >=4)
- `--use_multivew_reg = True` (activate / deactivate multiview regularization)
- `--num_multiview = 4` (number of views per multiview regularization step)
- `--desc = new_experiment` (name your run so that you find it in wandb)

Load Checkpoint:
- `--resume = path/to/network.pkl` (resume the training from a network checkpoint)
- `--resume_kimg = 5000` (continue counting from the loaded checkpoint)

## Evaluation

Evaluate the checkpoint using FID and FID<sub>3D</sub>. Select the respective data path for the matching resolution.
```shell
# FID 50k
python calc_metrics.py --network path/to/network.pkl --data data/FFHQC/512 --metrics fid50k_full

# FID_3D 50k
python calc_metrics.py --network path/to/network.pkl --data data/FFHQC/512 --metrics fid3d_50k_full
```

## Dataset Download

You can download the FFHQC dataset here:
- [FFHQC](https://huggingface.co/anonym892312603527/neurips25/resolve/main/FFHQC.tar?download=true)

## Pre-trained Models

You can download our pretrained models here:

FFHQC
- [ffhq_512.pkl](https://huggingface.co/anonym892312603527/neurips25/resolve/main/models/ffhq_512.pkl?download=true)
- [ffhq_1024.pkl](https://huggingface.co/anonym892312603527/neurips25/resolve/main/models/ffhqc_1024.pkl?download=true)
- [ffhq_2048.pkl](https://huggingface.co/anonym892312603527/neurips25/resolve/main/models/ffhqc_2048.pkl?download=true) (still training, preliminary checkpoint)

Vanilla FFHQ
- [ffhq_512.pkl](https://huggingface.co/anonym892312603527/neurips25/resolve/main/models/ffhq_512.pkl?download=true)


## Visualize

Visualize our results using https://github.com/Florian-Barthel/splatviz
<img src="assets/gan_mode.png" style="width: 600px;">


## Results


| FID    | FFHQ 512  | FFHQC 512 | FFHQC 1024 | FFHQC 2048 |
|--------|-----------|-----------|------------|-----------|
| GSGAN  | 5.02      | 5.17      | /          | /         |
| GGHead | **4.34**  | 5.37      | 9.91       | /         |
| Ours   | 4.94      | **4.53**  | **5.25**   | **7.8**   |

| FID<sub>3D</sub> | FFHQ 512  | FFHQC 512 | FFHQC 1024 | FFHQC 2048 |
|------------------|-----------|-----------|------------|------------|
| GSGAN            | 10.50     | 7.68      | /          | /          |
| GGHead           | 7.90      | 7.78      | 14.27      | /          |
| Ours             | **4.94**  | **4.53**  | **5.25**   | **7.8**    |

