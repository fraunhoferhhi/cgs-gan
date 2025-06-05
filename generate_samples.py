import os
import pickle
import click
import torch
from tqdm import tqdm

from dnnlib import EasyDict
from inference_helper import Logger
from inference_helper import save_ply
from inference_helper import MappingOperations

class ViewpointGrid:
    def __init__(self, settings):
        self.settings = settings
        self.logger = Logger(settings.save_dir)
        with open(settings.pkl, "rb") as input_file:
            self.generator = pickle.load(input_file)["G_ema"].to(settings.device)
        self.mapper = MappingOperations(self.generator, self.settings)
        self.z = self.mapper.get_random_z(settings.num_ids, state=settings.seed)
        self.w = self.mapper.map_to_w(self.z)

    def save_images(self):
        image = []
        for angles in tqdm(self.settings.head_angles, desc="Generating images"):
            cam = self.mapper.get_cam(angles)
            generated_tensor = self.generator(self.w, cam.repeat((self.settings.num_ids, 1)), random_bg=False)["image"]
            generated_tensor = torch.cat(generated_tensor.split(1), dim=2)
            image.append(generated_tensor)
        image = torch.cat(image, dim=3)
        self.logger.save_image(image, f"grid_seed_{self.settings.seed}")

    def save_ply(self):
        gs_scenes = self.generator(self.w, torch.zeros(self.w.shape[0], 25).to(self.settings.device), render_output=False)["gaussian_params"]
        for i in tqdm(range(self.settings.num_ids), desc="Generating ply"):
            gs_scene = EasyDict(gs_scenes[i])
            save_ply(gs_scene, os.path.join(self.logger.save_path, f"out_seed_{self.settings.seed}_index_{i}.ply"))


@click.command()
@click.option("--pkl",              type=str,   required=True,  help="path to network pkl")
@click.option("--device",           type=str,   default="cuda", help="default device: cpu or cuda")
@click.option("--truncation_psi",   type=float, default=0.8,    help="tradeoff between quality and variety (0 = quality and 1 = variety)")
@click.option("--num_ids",          type=int,   default=4,      help="number of ids to generate (number of rows)")
@click.option("--radius",           type=float, default=2.7,    help="radius of the camera")
@click.option("--seed",             type=int,   default=42)
@click.option("--save_dir",         type=str,   default="./results")
def main(**kwargs):
    opts = EasyDict(kwargs)
    # angles are defined in yaw and pitch with (0.5, 0.5) describing a frontal viewpoint
    # (0.0, 0.5) shows the head from 90° to the left and (1.0, 0.5) 90° to the right
    opts.head_angles = [
        (0.8, 0.5),
        (0.7, 0.5),
        (0.6, 0.5),
        (0.5, 0.5),
        (0.4, 0.5),
        (0.3, 0.5),
        (0.2, 0.5),
    ]
    sampler = ViewpointGrid(opts)
    sampler.save_images()
    sampler.save_ply()


if __name__ == "__main__":
    main()

