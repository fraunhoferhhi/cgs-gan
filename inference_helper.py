import cv2
import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement

from camera_utils import LookAtPoseSampler


### Generate cams and latent vectors

class MappingOperations:
    def __init__(self, generator, settings):
        self.generator = generator
        self.settings = settings

    def get_random_z(self, num, state=123):
        z_samples = np.random.RandomState(state).randn(num, 512)
        return z_samples

    def map_to_w(self, z):
        z = torch.from_numpy(z).to(self.settings.device)
        empty_cam = torch.zeros([z.shape[0], 25]).to(self.settings.device)
        w_samples = self.generator.mapping(z, empty_cam, truncation_psi=self.settings["truncation_psi"])
        return w_samples[:, 0, :]

    def get_cam(self, angles, stdev=0.0, num=1):
        focal_length = 4.2647
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=self.settings.device)
        intrinsics = intrinsics.reshape(-1, 9)
        intrinsics = intrinsics.repeat(num, 1)
        point = [0, 0, 0]
        camera_lookat_point = torch.tensor(point, device=self.settings.device)
        cam2world_pose = LookAtPoseSampler.sample(
            3.14 * angles[0],
            3.14 * angles[1],
            camera_lookat_point,
            horizontal_stddev=stdev,
            vertical_stddev=stdev,
            batch_size=num,
            radius=self.settings.radius,
            device=self.settings.device,
        )
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        return c


### Save ply files

def save_ply(gs, path):
    xyz = gs._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gs._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    if gs._features_rest.shape[0] == 0:
        f_rest = torch.zeros((gs._features_dc.shape[0], 3, (3 + 1) ** 2)).float().cuda()[:, :, 1:]
        f_rest = f_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    else:
        f_rest = gs._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    opacities = gs._opacity.detach().cpu().numpy()
    scale = gs._scaling.detach().cpu().numpy()
    rotation = gs._rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(gs)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def construct_list_of_attributes(gs):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(gs._features_dc.shape[1]*gs._features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(gs._features_rest.shape[1]*gs._features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(gs._scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(gs._rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l


### Logging results

def tensor_to_image(tensor, normalize=True):
    image = tensor.detach().cpu().numpy().squeeze()
    if normalize:
        image = 255 * ((image + 1) / 2)
        image = image.clip(0, 255).astype(np.uint8)
    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)
    return image


class Logger:
    def __init__(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path

    def save_image(self, image, image_name):
        image = tensor_to_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{self.save_path}/{image_name}.png", image)
