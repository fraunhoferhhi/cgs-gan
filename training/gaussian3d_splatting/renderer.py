import math
import torch

from diff_gaussian_rasterization import  GaussianRasterizationSettings, GaussianRasterizer


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def strip_symmetric(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=r.device)
    R = build_rotation(r)
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    L = R @ L
    return L


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


class Renderer:
    def __init__(self, sh_degree=0, white_background=True, radius=1):
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius
        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
        )
                
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.active_sh_degree = 0
        self.max_sh_degree = 0
        
    def get_scaling(self, _scaling):
        return self.scaling_activation(_scaling)
    
    def get_rotation(self, _rotation):
        return self.rotation_activation(_rotation)
    
    def get_xyz(self):
        return self._xyz
    
    def get_features(self, features_dc, features_rest):
        return torch.cat((features_dc, features_rest), dim=1)
    
    def get_opacity(self, _opacity):
        return self.opacity_activation(_opacity)

    def render(
        self,
        gaussian_params,
        viewpoint_camera,
        bg=None,
        scaling_modifier=1.0,
        override_color=None,
        compute_cov3D_python=False,
    ):
        _xyz = gaussian_params["_xyz"]
        _features_dc = gaussian_params["_features_dc"]
        _features_rest = gaussian_params["_features_rest"]
        _scaling = gaussian_params["_scaling"]
        _rotation = gaussian_params["_rotation"]
        _opacity = gaussian_params["_opacity"]

        if bg is None:
            bg = self.bg_color.to(_xyz.device)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(_xyz, dtype=_xyz.dtype, requires_grad=True, device=_xyz.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.to(_xyz.device),
            projmatrix=viewpoint_camera.full_proj_transform.to(_xyz.device),
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center.to(_xyz.device),
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rasterizer = rasterizer.to(_xyz.device)
        
        means3D = _xyz
        means2D = screenspace_points
        opacity = self.get_opacity(_opacity)

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.get_covariance(scaling_modifier)
        else:
            scales = self.get_scaling(_scaling)
            rotations = self.get_rotation(_rotation)
        
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            shs = self.get_features(_features_dc, _features_rest)
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        with torch.autocast(device_type=_xyz.device.type, dtype=torch.float32):
            rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )

        rendered_image = rendered_image / 0.5 - 1.
        return {
            "image": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
