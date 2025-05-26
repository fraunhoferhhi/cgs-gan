import PIL.Image
import numpy as np
import torch
import cv2


def setup_snapshot_image_grid(training_set, random_seed=0, gw=8, gh=8):
    rnd = np.random.RandomState(random_seed)
    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


def save_image_grid(img, fname, drange, grid_size, wandb_logger=None):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    if wandb_logger is not None:
        import wandb
        log_img = wandb.Image(img)
        wandb_logger.log({"samples": log_img})
    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def print_stats(_dict):
    for k in _dict.keys():
        if k == "_features_rest":
            continue
        print("{} shape: {}, min: {} max: {}".format(k, _dict[k].shape, _dict[k].min(), _dict[k].max()))


def save_images(rgb_image, depth_image, device=None):
    if str(device) == "cuda:0":
        # save intermediate image for debugging
        temp_img = rgb_image.detach().cpu().numpy()[0]
        temp_img = np.clip(((temp_img + 1) * 127.5), 0, 255)
        cv2.imwrite(
            "temp_saves/temp_G_{}_{}.jpg".format(str(rgb_image.device), 0),
            temp_img.transpose([1, 2, 0])[:, :, ::-1],
        )
        temp_img = rgb_image.detach().cpu().numpy()[1]
        temp_img = np.clip(((temp_img + 1) * 127.5), 0, 255)
        cv2.imwrite(
            "temp_saves/temp_G_{}_{}.jpg".format(str(rgb_image.device), 1),
            temp_img.transpose([1, 2, 0])[:, :, ::-1],
        )

        temp_depth = depth_image.detach().cpu().numpy()[0]
        temp_depth = (temp_depth - temp_depth.min()) / (temp_depth.max() - temp_depth.min()) * 255
        cv2.imwrite(
            "temp_saves/depth_G_{}_{}.jpg".format(str(depth_image.device), 0),
            temp_depth.transpose([1, 2, 0])[:, :, ::-1],
        )


def slerp(v0, v1, t):
    v0 = torch.nn.functional.normalize(v0, p=2, dim=-1)
    v1 = torch.nn.functional.normalize(v1, p=2, dim=-1)

    dot_product = torch.einsum("bi,bi->b", v0, v1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    theta = torch.acos(dot_product)

    sin_theta = torch.sin(theta)
    interpolated_vector = (torch.sin((1 - t) * theta) / sin_theta)[:, None] * v0 + (torch.sin(t * theta) / sin_theta)[
        :, None
    ] * v1

    return interpolated_vector
