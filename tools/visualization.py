import cv2
import numpy as np

import matplotlib
import matplotlib.cm


def gray_to_colormap(img, cmap='plasma', max_val=None):
    """
    Transfer gray map to matplotlib colormap
    """
    assert img.ndim == 2

    img[img<0] = 0
    mask_invalid = img < 1e-10
    if max_val is None:
        img = img / (img.max() + 1e-8)
    else:
        img = img / (max_val + 1e-8)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0
    return colormap

def depths_to_colors(depths, concat="width", max_val=None):
    colors = []
    for depth in depths:
        color = gray_to_colormap(depth.detach().cpu().numpy(), max_val=max_val)
        colors.append(color)
    if concat == "width":
        colors = np.concatenate(colors, axis=1)
    else:
        colors = np.stack(colors)
    return colors