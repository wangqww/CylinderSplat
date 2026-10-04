"""Helpers for the `theta_periodic` switch (D5) of the cylindrical volume branch.

The cylinder angle theta (and the equirect u axis) is periodic. With the switch
on, the plane convolutions pad theta circularly, and every bilinear read along
theta (deformable attention on the planes and on the panorama features, the
decoder's colour/depth window) wraps the sampling location and reads the map
through a one-cell circular halo, so a tap at the seam interpolates between
the last and the first column instead of a zero pad.

Coordinate conventions are the ones the callers already use:
- deformable attention: locations in [0, 1] of the map, pixel = loc * size - 0.5
  (align_corners=False), spatial shapes (H, W), location xy = (column, row);
- F.grid_sample(align_corners=False): pixel = ((g + 1) * size - 1) / 2.
"""

import torch
import torch.nn.functional as F


def conv2d_theta_circular(conv, x, theta_dim):
    """Apply `conv` (an nn.Conv2d) with circular padding along `theta_dim` (2 = rows,
    3 = columns) and the conv's own zero padding along the other spatial axis."""
    ph, pw = conv.padding
    if theta_dim == 2:
        if ph > 0:
            x = torch.cat([x[:, :, -ph:], x, x[:, :, :ph]], dim=2)
        padding = (0, pw)
    elif theta_dim == 3:
        if pw > 0:
            x = torch.cat([x[..., -pw:], x, x[..., :pw]], dim=3)
        padding = (ph, 0)
    else:
        raise ValueError(f'theta_dim must be 2 or 3, got {theta_dim}')
    return F.conv2d(x, conv.weight, conv.bias, conv.stride, padding, conv.dilation, conv.groups)


def wrap_sampling_locations(sampling_locations, spatial_shapes, periodic_axes):
    """Wrap deformable-attention sampling locations along each level's periodic axis
    and express them in units of the circularly padded maps of `pad_levels_circular`.

    sampling_locations: (..., num_levels, num_points, 2), xy in [0, 1] units of the
        unpadded maps; spatial_shapes: (num_levels, 2) as (H, W);
    periodic_axes: one entry per level, None, 0 (x / columns) or 1 (y / rows).
    """
    shapes = spatial_shapes.tolist()
    levels = []
    for lvl, ((h, w), axis) in enumerate(zip(shapes, periodic_axes)):
        loc = sampling_locations[..., lvl, :, :]
        if axis is not None:
            size = w if axis == 0 else h
            c = loc[..., axis]
            c = c - torch.floor(c)  # wrap into [0, 1)
            c = (c * size + 1) / (size + 2)  # pixel + 1 in the padded map
            other = loc[..., 1 - axis]
            loc = torch.stack([c, other] if axis == 0 else [other, c], dim=-1)
        levels.append(loc)
    return torch.stack(levels, dim=-3)


def pad_levels_circular(value, spatial_shapes, periodic_axes):
    """Give each level of a flattened multi-level value map a one-cell circular halo
    along its periodic axis.

    value: (bs, sum(H * W), num_heads, dim); spatial_shapes: (num_levels, 2) as (H, W);
    periodic_axes as in `wrap_sampling_locations`.
    Returns (value, spatial_shapes, level_start_index) of the padded maps.
    """
    shapes = spatial_shapes.tolist()
    chunks, new_shapes = [], []
    start = 0
    for (h, w), axis in zip(shapes, periodic_axes):
        v = value[:, start:start + h * w]
        start += h * w
        if axis is not None:
            v = v.reshape(v.shape[0], h, w, *v.shape[2:])
            dim = 2 if axis == 0 else 1
            n = v.shape[dim]
            v = torch.cat([v.narrow(dim, n - 1, 1), v, v.narrow(dim, 0, 1)], dim=dim)
            h, w = (h, w + 2) if axis == 0 else (h + 2, w)
            v = v.flatten(1, 2)
        chunks.append(v)
        new_shapes.append((h, w))
    value = torch.cat(chunks, dim=1).contiguous()
    spatial_shapes = torch.as_tensor(
        new_shapes, dtype=spatial_shapes.dtype, device=spatial_shapes.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
    return value, spatial_shapes, level_start_index


def circular_pad_width(x):
    """(..., h, w) -> (..., h, w + 2): one column of circular padding on each side."""
    return torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)


def wrap_grid_x(grid, w):
    """Re-express F.grid_sample (align_corners=False) x coordinates for a map of width
    `w` in units of `circular_pad_width(map)`, wrapping the pixel coordinate modulo w.
    grid: (..., 2) with xy in the last dimension; y is returned unchanged."""
    gx = grid[..., 0:1]
    px = ((gx + 1) * w - 1) / 2  # pixel coordinate, pixel centres at integers
    px = px - w * torch.floor(px / w)  # wrap into [0, w)
    gx = (2 * (px + 1) + 1) / (w + 2) - 1
    return torch.cat([gx, grid[..., 1:2]], dim=-1)
