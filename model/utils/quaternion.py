"""Quaternion helpers for moving Gaussians from a camera frame to the world frame.

Convention: quaternions are stored as (w, x, y, z), the order the panorama
rasterizer reads them in (pano_gaussian/cuda_rasterizer/forward.cu, computeCov3D:
r = q.x, x = q.y, y = q.z, z = q.w). The rasterizer builds the standard Hamilton
rotation matrix R(q) and a covariance R S^2 R^T, so a Gaussian whose local
rotation is q_local lives in the world with rotation R_c2w @ R(q_local), i.e.
q_world = q(R_c2w) (x) q_local (Hamilton product).
"""

import torch


def matrix_to_quaternion_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (..., 3, 3) -> unit quaternions (..., 4) in wxyz order, w >= 0.

    Branch-free Shepperd-style conversion: every candidate is computed and the
    numerically best one (largest denominator) is selected per matrix.
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    # 4 * (w^2, x^2, y^2, z^2)
    q_abs_sq = torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1)
    q_abs = torch.sqrt(torch.clamp(q_abs_sq, min=0.0))

    # Row k is the quaternion scaled by 4 * q_k, using q_k as the pivot.
    candidates = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    candidates = candidates / (2.0 * torch.clamp(q_abs[..., None], min=0.1))

    best = q_abs.argmax(dim=-1)
    q = torch.gather(candidates, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    return q / q.norm(dim=-1, keepdim=True)


def quaternion_multiply_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a (x) b of wxyz quaternions, broadcasting over leading dims."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def compose_quaternion_c2w(q_local: torch.Tensor, R_c2w: torch.Tensor) -> torch.Tensor:
    """Rotate camera-frame Gaussian quaternions into the world frame.

    Args:
        q_local: (..., N, 4) wxyz quaternions in the camera frame (need not be unit;
            the norm is preserved, because q(R_c2w) is a unit quaternion).
        R_c2w: (..., 3, 3) camera-to-world rotation, broadcast over N.

    Returns:
        (..., N, 4) wxyz quaternions q(R_c2w) (x) q_local, same dtype as q_local.
    """
    q_rot = matrix_to_quaternion_wxyz(R_c2w.to(torch.float32)).to(q_local.dtype)
    return quaternion_multiply_wxyz(q_rot.unsqueeze(-2), q_local)
