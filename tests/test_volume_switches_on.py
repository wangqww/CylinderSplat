"""D5 theta_periodic / D6 cell_center_anchor / D7 rgb_retrieval='visibility_softmax' switched ON.

CPU tests on small synthetic tensors (mmcv's pure-PyTorch deformable attention);
the `gpu` tests check the CUDA deformable-attention op against that path.
"""

import math
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch

from model.volume import cross_view_hybrid_attention as cvha_mod
from model.volume import image_cross_attention as ica_mod
from model.volume import tpvformer_encoder_cylinder as enc_mod
from model.volume import volume_gs_decoder_cylinder as dec_mod
from model.volume.theta_periodic import (conv2d_theta_circular, pad_levels_circular,
                                         wrap_sampling_locations)
from tests.test_switches_identity_volume import (  # noqa: F401 (cpu_encoders is a fixture)
    DIM, FEAT_H, FEAT_W, PC_RANGE, PILLARS, TPV_R, TPV_THETA, TPV_Z, build_decoder,
    build_encoder, cpu_encoders, decoder_inputs, deformable_attn_kwargs, encoder_inputs,
    hybrid_attn_cfg, perturb, plane_queries, plane_ref_2d, plane_shapes, to_device)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# D7: the only new state-dict entries (relative to the decoder; the stage models add the
# prefix 'volume_gs.gs_decoder.'), with their shapes.
D7_NEW_PARAMS = {
    'gaussian_to_color_vis.encoder.0.weight': (128, 36),
    'gaussian_to_color_vis.encoder.0.bias': (128,),
    'gaussian_to_color_vis.head.0.weight': (128, 128),
    'gaussian_to_color_vis.head.0.bias': (128,),
    'gaussian_to_color_vis.head.2.weight': (3, 128),
    'gaussian_to_color_vis.head.2.bias': (3,),
    'gaussian_to_color_vis.log_beta': (1,),
}


def circular_bilinear(img, px, py):
    """Bilinear read of img [C, H, W] at pixel coordinates px, py (pixel centres at integers,
    equal shapes): circular along x (the width), zero outside along y. Returns [C, *px.shape]."""
    _, h, w = img.shape
    x0, y0 = torch.floor(px), torch.floor(py)
    out = img.new_zeros((img.shape[0],) + tuple(px.shape))
    for dx in (0, 1):
        for dy in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            wgt = (1 - (px - xi).abs()) * (1 - (py - yi).abs())
            inside = ((yi >= 0) & (yi <= h - 1)).to(img.dtype)
            vals = img[:, yi.clamp(0, h - 1).long(), torch.remainder(xi, w).long()]
            out = out + vals * (wgt * inside)
    return out


# ============================================================================ D5 theta_periodic

@pytest.mark.parametrize('theta_dim', [2, 3])
def test_conv_pads_theta_circularly(theta_dim):
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 5, 3, 1, 1)
    x = torch.randn(2, 3, 6, 7)
    y = conv2d_theta_circular(conv, x, theta_dim)
    circ = (0, 0, 1, 1) if theta_dim == 2 else (1, 1, 0, 0)
    zero = (1, 1, 0, 0) if theta_dim == 2 else (0, 0, 1, 1)
    want = F.conv2d(F.pad(F.pad(x, circ, mode='circular'), zero), conv.weight, conv.bias)
    torch.testing.assert_close(y, want, rtol=0, atol=1e-6)
    # rolling theta rolls the output: theta = 0 and theta = 2pi - eps are neighbours
    torch.testing.assert_close(conv2d_theta_circular(conv, x.roll(3, theta_dim), theta_dim),
                               y.roll(3, theta_dim), rtol=0, atol=1e-6)
    # a feature in the last theta bin reaches the first bin; with zero padding it does not
    delta = torch.zeros(1, 3, 6, 7)
    last = (0, slice(None), -1, 3) if theta_dim == 2 else (0, slice(None), 3, -1)
    first = (0, slice(None), 0, 3) if theta_dim == 2 else (0, slice(None), 3, 0)
    delta[last] = 1.0
    with torch.no_grad():
        assert (conv2d_theta_circular(conv, delta, theta_dim)[first] - conv.bias).abs().max() > 1e-3
        torch.testing.assert_close(conv(delta)[first], conv.bias)


def msda_sample(value_map, loc, periodic_axes=None):
    """Read value_map [H, W, d] at locations loc [n, 2] (xy in [0, 1] map units) with mmcv's
    pure-PyTorch deformable-attention sampler; optionally through the D5 wrap + halo."""
    h, w, d = value_map.shape
    shapes = torch.tensor([[h, w]])
    value = value_map.reshape(1, h * w, 1, d)
    locs = loc.view(1, -1, 1, 1, 1, 2)
    attn = torch.ones(1, loc.shape[0], 1, 1, 1)
    if periodic_axes is not None:
        locs = wrap_sampling_locations(locs, shapes, periodic_axes)
        value, shapes, _ = pad_levels_circular(value, shapes, periodic_axes)
    return multi_scale_deformable_attn_pytorch(value, shapes, locs, attn)[0]  # [n, d]


@pytest.mark.parametrize('axis', [0, 1])
def test_wrapped_deformable_sampling_is_circular(axis):
    torch.manual_seed(0)
    h, w, d = 5, 8, 3
    value_map = torch.randn(h, w, d)
    loc = torch.rand(300, 2)
    loc[:, axis] = loc[:, axis] * 4 - 1.5  # far outside [0, 1] along the periodic axis
    got = msda_sample(value_map, loc, [axis])
    img = value_map.permute(2, 0, 1)
    if axis == 0:
        want = circular_bilinear(img, loc[:, 0] * w - 0.5, loc[:, 1] * h - 0.5)
    else:
        want = circular_bilinear(img.transpose(1, 2), loc[:, 1] * h - 0.5, loc[:, 0] * w - 0.5)
    torch.testing.assert_close(got, want.T, rtol=0, atol=1e-5)


def test_wrapped_deformable_sampling_reads_across_the_seam():
    h, w = 5, 8
    value_map = torch.zeros(h, w, 1)
    value_map[:, 0] = 1.0  # a feature in the first column (u = 0)
    loc = torch.tensor([[1 + 0.25 / w, 0.5], [0.25 / w - 1, 0.5]])  # a quarter pixel past u = 1 / before u = 0
    torch.testing.assert_close(msda_sample(value_map, loc, [0]), torch.full((2, 1), 0.75))
    torch.testing.assert_close(msda_sample(value_map, loc), torch.zeros(2, 1))  # today: zero pad


def image_attention_inputs(bs=3, seed=4):
    g = torch.Generator().manual_seed(seed)
    lens = [5, 7, 3]
    query = [torch.randn(bs, n, DIM, generator=g) for n in lens]
    refs = [torch.rand(bs, n, p, 2, generator=g) for n, p in zip(lens, PILLARS)]
    value = torch.randn(bs, FEAT_H * FEAT_W, DIM, generator=g)
    return query, refs, value, torch.tensor([[FEAT_H, FEAT_W]]), torch.tensor([0])


def run_image_attention(attn, inputs, shift=0.0):
    query, refs, value, ss, lsi = inputs
    moved = [r + torch.tensor([shift, 0.0], device=r.device) for r in refs]
    return attn(query, value=value, reference_points=moved, spatial_shapes=ss, level_start_index=lsi)


def test_image_attention_is_periodic_in_u():
    torch.manual_seed(0)
    attn = ica_mod.TPVMSDeformableAttention3D(theta_periodic=True, **deformable_attn_kwargs())
    attn.init_weights()
    perturb(attn, std=1.0)  # large offsets, many samples cross the seam
    inputs = image_attention_inputs()
    base = run_image_attention(attn, inputs)
    for shift in (1.0, -1.0, 2.0):
        for a, b in zip(base, run_image_attention(attn, inputs, shift)):
            # (u + k) mod 1 in float32 moves a sample by up to ~1e-7 of the width; with the
            # large perturbed weights that shows up as ~2e-5 in the output. Today's
            # non-periodic sampling differs by more than 1e-3 (checked below).
            torch.testing.assert_close(b, a, rtol=0, atol=1e-4)
    attn.theta_periodic = False  # today's sampling is not periodic
    assert any(not torch.allclose(a, b, atol=1e-3) for a, b in zip(
        run_image_attention(attn, inputs), run_image_attention(attn, inputs, 1.0)))


def plane_attention(theta_periodic=True):
    kwargs = {k: v for k, v in hybrid_attn_cfg().items() if k != 'type'}
    torch.manual_seed(0)
    attn = cvha_mod.TPVCrossViewHybridAttention(theta_periodic=theta_periodic, **kwargs)
    attn.init_weights()
    perturb(attn, std=0.5)
    return attn.eval()


def run_plane_attention(attn, ref_2d, bs=2):
    query, _ = plane_queries(bs)
    ss, lsi = plane_shapes()
    query = [q.to(ref_2d.device) for q in query]
    return attn(query, None, reference_points=ref_2d, spatial_shapes=ss.to(ref_2d.device),
                level_start_index=lsi.to(ref_2d.device))


def test_plane_attention_is_periodic_in_theta():
    attn = plane_attention()
    ref_2d = plane_ref_2d(2)
    # theta is the y (row) coordinate on the thetar plane and x (column) on the ztheta plane
    shift = torch.zeros(3, 1, 2)
    shift[0, 0, 1] = 1.0
    shift[1, 0, 0] = 1.0
    base = run_plane_attention(attn, ref_2d)
    for k in (1.0, -1.0):
        for a, b in zip(base, run_plane_attention(attn, ref_2d + k * shift)):
            torch.testing.assert_close(b, a, rtol=0, atol=1e-5)
    attn.theta_periodic = False
    assert any(not torch.allclose(a, b, atol=1e-3) for a, b in zip(
        run_plane_attention(attn, ref_2d), run_plane_attention(attn, ref_2d + shift)))


def test_encoder_theta_periodic_wiring(cpu_encoders, monkeypatch):
    torch.manual_seed(0)
    off = build_encoder()
    torch.manual_seed(0)
    enc = build_encoder(theta_periodic=True)
    attn = [m for m in enc.modules()
            if isinstance(m, (ica_mod.TPVMSDeformableAttention3D, cvha_mod.TPVCrossViewHybridAttention))]
    assert len(attn) == 3 and all(m.theta_periodic for m in attn)  # 2 plane + 1 image attention
    # same parameters and stored buffers (keys and values) as the switch-off encoder
    sd_off, sd_on = off.state_dict(), enc.state_dict()
    assert list(sd_on) == list(sd_off)
    assert all(torch.equal(sd_on[k], sd_off[k]) for k in sd_off)
    assert 'ref_3d_rz_periodic' not in sd_on

    convs, refs = [], []
    conv_fn, sampling_fn = enc_mod.conv2d_theta_circular, enc.pano_point_sampling_cylinder
    monkeypatch.setattr(enc_mod, 'conv2d_theta_circular',
                        lambda conv, x, dim: (convs.append(dim), conv_fn(conv, x, dim))[1])
    monkeypatch.setattr(enc, 'pano_point_sampling_cylinder',
                        lambda ref_3d, *args: (refs.append(ref_3d), sampling_fn(ref_3d, *args))[1])
    enc.eval()
    off.eval()
    inputs = encoder_inputs(2) + (None,)
    out = enc(*inputs)
    assert convs == [2, 3]  # thetar: theta = rows; ztheta: theta = columns; rz untouched
    assert refs[2] is enc.ref_3d_rz_periodic
    assert all(torch.isfinite(o).all() for o in out)
    assert any(not torch.allclose(a, b) for a, b in zip(out, off(*inputs)))


def test_rz_pillars_cover_the_full_circle(cpu_encoders):
    enc = build_encoder(theta_periodic=True)
    p = PILLARS[2]
    periodic, today = enc.ref_3d_rz_periodic, enc.ref_3d_rz  # [1, P, r*z, (r, theta, z)]
    assert periodic.shape == today.shape
    theta = periodic[0, :, 0, 1]
    torch.testing.assert_close(theta, (torch.arange(p, dtype=theta.dtype) + 0.5) / p)
    assert torch.equal(periodic[..., 1], theta.view(1, p, 1).expand_as(periodic[..., 1]))
    assert torch.equal(periodic[..., 0], today[..., 0]) and torch.equal(periodic[..., 2], today[..., 2])
    # the gap across the seam: one pillar spacing now, ~2 x 0.5 rad of the circle today
    gap = lambda t: 1 - (t.max() - t.min())  # noqa: E731
    torch.testing.assert_close(gap(theta), torch.tensor(1.0 / p))
    torch.testing.assert_close(gap(today[0, :, 0, 1]), torch.tensor(2 * 0.5 / PC_RANGE[4]))
    # a checkpoint load cannot overwrite them (non-persistent), and ref_3d_rz still loads
    before = periodic.clone()
    sd = enc.state_dict()
    sd['ref_3d_rz'] = sd['ref_3d_rz'] + 1.0
    enc.load_state_dict(sd)
    assert torch.equal(enc.ref_3d_rz_periodic, before)
    assert torch.equal(enc.ref_3d_rz, sd['ref_3d_rz'])


def seam_points(dtype):
    # just left / right of the back seam (u -> w, u -> 0), close to it, and two away from it
    return torch.tensor([[[1e-3, 0.2, -2.0], [-1e-3, -0.3, -2.5], [0.02, 0.1, -1.5],
                          [1.0, 0.0, 1.0], [-0.5, 0.4, 2.0]]], dtype=dtype)


def retrieved_features(dec, xyz, img, depth):
    """The 36 * num_cams values the colour head sees (head replaced by an identity)."""
    dec.gaussian_to_color = nn.Identity()
    metas = [{'lidar2img': torch.eye(4, dtype=xyz.dtype)[None]}]
    return dec.get_panorama_color(xyz, img, depth, metas)[0]


def expected_features(xyz, img, depth):
    """The decoder's window sampling written out, with a circular (theta-periodic) image."""
    eps = 1e-5
    h, w = img.shape[-2:]
    p = xyz[0] / (1 + eps)  # lidar2img = I, divided by (w' + eps) as in the decoder
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    u = w * (torch.atan2(x, z + eps) + math.pi) / (2 * math.pi)
    v = h * (torch.atan2(y, torch.sqrt(x ** 2 + z ** 2 + eps)) + math.pi / 2) / math.pi
    dist = torch.sqrt(x ** 2 + y ** 2 + z ** 2 + eps)

    def to_px(a, n):  # normalize() by (n - 1), then grid_sample(align_corners=False)
        return a * n / (n - 1) - 0.5

    prior = circular_bilinear(depth[0, 0], to_px(u, w), to_px(v, h))[0]
    feats = []
    for i in range(3):  # window rows: v offset i - 1
        for j in range(3):  # window columns: u offset j - 1
            rgb = circular_bilinear(img[0, 0], to_px(u + j - 1, w), to_px(v + i - 1, h))
            feats += [rgb.T, (dist - prior)[:, None]]
    return torch.cat(feats, dim=-1)  # [N, 36]


def test_colour_window_wraps_across_the_seam():
    dtype = torch.float64
    g = torch.Generator().manual_seed(0)
    img = torch.rand(1, 1, 3, 16, 32, generator=g, dtype=dtype)
    depth = torch.rand(1, 1, 1, 16, 32, generator=g, dtype=dtype) * 3 + 1
    xyz = seam_points(dtype)
    want = expected_features(xyz, img, depth)

    periodic = retrieved_features(build_decoder(theta_periodic=True).double(), xyz, img, depth)
    torch.testing.assert_close(periodic[:, :36], want, rtol=0, atol=1e-9)
    assert torch.equal(periodic[:, 36:], torch.zeros_like(periodic[:, 36:]))

    today = retrieved_features(build_decoder().double(), xyz, img, depth)
    torch.testing.assert_close(today[3:, :36], want[3:], rtol=0, atol=1e-9)  # away from the seam: same
    for n in range(3):  # at the seam today reads the zero pad
        assert not torch.allclose(today[n, :36], want[n], atol=1e-3)


# ============================================================================ D6 cell_center_anchor

def controlled_decoder(raw_offset, **kwargs):
    """A decoder whose raw (r, theta, z) offset channels are the constant raw_offset."""
    torch.manual_seed(0)
    dec = build_decoder(**kwargs)
    with torch.no_grad():
        for p in dec.decoder.parameters():
            p.zero_()
        dec.gs_decoder.weight.zero_()
        bias = torch.zeros(dec.gpv, 14)
        bias[:, 0:3] = raw_offset
        dec.gs_decoder.bias.copy_(bias.flatten())
    return dec


def cylindrical(xyz):
    x, y, z = xyz.double().unbind(-1)  # x = -r sin(theta), z = -r cos(theta)
    return torch.sqrt(x ** 2 + z ** 2), torch.remainder(torch.atan2(-x, -z), 2 * math.pi), y


def check_cells(xyz, frac):
    """Gaussians at fraction `frac` of their (r, theta, z) cell; xyz [bs, R, THETA, Z, gpv, 3]."""
    r, theta, y = cylindrical(xyz)
    dr = (PC_RANGE[3] - PC_RANGE[0]) / TPV_R
    dt = 2 * math.pi / TPV_THETA
    dz = (PC_RANGE[5] - PC_RANGE[2]) / TPV_Z
    i = torch.arange(TPV_R, dtype=torch.float64).view(1, -1, 1, 1, 1)
    j = torch.arange(TPV_THETA, dtype=torch.float64).view(1, 1, -1, 1, 1)
    k = torch.arange(TPV_Z, dtype=torch.float64).view(1, 1, 1, -1, 1)
    torch.testing.assert_close(r, ((i + frac) * dr).expand_as(r), rtol=0, atol=1e-4)
    torch.testing.assert_close(y, ((k + frac) * dz + PC_RANGE[2]).expand_as(y), rtol=0, atol=1e-4)
    dtheta = torch.remainder(theta - (j + frac) * dt + math.pi, 2 * math.pi) - math.pi
    assert dtheta[r > 1e-3].abs().max() < 1e-4


@pytest.mark.parametrize('raw, frac', [(0.0, 0.5), (30.0, 1.0), (-30.0, 0.0)])
def test_cell_center_anchor_is_centred_and_symmetric(raw, frac):
    # tanh(0) = 0 -> the cell centre; tanh(+-30) = +-1 -> exactly half a cell either way
    dec = controlled_decoder(raw, cell_center_anchor=True)
    check_cells(dec(*decoder_inputs(2))[..., :3], frac)


def test_default_anchor_is_the_lower_corner():
    dec = controlled_decoder(0.0)
    check_cells(dec(*decoder_inputs(2))[..., :3], 0.0)


def test_cell_center_anchor_keeps_r_non_negative():
    dec = build_decoder(cell_center_anchor=True)
    shape = (1, TPV_R, TPV_THETA, TPV_Z, 1, 1)
    zeros, ones = torch.zeros(shape), torch.ones(shape)
    beyond = torch.full(shape, -2 * PC_RANGE[3])  # every anchor radius would go negative
    xyz, _ = dec.get_offsets_reference_points(TPV_THETA, TPV_R, TPV_Z, zeros, beyond, zeros,
                                              ones, ones, ones, PC_RANGE, device='cpu')
    assert (xyz[..., 0] == 0).all() and (xyz[..., 2] == 0).all()  # clamped onto the axis
    torch.testing.assert_close(xyz[..., 1], ((torch.arange(TPV_Z) + 0.5) * (PC_RANGE[5] - PC_RANGE[2]) / TPV_Z
                                             + PC_RANGE[2]).view(1, 1, 1, -1, 1).expand_as(xyz[..., 1]))


# ============================================================================ D7 rgb_retrieval

def check_copied_from(dec, concat_sd):
    """gaussian_to_color_vis holds a copy of a concat head given as gaussian_to_color.* tensors."""
    vis = dec.gaussian_to_color_vis
    assert torch.equal(vis.encoder[0].weight, concat_sd['0.weight'][:, :36])
    assert torch.equal(vis.encoder[0].bias, concat_sd['0.bias'])
    assert torch.equal(vis.head[0].weight, concat_sd['2.weight'])
    assert torch.equal(vis.head[0].bias, concat_sd['2.bias'])
    assert torch.equal(vis.head[2].weight, concat_sd['4.weight'])
    assert torch.equal(vis.head[2].bias, concat_sd['4.bias'])


def concat_part(sd):
    prefix = 'gaussian_to_color.'
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def test_visibility_softmax_adds_only_the_new_head():
    concat = build_decoder()
    vis = build_decoder(rgb_retrieval='visibility_softmax')
    sc, sv = concat.state_dict(), vis.state_dict()
    assert [k for k in sv if k in sc] == list(sc)
    assert all(sv[k].shape == sc[k].shape for k in sc)
    assert {k: tuple(t.shape) for k, t in sv.items() if k not in sc} == D7_NEW_PARAMS
    assert concat.gaussian_to_color[0].weight.shape == (128, 216)  # 36 x num_cams
    with pytest.raises(ValueError):
        build_decoder(rgb_retrieval='mean')


def test_visibility_head_starts_as_a_copy_of_the_concat_head():
    dec = build_decoder(rgb_retrieval='visibility_softmax')
    check_copied_from(dec, dec.gaussian_to_color.state_dict())
    assert torch.equal(dec.gaussian_to_color_vis.log_beta, torch.zeros(1))


def test_visibility_head_is_filled_from_the_loaded_checkpoint():
    torch.manual_seed(0)
    ckpt = {k: v.clone() for k, v in build_decoder().state_dict().items()}  # today's head
    # (a) a checkpoint without the new names: strict load passes, the pre-hook copies
    torch.manual_seed(1)
    dec = build_decoder(rgb_retrieval='visibility_softmax')
    result = dec.load_state_dict(ckpt)
    assert not result.missing_keys and not result.unexpected_keys
    check_copied_from(dec, concat_part(ckpt))
    # (b) the legacy resume merge: sd = model.state_dict(); sd.update(ckpt); load
    torch.manual_seed(2)
    dec = build_decoder(rgb_retrieval='visibility_softmax')
    sd = dec.state_dict()
    sd.update(ckpt)
    dec.load_state_dict(sd)
    check_copied_from(dec, concat_part(ckpt))
    # (c) a checkpoint trained with the switch keeps its own head
    perturb(dec.gaussian_to_color_vis)
    trained = {k: v.clone() for k, v in dec.state_dict().items()}
    torch.manual_seed(3)
    fresh = build_decoder(rgb_retrieval='visibility_softmax')
    fresh.load_state_dict(trained)
    assert all(torch.equal(fresh.state_dict()[k], trained[k]) for k in trained)
    # (d) a round trip through the module's own state dict changes nothing
    dec.load_state_dict(dec.state_dict())
    assert all(torch.equal(dec.state_dict()[k], trained[k]) for k in trained)


@pytest.mark.parametrize('theta_periodic', [False, True])
def test_visibility_head_reproduces_the_concat_head_for_one_view(theta_periodic):
    """One view: slot 0 is the only non-zero slot of the concat input and its softmax weight is 1."""
    torch.manual_seed(0)
    concat = build_decoder(theta_periodic=theta_periodic).double()
    torch.manual_seed(1)
    vis = build_decoder(theta_periodic=theta_periodic, rgb_retrieval='visibility_softmax').double()
    vis.load_state_dict(concat.state_dict())
    inputs = decoder_inputs(1, batch=2, dtype=torch.float64)
    torch.testing.assert_close(vis(*inputs), concat(*inputs), rtol=0, atol=1e-12)


def test_visibility_head_equals_the_concat_head_when_only_view_0_sees_the_point():
    """Two views, view 1 out of sight (its slot is zero in the concat input, weight 0 here)."""
    torch.manual_seed(0)
    dec = build_decoder(rgb_retrieval='visibility_softmax').double()
    g = torch.Generator().manual_seed(5)
    bs, n = 2, 11
    x0 = torch.randn(bs, n, 36, generator=g, dtype=torch.float64)
    per_view = torch.stack([x0, torch.zeros_like(x0)], dim=2)
    visibility = torch.stack([x0[..., 3], torch.zeros_like(x0[..., 3])], dim=-1)
    valid = torch.tensor([True, False]).expand(bs, n, 2)
    want = dec.gaussian_to_color(F.pad(x0, (0, 36 * 5)))
    torch.testing.assert_close(dec.gaussian_to_color_vis(per_view, visibility, valid), want,
                               rtol=0, atol=1e-12)


def test_visibility_softmax_weighting():
    torch.manual_seed(0)
    head = dec_mod.VisibilitySoftmaxColor().double()
    g = torch.Generator().manual_seed(6)
    bs, n = 2, 5
    a = torch.randn(bs, n, 36, generator=g, dtype=torch.float64)
    b = torch.randn(bs, n, 36, generator=g, dtype=torch.float64)
    both = torch.stack([a, b], dim=2)
    alone = head(a[:, :, None], torch.zeros(bs, n, 1, dtype=torch.float64),
                 torch.ones(bs, n, 1, dtype=torch.bool))
    # a view that does not see the point gets no weight
    invalid_b = torch.tensor([True, False]).expand(bs, n, 2)
    torch.testing.assert_close(head(both, torch.zeros(bs, n, 2, dtype=torch.float64), invalid_b),
                               alone, rtol=0, atol=1e-12)
    # the view whose depth prior agrees with the Gaussian dominates
    vis = torch.tensor([0.0, 30.0], dtype=torch.float64).expand(bs, n, 2)
    torch.testing.assert_close(head(both, vis, torch.ones(bs, n, 2, dtype=torch.bool)), alone,
                               rtol=0, atol=1e-9)
    # any number of views, independent of their order, finite when no view sees the point
    feats = torch.randn(bs, n, 8, 36, generator=g, dtype=torch.float64)
    vis = torch.randn(bs, n, 8, generator=g, dtype=torch.float64)
    valid = torch.rand(bs, n, 8, generator=g) > 0.3
    out = head(feats, vis, valid)
    assert out.shape == (bs, n, 3)
    perm = torch.randperm(8, generator=g)
    torch.testing.assert_close(head(feats[:, :, perm], vis[:, :, perm], valid[:, :, perm]), out,
                               rtol=0, atol=1e-12)
    assert torch.isfinite(head(feats, vis, torch.zeros_like(valid))).all()


def test_visibility_head_trains_and_the_concat_head_is_frozen():
    torch.manual_seed(0)
    dec = build_decoder(rgb_retrieval='visibility_softmax')
    assert not any(p.requires_grad for p in dec.gaussian_to_color.parameters())
    out = dec(*decoder_inputs(2))
    out[..., 3:6].sum().backward()  # rgb channels
    assert all(p.grad is None for p in dec.gaussian_to_color.parameters())
    for name, p in dec.gaussian_to_color_vis.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


# ============================================================================ config plumbing

def test_switches_reach_the_volume_modules_from_the_config(cpu_encoders):
    from mmengine.config import Config
    from mmengine.registry import MODELS
    from tools import switches

    path = os.path.join(REPO_ROOT, 'configs', 'OmniScene', 'omni_gs_160x320_mp3d_cylinder_all_256.py')
    default = MODELS.build(Config.fromfile(path).model.volume_gs)
    cfg = Config.fromfile(path)
    switches.apply(cfg, ['theta_periodic=true', 'cell_center_anchor=true',
                         'rgb_retrieval=visibility_softmax'])
    vol = MODELS.build(cfg.model.volume_gs)

    assert not default.encoder.theta_periodic and not default.gs_decoder.theta_periodic
    assert not default.gs_decoder.cell_center_anchor and default.gs_decoder.rgb_retrieval == 'concat'
    enc, dec = vol.encoder, vol.gs_decoder
    assert enc.theta_periodic and dec.theta_periodic and dec.cell_center_anchor
    assert dec.rgb_retrieval == 'visibility_softmax'
    attn = [m for m in enc.modules()
            if isinstance(m, (ica_mod.TPVMSDeformableAttention3D, cvha_mod.TPVCrossViewHybridAttention))]
    assert len(attn) == 5 and all(m.theta_periodic for m in attn)  # 3 plane + 2 image attentions
    sd, sd0 = vol.state_dict(), default.state_dict()
    assert [k for k in sd if k in sd0] == list(sd0)
    assert {k: tuple(t.shape) for k, t in sd.items() if k not in sd0} == \
        {'gs_decoder.' + k: s for k, s in D7_NEW_PARAMS.items()}


# ============================================================================ CUDA op

@pytest.mark.gpu
def test_periodic_attention_cuda_op_matches_pytorch_path():
    attn = plane_attention()
    ref_2d = plane_ref_2d(2)
    ref_2d[:, :, :2] += torch.linspace(-0.4, 0.4, ref_2d.shape[1]).view(1, -1, 1, 1, 1)
    cpu = run_plane_attention(attn, ref_2d)
    gpu = run_plane_attention(attn.cuda(), ref_2d.cuda())
    for a, b in zip(cpu, gpu):
        torch.testing.assert_close(b.cpu(), a, rtol=1e-4, atol=1e-4)

    torch.manual_seed(0)
    attn = ica_mod.TPVMSDeformableAttention3D(theta_periodic=True, **deformable_attn_kwargs())
    attn.init_weights()
    perturb(attn, std=1.0)
    inputs = image_attention_inputs()
    cpu = run_image_attention(attn, inputs)
    gpu = run_image_attention(attn.cuda(), to_device(inputs, 'cuda'))
    for a, b in zip(cpu, gpu):
        torch.testing.assert_close(b.cpu(), a, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
def test_theta_periodic_encoder_runs_on_cuda():
    torch.manual_seed(0)
    enc = build_encoder(theta_periodic=True).cuda().eval()
    out = enc(*to_device(encoder_inputs(2) + (None,), 'cuda'))
    assert all(torch.isfinite(o).all() for o in out)
