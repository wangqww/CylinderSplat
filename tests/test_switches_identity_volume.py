"""D5 / D6 / D7 switches OFF == the released code (f7b20b9) for the cylindrical volume branch.

Every live class is compared with its frozen verbatim copy in
tests/legacy_ref/volume_ref.py: seeded construction (same RNG consumption, so
identical parameter / buffer names, order, values and RNG state afterwards) and
forward outputs and gradients, bitwise, on small synthetic tensors. The CPU tests
run mmcv's pure-PyTorch deformable attention; the `gpu` tests repeat the encoder
check through the CUDA op. test_legacy_ref_blocks_are_verbatim re-checks the copies
against `git show f7b20b9:<path>` when git and the commit are available.

The switches are off by default and when passed explicitly as
theta_periodic=False / cell_center_anchor=False / rgb_retrieval='concat'.
"""

import functools
import math
import re

import pytest
import torch

from model.volume import cross_view_hybrid_attention as cvha_mod
from model.volume import image_cross_attention as ica_mod
from model.volume import tpvformer_encoder_cylinder as enc_mod
from model.volume import volume_gs_decoder_cylinder as dec_mod
from tests.legacy_ref import verbatim
from tests.legacy_ref import volume_ref as ref

TPV_THETA, TPV_R, TPV_Z = 8, 4, 4
DIM, HEADS = 16, 4
PC_RANGE = [0.0, 0.0, -7.0, 10.0, 6.28, 3.0]  # r, theta, z as in the configs
PILLARS = [4, 2, 8]  # image cross-attention pillar points: thetar, ztheta, rz
NUM_POINTS = [8, 4, 16]  # per plane; num_points / multiplier equal across planes
CROSS_VIEW_PILLARS = [2, 2, 2]
FEAT_H, FEAT_W = 8, 16  # backbone feature map read by the image cross-attention
RGB_H, RGB_W = 16, 32  # colour / depth images read by the decoder

ENCODER_OFF = [{}, {'theta_periodic': False}]
DECODER_OFF = [{}, {'theta_periodic': False, 'cell_center_anchor': False, 'rgb_retrieval': 'concat'}]


# ----------------------------------------------------------------------------- builders

def hybrid_attn_cfg(legacy=False):
    return dict(
        type=ref.LEGACY_HYBRID_ATTENTION if legacy else 'TPVCrossViewHybridAttention',
        tpv_h=TPV_THETA, tpv_w=TPV_R, tpv_z=TPV_Z, num_anchors=2, embed_dims=DIM,
        num_heads=HEADS, num_points=4, init_mode=0, dropout=0.1)


def deformable_attn_kwargs():
    return dict(embed_dims=DIM, num_heads=HEADS, num_points=NUM_POINTS, num_z_anchors=PILLARS,
                num_levels=1, floor_sampling_offset=False, tpv_h=TPV_THETA, tpv_w=TPV_R, tpv_z=TPV_Z)


def encoder_cfg(legacy=False, **kwargs):
    """A small copy of the all_256 encoder config (same layer layout, tiny sizes)."""
    cross_attn = dict(
        type='TPVImageCrossAttention', pc_range=PC_RANGE, dropout=0.1,
        deformable_attention=dict(
            type=ref.LEGACY_DEFORMABLE_ATTENTION if legacy else 'TPVMSDeformableAttention3D',
            **deformable_attn_kwargs()),
        embed_dims=DIM, tpv_h=TPV_THETA, tpv_w=TPV_R, tpv_z=TPV_Z)
    self_cross_layer = dict(
        type='TPVFormerLayer', attn_cfgs=[hybrid_attn_cfg(legacy), cross_attn],
        feedforward_channels=2 * DIM, ffn_dropout=0.1,
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))
    self_layer = dict(
        type='TPVFormerLayer', attn_cfgs=[hybrid_attn_cfg(legacy)],
        feedforward_channels=2 * DIM, ffn_dropout=0.1,
        operation_order=('self_attn', 'norm', 'ffn', 'norm'))
    cfg = dict(
        tpv_theta=TPV_THETA, tpv_r=TPV_R, tpv_z=TPV_Z, num_feature_levels=1, num_layers=2,
        pc_range=PC_RANGE, num_points_in_pillar=PILLARS,
        num_points_in_pillar_cross_view=CROSS_VIEW_PILLARS, return_intermediate=False,
        transformerlayers=[self_cross_layer, self_layer], embed_dims=DIM,
        positional_encoding=dict(type='TPVFormerPositionalEncoding', num_feats=[4, 6, 6],
                                 h=TPV_THETA, w=TPV_R, z=TPV_Z))
    cfg.update(kwargs)
    return cfg


def build_encoder(legacy=False, **kwargs):
    cls = ref.TPVFormerEncoderCylinder if legacy else enc_mod.TPVFormerEncoderCylinder
    return cls(**encoder_cfg(legacy, **kwargs))


def decoder_kwargs(**kwargs):
    cfg = dict(tpv_theta=TPV_THETA, tpv_r=TPV_R, tpv_z=TPV_Z, pc_range=PC_RANGE, gs_dim=14,
               in_dims=DIM, hidden_dims=2 * DIM, out_dims=DIM, scale_theta=1, scale_r=1,
               scale_z=1, gpv=2, offset_max=[0.5, 0.5, 0.5], scale_max=[0.5, 0.5, 0.5])
    cfg.update(kwargs)
    return cfg


def build_decoder(legacy=False, **kwargs):
    cls = ref.VolumeGaussianDecoderCylinder if legacy else dec_mod.VolumeGaussianDecoderCylinder
    return cls(**decoder_kwargs(**kwargs))


@pytest.fixture
def cpu_encoders(monkeypatch):
    """The encoders build their reference points with device='cuda'; build them on the CPU."""
    for cls in (enc_mod.TPVFormerEncoderCylinder, ref.TPVFormerEncoderCylinder):
        monkeypatch.setattr(cls, 'get_reference_points',
                            staticmethod(functools.partial(cls.get_reference_points, device='cpu')))


# ----------------------------------------------------------------------------- inputs

def _rot(yaw, tilt):
    cy, sy, ct, st = math.cos(yaw), math.sin(yaw), math.cos(tilt), math.sin(tilt)
    ry = torch.tensor([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
    rx = torch.tensor([[1., 0., 0.], [0., ct, -st], [0., st, ct]])
    return ry @ rx


def make_metas(num_views, batch=1, seed=0, dtype=torch.float32):
    """(b v) camera metas as OmniGaussianCylinderAll.get_data builds them: for every reference
    view i, lidar2img_k = w2i_k @ inv(w2i_i); a single view keeps its absolute (non-identity) w2i."""
    g = torch.Generator().manual_seed(seed)
    metas = []
    for _ in range(batch):
        w2i = []
        for _ in range(num_views):
            m = torch.eye(4)
            m[:3, :3] = _rot(float(torch.rand(1, generator=g)) * 2 * math.pi,
                             float(torch.randn(1, generator=g)) * 0.1)
            m[:3, 3] = torch.randn(3, generator=g) * 0.5
            w2i.append(m)
        w2i = torch.stack(w2i).to(dtype)
        if num_views < 2:
            metas.append({'lidar2img': w2i})
            continue
        for i in range(num_views):
            metas.append({'lidar2img': w2i @ w2i[i].inverse()})
    return metas


def encoder_inputs(num_views, batch=1, seed=0, with_project=True):
    g = torch.Generator().manual_seed(seed)
    bs = batch * num_views
    feats = [torch.randn(bs, num_views, DIM, FEAT_H, FEAT_W, generator=g)]
    if with_project:
        project = [torch.randn(bs, DIM, TPV_THETA, TPV_R, generator=g),
                   torch.randn(bs, DIM, TPV_Z, TPV_THETA, generator=g),
                   torch.randn(bs, DIM, TPV_R, TPV_Z, generator=g)]
    else:
        project = [None, None, None]
    return feats, project, make_metas(num_views, batch, seed)


def decoder_inputs(num_views, batch=1, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    bs = batch * num_views
    tpv_list = [torch.randn(bs, TPV_THETA * TPV_R, DIM, generator=g),
                torch.randn(bs, TPV_Z * TPV_THETA, DIM, generator=g),
                torch.randn(bs, TPV_R * TPV_Z, DIM, generator=g)]
    img_color = torch.rand(bs, num_views, 3, RGB_H, RGB_W, generator=g)
    img_depth = torch.rand(bs, num_views, 1, RGB_H, RGB_W, generator=g) * 6 + 0.5
    tpv_list = [t.to(dtype) for t in tpv_list]
    return tpv_list, img_color.to(dtype), img_depth.to(dtype), make_metas(num_views, batch, seed, dtype)


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


# ----------------------------------------------------------------------------- checks

def assert_same_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert sa[k].dtype == sb[k].dtype and sa[k].shape == sb[k].shape, k
        assert torch.equal(sa[k], sb[k]), k
    assert [n for n, _ in a.named_parameters()] == [n for n, _ in b.named_parameters()]
    assert [p.requires_grad for p in a.parameters()] == [p.requires_grad for p in b.parameters()]


def assert_same_outputs(xs, ys):
    xs = list(xs) if isinstance(xs, (list, tuple)) else [xs]
    ys = list(ys) if isinstance(ys, (list, tuple)) else [ys]
    assert len(xs) == len(ys)
    for x, y in zip(xs, ys):
        assert x.shape == y.shape and x.dtype == y.dtype
        assert torch.equal(x, y)


def assert_same_grads(a, b):
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert (pa.grad is None) == (pb.grad is None), na
        if pa.grad is not None:
            assert torch.equal(pa.grad, pb.grad), na


def seeded_pair(build, legacy_kwargs=None, live_kwargs=None, seed=0):
    """Build legacy then live from the same seed; check the RNG state afterwards is the same."""
    torch.manual_seed(seed)
    legacy = build(legacy=True, **(legacy_kwargs or {}))
    rng_legacy = torch.random.get_rng_state()
    torch.manual_seed(seed)
    live = build(**(live_kwargs or {}))
    assert torch.equal(rng_legacy, torch.random.get_rng_state())
    return legacy, live


def run_pair(legacy, live, inputs, train, seed=1):
    """Forward (and backward) both modules from the same RNG state (dropout)."""
    legacy.train(train)
    live.train(train)
    outs = []
    for module in (legacy, live):
        torch.manual_seed(seed)
        out = module(*inputs)
        if train:
            total = sum(o.sum() for o in (out if isinstance(out, (list, tuple)) else [out]))
            total.backward()
        outs.append(out)
    return outs


# ----------------------------------------------------------------------------- frozen copies are verbatim

_VERBATIM = re.compile(r"# ---- verbatim: (\S+) lines (\d+)-(\d+) at f7b20b9 \(decorator line (\d+) dropped\) ----")
_REGISTRY = "# ---- registry names for building a legacy encoder from a config (not part of the copies) ----"


def test_legacy_ref_blocks_are_verbatim():
    """volume_ref.py holds the four released classes, each without only its register decorator,
    and nothing but blank lines between them."""
    lines = verbatim.source_lines("volume_ref")
    marks = [(i, m) for i, line in enumerate(lines) if (m := _VERBATIM.fullmatch(line))]
    assert len(marks) == 4
    covered = set()
    for i, m in marks:
        path, a, b, dropped = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        end = verbatim.assert_verbatim(lines, i + 1, path, a, b)
        assert dropped == a - 1 and verbatim.git_show(path)[dropped - 1] == "@MODELS.register_module()", path
        covered.update(range(i, end))
    registry = lines.index(_REGISTRY)
    assert registry > max(covered)
    assert verbatim.stray_lines(lines, covered, marks[0][0], registry) == []


# ----------------------------------------------------------------------------- attention modules

def plane_queries(bs, seed=0):
    g = torch.Generator().manual_seed(seed)
    lens = [TPV_THETA * TPV_R, TPV_Z * TPV_THETA, TPV_R * TPV_Z]
    return [torch.randn(bs, n, DIM, generator=g) for n in lens], lens


def plane_ref_2d(bs, seed=0):
    ref_2d = enc_mod.TPVFormerEncoderCylinder.get_cross_view_ref_points(
        TPV_THETA, TPV_R, TPV_Z, CROSS_VIEW_PILLARS)
    return ref_2d.unsqueeze(0).expand(bs, -1, -1, -1, -1).clone()


def plane_shapes():
    ss = torch.tensor([[TPV_THETA, TPV_R], [TPV_Z, TPV_THETA], [TPV_R, TPV_Z]])
    lsi = torch.tensor([0, TPV_THETA * TPV_R, TPV_THETA * TPV_R + TPV_Z * TPV_THETA])
    return ss, lsi


def perturb(module, seed=2, std=0.2):
    """Move the weights off their structured init so every code path is exercised."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn(p.shape, generator=g) * std)


def test_hybrid_attention_off_matches_legacy():
    kwargs = {k: v for k, v in hybrid_attn_cfg().items() if k != 'type'}
    torch.manual_seed(0)
    legacy = ref.TPVCrossViewHybridAttention(**kwargs)
    legacy.init_weights()
    torch.manual_seed(0)
    live = cvha_mod.TPVCrossViewHybridAttention(theta_periodic=False, **kwargs)
    live.init_weights()
    assert_same_state(legacy, live)
    perturb(live)
    legacy.load_state_dict(live.state_dict())

    bs = 2
    query, _ = plane_queries(bs)
    pos, _ = plane_queries(bs, seed=1)
    ss, lsi = plane_shapes()
    ref_2d = plane_ref_2d(bs)
    for train in (False, True):
        outs = []
        for module in (legacy, live):
            module.train(train)
            module.zero_grad()
            torch.manual_seed(3)
            out = module(query, None, query_pos=pos, reference_points=ref_2d,
                         spatial_shapes=ss, level_start_index=lsi)
            sum(o.sum() for o in out).backward()
            outs.append(out)
        assert_same_outputs(*outs)
        assert_same_grads(legacy, live)


def test_image_deformable_attention_off_matches_legacy():
    torch.manual_seed(0)
    legacy = ref.TPVMSDeformableAttention3D(**deformable_attn_kwargs())
    legacy.init_weights()
    torch.manual_seed(0)
    live = ica_mod.TPVMSDeformableAttention3D(theta_periodic=False, **deformable_attn_kwargs())
    live.init_weights()
    assert_same_state(legacy, live)
    perturb(live, std=1.0)  # large offsets: many samples leave [0, 1] across the u seam
    legacy.load_state_dict(live.state_dict())

    g = torch.Generator().manual_seed(4)
    bs = 3
    lens = [5, 7, 3]
    query = [torch.randn(bs, n, DIM, generator=g) for n in lens]
    refs = [torch.rand(bs, n, p, 2, generator=g) for n, p in zip(lens, PILLARS)]
    value = torch.randn(bs, FEAT_H * FEAT_W, DIM, generator=g)
    ss = torch.tensor([[FEAT_H, FEAT_W]])
    lsi = torch.tensor([0])
    outs = []
    for module in (legacy, live):
        module.zero_grad()
        out = module(query, value=value, reference_points=refs, spatial_shapes=ss,
                     level_start_index=lsi)
        sum(o.sum() for o in out).backward()
        outs.append(out)
    assert_same_outputs(*outs)
    assert_same_grads(legacy, live)


# ----------------------------------------------------------------------------- encoder

@pytest.mark.parametrize('off_kwargs', ENCODER_OFF)
def test_encoder_off_construction_matches_legacy(cpu_encoders, off_kwargs):
    legacy, live = seeded_pair(build_encoder, live_kwargs=off_kwargs)
    assert_same_state(legacy, live)
    assert not live.theta_periodic
    assert not hasattr(live, 'ref_3d_rz_periodic')
    for m in live.modules():
        if isinstance(m, (ica_mod.TPVMSDeformableAttention3D, cvha_mod.TPVCrossViewHybridAttention)):
            assert m.theta_periodic is False


@pytest.mark.parametrize('off_kwargs', ENCODER_OFF)
@pytest.mark.parametrize('num_views', [1, 2])
@pytest.mark.parametrize('train', [False, True])
@pytest.mark.parametrize('with_project', [True, False])
def test_encoder_off_forward_matches_legacy(cpu_encoders, off_kwargs, num_views, train, with_project):
    legacy, live = seeded_pair(build_encoder, live_kwargs=off_kwargs)
    feats, project, metas = encoder_inputs(num_views, batch=1 if num_views == 2 else 2,
                                           with_project=with_project)
    outs = run_pair(legacy, live, (feats, project, metas, None), train)
    assert_same_outputs(*outs)
    if train:
        assert_same_grads(legacy, live)


@pytest.mark.gpu
@pytest.mark.parametrize('num_views', [1, 2])
def test_encoder_off_forward_matches_legacy_cuda(num_views):
    """Same as above through mmcv's CUDA deformable-attention op."""
    legacy, live = seeded_pair(build_encoder)
    legacy, live = legacy.cuda(), live.cuda()
    inputs = to_device(encoder_inputs(num_views, batch=1 if num_views == 2 else 2) + (None,), 'cuda')
    outs = run_pair(legacy, live, inputs, train=False)
    assert_same_outputs(*outs)


# ----------------------------------------------------------------------------- decoder

@pytest.mark.parametrize('off_kwargs', DECODER_OFF)
def test_decoder_off_construction_matches_legacy(off_kwargs):
    legacy, live = seeded_pair(build_decoder, live_kwargs=off_kwargs)
    assert_same_state(legacy, live)  # the 'concat' state-dict keys are exactly today's
    assert not hasattr(live, 'gaussian_to_color_vis')
    assert len(live._load_state_dict_pre_hooks) == len(legacy._load_state_dict_pre_hooks)


@pytest.mark.parametrize('off_kwargs', DECODER_OFF)
@pytest.mark.parametrize('num_views', [1, 2, 3])
@pytest.mark.parametrize('train', [False, True])
def test_decoder_off_forward_matches_legacy(off_kwargs, num_views, train):
    legacy, live = seeded_pair(build_decoder, live_kwargs=off_kwargs)
    inputs = decoder_inputs(num_views, batch=2)
    outs = run_pair(legacy, live, inputs, train)
    assert_same_outputs(*outs)
    if train:
        assert_same_grads(legacy, live)
