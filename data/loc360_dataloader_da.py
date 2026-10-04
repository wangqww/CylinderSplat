"""The 360Loc evaluation loader with the paper's PCC reference (evaluate.py row loc360_double_256_da).

load_360Loc_data (data/loc360_dataloader_double_all_512.py) with Dataset360Loc(pcc_reference=
'depth_anywhere'): outputs['depth'] is the Depth Anywhere pseudo-GT instead of the UniK3D prior the
model receives as input. Same split, samples, seeds and DataLoader keywords. A separate module, so
the released loader file keeps its single DataLoader(...) call (tests/test_entries_table.py).
"""

from torch.utils.data import DataLoader

from data.loc360_dataloader_double_all_512 import Dataset360Loc, get_generator, worker_init_fn

# load_360Loc_data's per-stage seed and persistent_workers
_STAGES = {'val': (3456, True), 'test': (2345, False)}


def load_360Loc_data_da(batch_size, stage='val'):
    if stage == 'train':
        raise ValueError("load_360Loc_data_da is an evaluation loader")
    seed, persistent_workers = _STAGES.get(stage, (6789, True))
    return DataLoader(
        Dataset360Loc(stage=stage, pcc_reference="depth_anywhere"),
        batch_size=batch_size,
        num_workers=1,
        generator=get_generator(seed),
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        shuffle=False
    )
