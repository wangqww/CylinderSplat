"""Dataset roots of the loaders.

Each root defaults to the author's machine; set the environment variable to an absolute path to use your
copy (runs change the working directory). scripts/reproduce.sh sets them to its download directory.
"""

import os
from pathlib import Path

# PanoGRF renders (Matterport3D, Replica, Residential): the png_render_* folders
PANO_GRF_ROOT = Path(os.environ.get("CYLINDERSPLAT_PANO_GRF", "/data/qiwei/nips25/pano_grf"))
# 360Loc: the location folders (atrium, concourse, hall, piatrium)
LOC360_ROOT = Path(os.environ.get("CYLINDERSPLAT_360LOC", "/data/qiwei/nips25/360Loc"))
