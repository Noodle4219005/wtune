#!/usr/bin/env bash
# bin/fetch-natural-earth.sh -- pre-download the Natural Earth shapefiles cartopy draws
# coastlines/borders from, so `wtune post plot` works on compute nodes without internet.
# Run once on a node that has internet (a login node), then export WTUNE_CARTOPY_DATA.
#
#   bin/fetch-natural-earth.sh [DEST]    default DEST: envs/cartopy-data
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${1:-${WTUNE_CARTOPY_DATA:-$repo/envs/cartopy-data}}"
mkdir -p "$dest"
pixi_bin="$(command -v pixi || echo "$HOME/.pixi/bin/pixi")"
"$pixi_bin" run --manifest-path "$repo/envs/wtune/pixi.toml" python - "$dest" <<'EOF'
import sys
import cartopy
from cartopy.io import shapereader
cartopy.config["data_dir"] = sys.argv[1]
for res in ("110m", "50m", "10m"):
    for cat, name in (("physical", "coastline"), ("physical", "land"), ("physical", "lakes"),
                      ("cultural", "admin_0_boundary_lines_land"),
                      ("cultural", "admin_1_states_provinces_lakes")):
        print(shapereader.natural_earth(resolution=res, category=cat, name=name))
EOF
echo "export WTUNE_CARTOPY_DATA=$dest"
