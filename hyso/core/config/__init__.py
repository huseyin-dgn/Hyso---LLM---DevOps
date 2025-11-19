# Config paketinin halka açık arayüzü

from __future__ import annotations

from .loader import (
    Config,
    load_config,
    merge_config,
    parse_overrides,
    load_config_with_overrides,
    save_config,
)

__all__ = [
    "Config",
    "load_config",
    "merge_config",
    "parse_overrides",
    "load_config_with_overrides",
    "save_config",
]

# -------------
#python train.py training.lr=0.0001 model.layers=12 training.epochs=20
# -------------

# import sys
# from hyso.core.config import load_config_with_overrides
# from hyso.core.storage import RunPathFactory, save_manifest, Manifest

# factory = RunPathFactory.from_root("runs")
# run_paths = factory.create()

# cfg = load_config_with_overrides("configs/base.yaml", override_pairs=sys.argv[1:])

# manifest = Manifest.new(
#     run_id=run_paths.run_id,
#     model=cfg.model,
#     training=cfg.training,
# )
# save_manifest(run_paths.manifest_path, manifest)

# lr = cfg.training["lr"]
# batch_size = cfg.training.batch_size



