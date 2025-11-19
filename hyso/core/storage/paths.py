# Run path yönetimi için yardımcılar

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional
from datetime import datetime
import uuid


@dataclass(slots=True)
class RunPaths:
    root_dir: Path
    run_id: str
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    artifacts_dir: Path
    configs_dir: Path
    manifest_path: Path
    metrics_path: Path

    def to_dict(self) -> dict[str, str]:
        return {f.name: str(getattr(self, f.name)) for f in fields(self)}


@dataclass(slots=True)
class RunPathFactory:
    root_dir: Path
    prefix: str = "run"
    time_format: str = "%Y%m%d-%H%M%S"
    use_uuid: bool = True

    @classmethod
    def from_root(
        cls,
        root_dir: str | Path = "runs",
        prefix: str = "run",
        time_format: str = "%Y%m%d-%H%M%S",
        use_uuid: bool = True,
    ) -> "RunPathFactory":
        root = Path(root_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return cls(root_dir=root, prefix=prefix, time_format=time_format, use_uuid=use_uuid)

    def _generate_run_id(self) -> str:
        ts = datetime.now().strftime(self.time_format)
        if self.use_uuid:
            uid = uuid.uuid4().hex[:8]
            return f"{self.prefix}_{ts}_{uid}"
        return f"{self.prefix}_{ts}"

    def create(self, run_id: Optional[str] = None, create_dirs: bool = True) -> RunPaths:
        if run_id is None:
            run_id = self._generate_run_id()

        run_dir = self.root_dir / run_id
        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"
        artifacts_dir = run_dir / "artifacts"
        configs_dir = run_dir / "configs"
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.csv"

        if create_dirs:
            for p in (run_dir, checkpoints_dir, logs_dir, artifacts_dir, configs_dir):
                p.mkdir(parents=True, exist_ok=True)

        return RunPaths(
            root_dir=self.root_dir,
            run_id=run_id,
            run_dir=run_dir,
            checkpoints_dir=checkpoints_dir,
            logs_dir=logs_dir,
            artifacts_dir=artifacts_dir,
            configs_dir=configs_dir,
            manifest_path=manifest_path,
            metrics_path=metrics_path,
        )
