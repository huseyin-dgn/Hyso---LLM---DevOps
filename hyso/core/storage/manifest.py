# Run manifest veri yapısı ve dosya işlemleri

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Mapping
from datetime import datetime
import json
import platform
import sys


@dataclass(slots=True)
class Manifest:
    run_id: str
    created_at: str
    model: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        run_id: str,
        model: Mapping[str, Any] | None = None,
        data: Mapping[str, Any] | None = None,
        training: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        include_environment: bool = True,
    ) -> "Manifest":
        created_at = datetime.now().isoformat(timespec="seconds")
        env: Dict[str, Any] = {}
        if include_environment:
            env = {
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "interpreter": sys.executable,
            }
        return cls(
            run_id=run_id,
            created_at=created_at,
            model=dict(model) if model is not None else {},
            data=dict(data) if data is not None else {},
            training=dict(training) if training is not None else {},
            environment=env,
            extra=dict(extra) if extra is not None else {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Manifest":
        run_id = str(payload.get("run_id", "unknown"))
        created_at = str(payload.get("created_at", datetime.now().isoformat(timespec="seconds")))
        model = dict(payload.get("model", {}))
        data = dict(payload.get("data", {}))
        training = dict(payload.get("training", {}))
        environment = dict(payload.get("environment", {}))
        extra = dict(payload.get("extra", {}))
        return cls(
            run_id=run_id,
            created_at=created_at,
            model=model,
            data=data,
            training=training,
            environment=environment,
            extra=extra,
        )

def save_manifest(path: str | Path, manifest: Manifest | Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(manifest, Manifest):
        payload = manifest.to_dict()
    else:
        payload = dict(manifest)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_manifest(path: str | Path) -> Manifest:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Manifest not found: {target}")
    with target.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return Manifest.from_dict(payload)
    raise ValueError(f"Manifest file is not a JSON object: {target}")
