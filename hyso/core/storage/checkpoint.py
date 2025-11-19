# Checkpoint yönetimi

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Literal

import torch


@dataclass(slots=True)
class CheckpointConfig:
    directory: Path
    prefix: str = "ckpt"
    max_to_keep: int = 5
    best_metric_name: Optional[str] = None
    best_mode: Literal["min", "max"] = "min"
    keep_best: bool = True

    @classmethod
    def from_dir(
        cls,
        directory: str | Path,
        prefix: str = "ckpt",
        max_to_keep: int = 5,
        best_metric_name: Optional[str] = None,
        best_mode: Literal["min", "max"] = "min",
        keep_best: bool = True,
    ) -> "CheckpointConfig":
        d = Path(directory).expanduser().resolve()
        d.mkdir(parents=True, exist_ok=True)
        return cls(
            directory=d,
            prefix=prefix,
            max_to_keep=max_to_keep,
            best_metric_name=best_metric_name,
            best_mode=best_mode,
            keep_best=keep_best,
        )


class CheckpointManager:
    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config
        self._best_value: Optional[float] = None

    @property
    def directory(self) -> Path:
        return self.config.directory

    def _build_payload(
        self,
        epoch: int,
        model: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if hasattr(model, "state_dict"):
            model_state = model.state_dict()
        else:
            model_state = model
        if optimizer is not None and hasattr(optimizer, "state_dict"):
            opt_state = optimizer.state_dict()
        else:
            opt_state = None
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            sched_state = scheduler.state_dict()
        else:
            sched_state = None
        payload: Dict[str, Any] = {
            "epoch": int(epoch),
            "model": model_state,
            "optimizer": opt_state,
            "scheduler": sched_state,
            "metrics": dict(metrics) if metrics is not None else {},
            "extra": dict(extra) if extra is not None else {},
        }
        return payload

    def _checkpoint_path(self, epoch: int) -> Path:
        name = f"{self.config.prefix}_epoch{epoch:04d}.pt"
        return self.directory / name

    def _latest_path(self) -> Path:
        return self.directory / f"{self.config.prefix}_latest.pt"

    def _best_path(self) -> Path:
        return self.directory / f"{self.config.prefix}_best.pt"

    def _update_best(self, metrics: Mapping[str, Any]) -> bool:
        if not self.config.keep_best or self.config.best_metric_name is None:
            return False
        if self.config.best_metric_name not in metrics:
            return False
        value_raw = metrics[self.config.best_metric_name]
        try:
            value = float(value_raw)
        except (TypeError, ValueError):
            return False
        if self._best_value is None:
            self._best_value = value
            return True
        if self.config.best_mode == "min":
            if value < self._best_value:
                self._best_value = value
                return True
        else:
            if value > self._best_value:
                self._best_value = value
                return True
        return False

    def _prune_old_checkpoints(self) -> None:
        if self.config.max_to_keep <= 0:
            return
        pattern = f"{self.config.prefix}_epoch*.pt"
        files = sorted(self.directory.glob(pattern))
        excess = len(files) - self.config.max_to_keep
        if excess <= 0:
            return
        for f in files[:excess]:
            try:
                f.unlink()
            except OSError:
                pass

    def save(
        self,
        epoch: int,
        model: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        payload = self._build_payload(epoch, model, optimizer, scheduler, metrics, extra)
        path = self._checkpoint_path(epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        latest_path = self._latest_path()
        torch.save(payload, latest_path)
        if metrics is not None and self._update_best(metrics):
            best_path = self._best_path()
            torch.save(payload, best_path)
        self._prune_old_checkpoints()
        return path

    def load(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> Dict[str, Any]:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return torch.load(p, map_location=map_location)

    def load_latest(
        self,
        map_location: str | torch.device | None = None,
    ) -> Optional[Dict[str, Any]]:
        p = self._latest_path()
        if not p.exists():
            return None
        return torch.load(p, map_location=map_location)

    def load_best(
        self,
        map_location: str | torch.device | None = None,
    ) -> Optional[Dict[str, Any]]:
        p = self._best_path()
        if not p.exists():
            return None
        return torch.load(p, map_location=map_location)

    @staticmethod
    def restore_objects(
        checkpoint: Mapping[str, Any],
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True,
    ) -> int:
        epoch = int(checkpoint.get("epoch", 0))
        if model is not None and "model" in checkpoint:
            model_state = checkpoint["model"]
            if hasattr(model, "load_state_dict"):
                model.load_state_dict(model_state, strict=strict)
        if optimizer is not None and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            opt_state = checkpoint["optimizer"]
            if hasattr(optimizer, "load_state_dict"):
                optimizer.load_state_dict(opt_state)
        if scheduler is not None and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            sched_state = checkpoint["scheduler"]
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(sched_state)
        return epoch
