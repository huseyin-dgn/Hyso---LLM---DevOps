from __future__ import annotations
from typing import Any, Dict, List, Optional
import math
import torch


class BaseLRScheduler:
    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError


class NoOpScheduler(BaseLRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.last_step = 0

    def step(self) -> None:
        self.last_step += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"last_step": self.last_step}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.last_step = int(state_dict.get("last_step", 0))


class WarmupCosineScheduler(BaseLRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        base_lrs: Optional[List[float]] = None,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr = float(min_lr)
        if base_lrs is None:
            self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        else:
            self.base_lrs = list(base_lrs)
        self.last_step = 0
        self._apply_lr(self.last_step)

    def _schedule_factor(self, step: int) -> float:
        if self.total_steps <= 0:
            return 1.0
        if step <= self.warmup_steps and self.warmup_steps > 0:
            return step / float(self.warmup_steps)
        if self.total_steps <= self.warmup_steps:
            return 1.0
        t = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        t = min(max(t, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    def _apply_lr(self, step: int) -> None:
        factor = self._schedule_factor(step)
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            base_lr = float(base)
            lr = self.min_lr + (base_lr - self.min_lr) * factor
            pg["lr"] = float(lr)

    def step(self) -> None:
        self.last_step += 1
        self._apply_lr(self.last_step)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "base_lrs": self.base_lrs,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.min_lr = float(state_dict.get("min_lr", self.min_lr))
        self.base_lrs = list(state_dict.get("base_lrs", self.base_lrs))
        self.last_step = int(state_dict.get("last_step", 0))
        self._apply_lr(self.last_step)


class WarmupLinearScheduler(BaseLRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        base_lrs: Optional[List[float]] = None,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr = float(min_lr)
        if base_lrs is None:
            self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        else:
            self.base_lrs = list(base_lrs)
        self.last_step = 0
        self._apply_lr(self.last_step)

    def _schedule_factor(self, step: int) -> float:
        if self.total_steps <= 0:
            return 1.0
        if step <= self.warmup_steps and self.warmup_steps > 0:
            return step / float(self.warmup_steps)
        if self.total_steps <= self.warmup_steps:
            return 1.0
        t = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        t = min(max(t, 0.0), 1.0)
        return 1.0 - t

    def _apply_lr(self, step: int) -> None:
        factor = self._schedule_factor(step)
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            base_lr = float(base)
            lr = self.min_lr + (base_lr - self.min_lr) * factor
            pg["lr"] = float(lr)

    def step(self) -> None:
        self.last_step += 1
        self._apply_lr(self.last_step)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "base_lrs": self.base_lrs,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.min_lr = float(state_dict.get("min_lr", self.min_lr))
        self.base_lrs = list(state_dict.get("base_lrs", self.base_lrs))
        self.last_step = int(state_dict.get("last_step", 0))
        self._apply_lr(self.last_step)


class StepScheduler(BaseLRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int = 1000,
        gamma: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.step_size = max(1, int(step_size))
        self.gamma = float(gamma)
        self.last_step = 0

    def step(self) -> None:
        self.last_step += 1
        if self.last_step % self.step_size != 0:
            return
        for pg in self.optimizer.param_groups:
            pg["lr"] = float(pg["lr"]) * self.gamma

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_size": self.step_size,
            "gamma": self.gamma,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.step_size = int(state_dict.get("step_size", self.step_size))
        self.gamma = float(state_dict.get("gamma", self.gamma))
        self.last_step = int(state_dict.get("last_step", 0))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine_warmup",
    total_steps: Optional[int] = None,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
    base_lrs: Optional[List[float]] = None,
    step_size: int = 1000,
    gamma: float = 0.1,
) -> BaseLRScheduler:
    t = (scheduler_type or "none").lower()

    if t in ("none", "constant"):
        return NoOpScheduler(optimizer)

    if t in ("cosine", "cosine_warmup"):
        if total_steps is None:
            raise ValueError("total_steps must be provided for cosine_warmup scheduler")
        return WarmupCosineScheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
            base_lrs=base_lrs,
        )

    if t in ("linear", "linear_warmup"):
        if total_steps is None:
            raise ValueError("total_steps must be provided for linear_warmup scheduler")
        return WarmupLinearScheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
            base_lrs=base_lrs,
        )

    if t in ("step", "steplr"):
        return StepScheduler(
            optimizer=optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    raise ValueError(f"Unknown scheduler_type: {scheduler_type}")


__all__ = [
    "BaseLRScheduler",
    "NoOpScheduler",
    "WarmupCosineScheduler",
    "WarmupLinearScheduler",
    "StepScheduler",
    "build_scheduler",
]