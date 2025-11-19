from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Iterable
from pathlib import Path
import time, math, json, os
import torch
from torch.nn.utils import clip_grad_norm_

# --------- State & Base ---------
@dataclass
class TrainState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[Any] = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    device: Optional[torch.device] = None
    epoch: int = 0
    step: int = 0                 # step in current epoch
    global_step: int = 0          # absolute step
    best_score: Optional[float] = None
    best_step: Optional[int] = None
    save_dir: str = "checkpoints"
    cfg: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

class Callback:
    priority: int = 10  # küçük önce çalışır
    def setup(self, state: TrainState): pass
    def on_fit_start(self, state: TrainState): pass
    def on_epoch_start(self, state: TrainState): pass
    def on_batch_start(self, state: TrainState, batch: Any | None = None): pass
    def on_before_backward(self, state: TrainState, loss: torch.Tensor) -> torch.Tensor: return loss
    def on_after_backward(self, state: TrainState): pass
    def on_before_optimizer_step(self, state: TrainState): pass
    def on_after_optimizer_step(self, state: TrainState): pass
    def on_batch_end(self, state: TrainState, loss_val: float | None = None): pass
    def on_validation_start(self, state: TrainState): pass
    def on_validation_end(self, state: TrainState, metrics: Dict[str, float]): pass
    def on_checkpoint(self, state: TrainState): pass
    def on_fit_end(self, state: TrainState): pass

class CallbackList(Callback):
    def __init__(self, callbacks: Iterable[Callback]):
        self.cbs = sorted(list(callbacks), key=lambda c: getattr(c, "priority", 10))
    def setup(self, state):                 [c.setup(state) for c in self.cbs]
    def on_fit_start(self, state):          [c.on_fit_start(state) for c in self.cbs]
    def on_epoch_start(self, state):        [c.on_epoch_start(state) for c in self.cbs]
    def on_batch_start(self, state, b=None):[c.on_batch_start(state, b) for c in self.cbs]
    def on_before_backward(self, state, l): 
        for c in self.cbs: l = c.on_before_backward(state, l)
        return l
    def on_after_backward(self, state):     [c.on_after_backward(state) for c in self.cbs]
    def on_before_optimizer_step(self, state): [c.on_before_optimizer_step(state) for c in self.cbs]
    def on_after_optimizer_step(self, state):  [c.on_after_optimizer_step(state) for c in self.cbs]
    def on_batch_end(self, state, loss_val=None): [c.on_batch_end(state, loss_val) for c in self.cbs]
    def on_validation_start(self, state):   [c.on_validation_start(state) for c in self.cbs]
    def on_validation_end(self, state, m):  [c.on_validation_end(state, m) for c in self.cbs]
    def on_checkpoint(self, state):         [c.on_checkpoint(state) for c in self.cbs]
    def on_fit_end(self, state):            [c.on_fit_end(state) for c in self.cbs]

# --------- Utils ---------
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def _save_ckpt(path: Path, state: TrainState, tag: str, metric_name: Optional[str] = None):
    obj = {
        "tag": tag,
        "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": (state.scheduler.state_dict() if state.scheduler and hasattr(state.scheduler, "state_dict") else None),
        "scaler": (state.scaler.state_dict() if state.scaler is not None else None),
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_score": state.best_score,
        "best_step": state.best_step,
        "metric_name": metric_name,
    }
    torch.save(obj, path)

# --------- Concrete Callbacks ---------
class ModelCheckpoint(Callback):
    priority = 20
    def __init__(self, save_dir="checkpoints", monitor="val_loss", mode="min",
                 save_last=True, save_best=True, max_keep=3, filename_prefix="ckpt"):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_best = save_best
        self.max_keep = max_keep
        self.filename_prefix = filename_prefix
        self.saved: List[Path] = []
    def setup(self, state):
        _ensure_dir(self.save_dir)
        if state.best_score is None:
            state.best_score = math.inf if self.mode == "min" else -math.inf
    def _better(self, x, y):
        return (x < y) if self.mode == "min" else (x > y)
    def on_validation_end(self, state, metrics):
        metric = metrics.get(self.monitor)
        if metric is None: return
        if self.save_best and self._better(metric, state.best_score):
            state.best_score = float(metric)
            state.best_step = state.global_step
            path = Path(self.save_dir) / f"{self.filename_prefix}_best.pth"
            _save_ckpt(path, state, "best", self.monitor)
        if self.save_last:
            path = Path(self.save_dir) / f"{self.filename_prefix}_last.pth"
            _save_ckpt(path, state, "last", self.monitor)
        if self.max_keep and self.max_keep > 0:
            # örnek rotasyon: epoch-step isimli klasörlü kayıt istersen burayı genişlet
            pass

class EarlyStopping(Callback):
    priority = 30
    def __init__(self, monitor="val_loss", mode="min", patience=3, min_delta=0.0, restore_best=True):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.wait = 0
        self.best_snapshot: Optional[Dict[str, torch.Tensor]] = None
    def setup(self, state):
        if state.best_score is None:
            state.best_score = math.inf if self.mode == "min" else -math.inf
    def _improved(self, cur, best):
        if self.mode == "min":
            return cur < (best - self.min_delta)
        return cur > (best + self.min_delta)
    def on_validation_end(self, state, metrics):
        cur = metrics.get(self.monitor)
        if cur is None: return
        if self._improved(cur, state.best_score):
            state.best_score = float(cur)
            state.best_step = state.global_step
            self.wait = 0
            if self.restore_best:
                self.best_snapshot = {k: v.detach().cpu().clone() for k, v in state.model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                state.extra["stop_training"] = True
                if self.restore_best and self.best_snapshot is not None:
                    state.model.load_state_dict(self.best_snapshot, strict=True)

class GradClip(Callback):
    priority = 15
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    def on_before_optimizer_step(self, state):
        if self.max_norm and self.max_norm > 0:
            clip_grad_norm_(state.model.parameters(), self.max_norm)

class EMACallback(Callback):
    priority = 25
    def __init__(self, decay=0.9999, update_after=100, use_for_validation=True):
        self.decay = float(decay)
        self.update_after = int(update_after)
        self.use_for_validation = bool(use_for_validation)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._in_val = False
    def setup(self, state):
        for n, p in state.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()
    def on_after_optimizer_step(self, state):
        if state.global_step < self.update_after: return
        with torch.no_grad():
            for n, p in state.model.named_parameters():
                if not p.requires_grad: continue
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
    def on_validation_start(self, state):
        if not self.use_for_validation: return
        self.backup = {}
        for n, p in state.model.named_parameters():
            self.backup[n] = p.detach().clone()
            if n in self.shadow:
                p.data.copy_(self.shadow[n])
        self._in_val = True
    def on_validation_end(self, state, metrics):
        if not self._in_val: return
        for n, p in state.model.named_parameters():
            p.data.copy_(self.backup[n])
        self.backup = {}
        self._in_val = False

class CosineWarmupLR(Callback):
    priority = 12
    def __init__(self, warmup_steps: int, total_steps: int, base_lrs: Optional[List[float]] = None):
        self.warmup_steps = int(warmup_steps)
        self.total_steps = max(1, int(total_steps))
        self.base_lrs = base_lrs
    def setup(self, state):
        if self.base_lrs is None:
            self.base_lrs = [pg["lr"] for pg in state.optimizer.param_groups]
        self._apply_lr(state, state.global_step)
    def on_after_optimizer_step(self, state):
        self._apply_lr(state, state.global_step + 1)
    def _schedule(self, step):
        if step <= self.warmup_steps:
            return step / max(1, self.warmup_steps)
        t = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    def _apply_lr(self, state, step):
        s = self._schedule(step)
        for pg, base in zip(state.optimizer.param_groups, self.base_lrs):
            pg["lr"] = float(base) * float(s)

class LrMonitor(Callback):
    priority = 50
    def __init__(self, log_every_n_steps=50):
        self.log_every = int(log_every_n_steps)
    def on_batch_end(self, state, loss_val=None):
        if self.log_every and state.global_step % self.log_every == 0:
            lrs = [pg["lr"] for pg in state.optimizer.param_groups]
            state.extra.setdefault("logs", []).append({"step": state.global_step, "lr": float(lrs[0])})

class CSVLogger(Callback):
    priority = 60
    def __init__(self, log_path="logs/train.csv", write_header=True):
        self.log_path = log_path
        self.write_header = write_header
        self._initialized = False

    def on_fit_start(self, state):
        p = Path(self.log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if self.write_header and not p.exists():
            with p.open("w", encoding="utf-8") as f:
                f.write("step,epoch,loss,lr,tag,json\n")
        self._initialized = True

    def _append(self, row: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(row + "\n")

    def on_batch_end(self, state, loss_val=None):
        if not self._initialized:
            return
        # Güvenli sayısal formatlama
        try:
            lr_val = float(state.optimizer.param_groups[0]["lr"])
        except Exception:
            lr_val = state.optimizer.param_groups[0]["lr"]  # neyse o (string vs.)
        if loss_val is None:
            loss_str = ""
        else:
            try:
                loss_str = f"{float(loss_val):.6f}"
            except Exception:
                loss_str = str(loss_val)

        payload = {
            "type": "batch",
            "extra": state.extra.get("logs", [])[-1] if state.extra.get("logs") else None
        }
        self._append(f"{state.global_step},{state.epoch},{loss_str},{lr_val},batch,{json.dumps(payload, ensure_ascii=False)}")

    def on_validation_end(self, state, metrics):
        if not self._initialized:
            return
        try:
            lr_val = float(state.optimizer.param_groups[0]["lr"])
        except Exception:
            lr_val = state.optimizer.param_groups[0]["lr"]
        self._append(f"{state.global_step},{state.epoch},,{lr_val},val,{json.dumps(metrics, ensure_ascii=False)}")

class Throughput(Callback):
    priority = 55
    def __init__(self, window=50):
        self.window = int(window)
        self.times: List[float] = []
        self._t0 = None
    def on_batch_start(self, state, batch=None):
        self._t0 = time.time()
    def on_batch_end(self, state, loss_val=None):
        if self._t0 is None: return
        dt = time.time() - self._t0
        self.times.append(dt)
        if len(self.times) > self.window: self.times.pop(0)
        if "last_batch_token_count" in state.extra:
            # opsiyonel: eğitim döngüsü burayı doldurursa token/sn loglanır
            tokens = state.extra["last_batch_token_count"]
            avg_dt = sum(self.times)/len(self.times)
            tps = tokens / avg_dt if avg_dt > 0 else 0.0
            state.extra.setdefault("logs", []).append({"step": state.global_step, "tps": float(tps)})

class HysoCallbacks:
    @staticmethod
    def default(total_steps: int,
                save_dir: str = "checkpoints",
                monitor: str = "val_loss",
                mode: str = "min",
                patience: int = 3,
                min_delta: float = 0.0,
                ema_decay: Optional[float] = 0.9999,
                ema_update_after: int = 100,
                clip_grad: Optional[float] = 1.0,
                warmup_steps: int = 1000,
                log_csv: str = "logs/train.csv") -> CallbackList:
        cbs: List[Callback] = []
        if clip_grad and clip_grad > 0: cbs.append(GradClip(clip_grad))
        cbs.append(CosineWarmupLR(warmup_steps=warmup_steps, total_steps=total_steps))
        if ema_decay: cbs.append(EMACallback(decay=ema_decay, update_after=ema_update_after, use_for_validation=True))
        cbs.append(ModelCheckpoint(save_dir=save_dir, monitor=monitor, mode=mode, save_last=True, save_best=True))
        cbs.append(EarlyStopping(monitor=monitor, mode=mode, patience=patience, min_delta=min_delta, restore_best=True))
        cbs.append(LrMonitor(log_every_n_steps=50))
        cbs.append(Throughput(window=50))
        cbs.append(CSVLogger(log_path=log_csv, write_header=True))
        return CallbackList(cbs)

    @staticmethod
    def minimal(total_steps: int, save_dir: str = "checkpoints") -> CallbackList:
        return CallbackList([
            CosineWarmupLR(warmup_steps=1000, total_steps=total_steps),
            ModelCheckpoint(save_dir=save_dir),
            CSVLogger(),
        ])

__all__ = [
    "TrainState", "Callback", "CallbackList",
    "ModelCheckpoint", "EarlyStopping", "GradClip",
    "EMACallback", "CosineWarmupLR", "LrMonitor",
    "CSVLogger", "Throughput", "HysoCallbacks"
]