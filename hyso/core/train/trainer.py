from __future__ import annotations
from typing import Optional, Dict, Any, Union, List
import time
import contextlib

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .callbacks import TrainState, Callback, CallbackList, HysoCallbacks
from .metrics import LLMetrics, AverageMetric
from .scheduler import build_scheduler, BaseLRScheduler


class DecoderOnlyDataset(Dataset):
    def __init__(self, sequences: List[Union[List[int], torch.Tensor]]):
        self.sequences = []
        for x in sequences:
            t = torch.as_tensor(x, dtype=torch.long)
            if t.dim() > 1:
                t = t.reshape(-1)
            self.sequences.append(t)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.sequences[idx]}


class EncoderOnlyDataset(Dataset):
    def __init__(self, sequences: List[Union[List[int], torch.Tensor]], labels: List[Union[int, torch.Tensor]]):
        assert len(sequences) == len(labels)
        self.sequences = []
        self.labels = []
        for x in sequences:
            t = torch.as_tensor(x, dtype=torch.long)
            if t.dim() > 1:
                t = t.reshape(-1)
            self.sequences.append(t)
        for y in labels:
            lab = torch.as_tensor(y, dtype=torch.long)
            if lab.dim() > 1:
                lab = lab.reshape(-1)
            self.labels.append(lab)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.sequences[idx], "labels": self.labels[idx]}


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        src_sequences: List[Union[List[int], torch.Tensor]],
        tgt_sequences: List[Union[List[int], torch.Tensor]],
    ):
        assert len(src_sequences) == len(tgt_sequences)
        self.src_sequences = []
        self.tgt_sequences = []
        for x in src_sequences:
            t = torch.as_tensor(x, dtype=torch.long)
            if t.dim() > 1:
                t = t.reshape(-1)
            self.src_sequences.append(t)
        for y in tgt_sequences:
            t = torch.as_tensor(y, dtype=torch.long)
            if t.dim() > 1:
                t = t.reshape(-1)
            self.tgt_sequences.append(t)

    def __len__(self) -> int:
        return len(self.src_sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"src_ids": self.src_sequences[idx], "tgt_ids": self.tgt_sequences[idx]}


def decoder_only_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int = 0,
) -> Dict[str, torch.Tensor]:
    input_list = []
    for item in batch:
        ids = item["input_ids"]
        if not torch.is_tensor(ids):
            ids = torch.as_tensor(ids, dtype=torch.long)
        if ids.dim() > 1:
            ids = ids.reshape(-1)
        input_list.append(ids)
    max_len = max(x.size(0) for x in input_list)
    bsz = len(input_list)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, ids in enumerate(input_list):
        l = ids.size(0)
        input_ids[i, :l] = ids
        attention_mask[i, :l] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def encoder_only_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int = 0,
) -> Dict[str, torch.Tensor]:
    input_list = []
    label_list = []
    for item in batch:
        ids = item["input_ids"]
        if not torch.is_tensor(ids):
            ids = torch.as_tensor(ids, dtype=torch.long)
        if ids.dim() > 1:
            ids = ids.reshape(-1)
        lab = item.get("labels", item.get("label", None))
        if lab is not None and not torch.is_tensor(lab):
            lab = torch.as_tensor(lab, dtype=torch.long)
        if lab is not None and lab.dim() > 1:
            lab = lab.reshape(-1)
        input_list.append(ids)
        label_list.append(lab)
    max_len = max(x.size(0) for x in input_list)
    bsz = len(input_list)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, ids in enumerate(input_list):
        l = ids.size(0)
        input_ids[i, :l] = ids
        attention_mask[i, :l] = 1
    if label_list[0] is not None:
        labels = torch.stack(label_list)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def seq2seq_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int = 0,
) -> Dict[str, torch.Tensor]:
    src_list = []
    tgt_list = []
    for item in batch:
        src = item["src_ids"]
        tgt = item["tgt_ids"]
        if not torch.is_tensor(src):
            src = torch.as_tensor(src, dtype=torch.long)
        if not torch.is_tensor(tgt):
            tgt = torch.as_tensor(tgt, dtype=torch.long)
        if src.dim() > 1:
            src = src.reshape(-1)
        if tgt.dim() > 1:
            tgt = tgt.reshape(-1)
        src_list.append(src)
        tgt_list.append(tgt)
    max_src = max(x.size(0) for x in src_list)
    max_tgt = max(y.size(0) for y in tgt_list)
    bsz = len(src_list)
    src_ids = torch.full((bsz, max_src), pad_id, dtype=torch.long)
    src_mask = torch.zeros((bsz, max_src), dtype=torch.long)
    tgt_ids = torch.full((bsz, max_tgt), pad_id, dtype=torch.long)
    for i, src in enumerate(src_list):
        ls = src.size(0)
        src_ids[i, :ls] = src
        src_mask[i, :ls] = 1
    for i, tgt in enumerate(tgt_list):
        lt = tgt.size(0)
        tgt_ids[i, :lt] = tgt
    return {"src_ids": src_ids, "src_mask": src_mask, "tgt_ids": tgt_ids}


def build_dataloader(
    dataset: Dataset,
    mode: str,
    batch_size: int,
    pad_id: int = 0,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    mode = mode.lower()
    if mode == "decoder_only":
        collate_fn = lambda b: decoder_only_collate_fn(b, pad_id=pad_id)
    elif mode == "encoder_only":
        collate_fn = lambda b: encoder_only_collate_fn(b, pad_id=pad_id)
    elif mode == "seq2seq":
        collate_fn = lambda b: seq2seq_collate_fn(b, pad_id=pad_id)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


class HysoTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader=None,
        *,
        mode: str = "decoder_only",
        tokenizer: Optional[Any] = None,
        epochs: int = 1,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        optimizer: str = "adamw",
        scheduler: str = "cosine_warmup",
        warmup_ratio: float = 0.1,
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        callbacks: Optional[Union[CallbackList, List[Callback], Any]] = None,
        metrics: Optional[LLMetrics] = None,
        ignore_index: int = -100,
        save_dir: str = "checkpoints",
        log_interval: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        greedy: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mode = mode.lower()
        self.tokenizer = tokenizer
        self.epochs = int(epochs)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.optimizer_name = (optimizer or "adamw").lower()
        self.scheduler_name = (scheduler or "none").lower()
        self.warmup_ratio = float(warmup_ratio)
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.ignore_index = int(ignore_index)
        self.save_dir = save_dir
        self.log_interval = max(1, int(log_interval))
        self.callbacks_cfg = callbacks
        self.metrics = metrics
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.greedy = bool(greedy)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[BaseLRScheduler] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None

    def _get_pad_id(self) -> int:
        if hasattr(self.model, "pad_id"):
            return int(self.model.pad_id)
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_id"):
            return int(self.tokenizer.pad_id)
        return 0

    def _sanitize_parameters(self):
        for p in self.model.parameters():
            if p is not None and torch.is_tensor(p.data):
                if not torch.isfinite(p.data).all():
                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e4, neginf=-1e4)

    def _sanitize_gradients(self):
        for p in self.model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1e4, neginf=-1e4)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        params = self.model.parameters()
        if self.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        return self.optimizer

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int) -> Optional[BaseLRScheduler]:
        if self.scheduler_name in ("none", "constant", "", None):
            self.scheduler = None
            return None
        warmup_steps = int(self.warmup_ratio * total_steps)
        self.scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_type=self.scheduler_name,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
        return self.scheduler

    def _build_callbacks(self, total_steps: int) -> CallbackList:
        cfg = self.callbacks_cfg
        if cfg is None:
            return HysoCallbacks.default(
                total_steps=total_steps,
                save_dir=self.save_dir,
            )
        if isinstance(cfg, CallbackList):
            return cfg
        if isinstance(cfg, list):
            return CallbackList(cfg)
        return cfg

    def _move_to_device(self, batch: Any) -> Any:
        if torch.is_tensor(batch):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return [self._move_to_device(x) for x in batch]
        return batch

    def _compute_loss(self, batch: Dict[str, Any]) -> tuple:
        mode = self.mode
        logits = None
        labels = None
        n_tokens = 0
        pad_id = self._get_pad_id()

        if mode == "decoder_only":
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            labels = batch.get("labels", None)
            if labels is None:
                labels = input_ids[:, 1:].contiguous()
            labels = labels.masked_fill(labels == pad_id, self.ignore_index)
            if logits.size(1) > labels.size(1):
                logits = logits[:, : labels.size(1), :]
            elif logits.size(1) < labels.size(1):
                labels = labels[:, : logits.size(1)]
            vocab_size = logits.size(-1)
            logits_for_loss = torch.nan_to_num(
                logits.float(), nan=0.0, posinf=1e4, neginf=-1e4
            )
            valid = labels != self.ignore_index
            n_tokens = int(valid.sum().item())
            if n_tokens == 0:
                loss = torch.zeros((), device=logits_for_loss.device, dtype=logits_for_loss.dtype, requires_grad=True)
                return loss, logits_for_loss, labels, n_tokens
            loss = F.cross_entropy(
                logits_for_loss.reshape(-1, vocab_size),
                labels.reshape(-1),
                ignore_index=self.ignore_index,
            )
            return loss, logits_for_loss, labels, n_tokens

        elif mode == "seq2seq":
            src_ids = batch["src_ids"]
            src_mask = batch.get("src_mask", None)
            tgt_ids = batch["tgt_ids"]
            if src_mask is not None:
                logits = self.model(
                    src_ids=src_ids,
                    tgt_ids=tgt_ids,
                    src_attention_mask=src_mask,
                )
            else:
                logits = self.model(src_ids=src_ids, tgt_ids=tgt_ids)
            labels = batch.get("labels", None)
            if labels is None:
                labels = tgt_ids[:, 1:].contiguous()
            labels = labels.masked_fill(labels == pad_id, self.ignore_index)
            if logits.size(1) > labels.size(1):
                logits = logits[:, : labels.size(1), :]
            elif logits.size(1) < labels.size(1):
                labels = labels[:, : logits.size(1)]
            vocab_size = logits.size(-1)
            logits_for_loss = torch.nan_to_num(
                logits.float(), nan=0.0, posinf=1e4, neginf=-1e4
            )
            valid = labels != self.ignore_index
            n_tokens = int(valid.sum().item())
            if n_tokens == 0:
                loss = torch.zeros((), device=logits_for_loss.device, dtype=logits_for_loss.dtype, requires_grad=True)
                return loss, logits_for_loss, labels, n_tokens
            loss = F.cross_entropy(
                logits_for_loss.reshape(-1, vocab_size),
                labels.reshape(-1),
                ignore_index=self.ignore_index,
            )
            return loss, logits_for_loss, labels, n_tokens

        elif mode == "encoder_only":
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            labels = batch["labels"]
            if attention_mask is not None:
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                logits = self.model(input_ids=input_ids)
            num_classes = logits.size(-1)
            logits_for_loss = torch.nan_to_num(
                logits.float(), nan=0.0, posinf=1e4, neginf=-1e4
            )
            valid = labels != self.ignore_index
            n_tokens = int(valid.sum().item())
            if n_tokens == 0:
                loss = torch.zeros((), device=logits_for_loss.device, dtype=logits_for_loss.dtype, requires_grad=True)
                return loss, logits_for_loss, labels, n_tokens
            loss = F.cross_entropy(
                logits_for_loss.reshape(-1, num_classes),
                labels.reshape(-1),
                ignore_index=self.ignore_index,
            )
            return loss, logits_for_loss, labels, n_tokens

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _run_validation(self, state: TrainState) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        loss_meter = AverageMetric()
        metric_obj = self.metrics
        if metric_obj is not None:
            metric_obj.reset()
        amp_enabled = self.use_amp and self.device.type == "cuda"
        autocast_ctx = torch.amp.autocast("cuda") if amp_enabled else contextlib.nullcontext()
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_to_device(batch)
                with autocast_ctx:
                    loss, logits, labels, n_tokens = self._compute_loss(batch)
                loss_val = float(loss.detach().item())
                loss_meter.update(loss_val, weight=max(n_tokens, 1))
                if metric_obj is not None:
                    metric_obj.update_loss(loss_val, weight=max(n_tokens, 1))
                    if logits is not None and labels is not None:
                        metric_obj.update_logits(logits.detach(), labels)
                    if metric_obj.use_bleu and self.tokenizer is not None and logits is not None and labels is not None:
                        preds_ids = logits.argmax(dim=-1)
                        labels_for_decode = labels.clone()
                        pad_id = self._get_pad_id()
                        labels_for_decode[labels_for_decode == self.ignore_index] = pad_id
                        try:
                            pred_texts = self.tokenizer.batch_decode(preds_ids)
                            ref_texts = self.tokenizer.batch_decode(labels_for_decode)
                            metric_obj.update_texts(pred_texts, ref_texts)
                        except Exception:
                            pass
        val_loss = float(loss_meter.compute())
        metrics_out: Dict[str, float] = {"val_loss": val_loss}
        if metric_obj is not None:
            other = metric_obj.compute()
            for k, v in other.items():
                if k == "loss":
                    continue
                metrics_out[f"val_{k}"] = float(v)
        state.extra["val_loss"] = val_loss
        state.extra["val_metrics"] = metrics_out
        return metrics_out

    def fit(self, epochs: Optional[int] = None, max_steps: Optional[int] = None) -> Dict[str, Any]:
        num_epochs = int(epochs) if epochs is not None else self.epochs
        self.model.to(self.device)
        self._sanitize_parameters()
        steps_per_epoch = len(self.train_loader)
        if steps_per_epoch == 0:
            raise ValueError("train_loader is empty")
        total_steps = steps_per_epoch * num_epochs
        if max_steps is not None:
            total_steps = min(total_steps, int(max_steps))
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, total_steps)
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.scaler = scaler
        state = TrainState(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=self.device,
            epoch=0,
            step=0,
            global_step=0,
            save_dir=self.save_dir,
            cfg=None,
        )
        callbacks = self._build_callbacks(total_steps)
        callbacks.setup(state)
        callbacks.on_fit_start(state)
        global_start = time.time()
        for epoch in range(num_epochs):
            state.epoch = epoch
            state.step = 0
            self.model.train()
            train_loss_meter = AverageMetric()
            callbacks.on_epoch_start(state)
            epoch_start = time.time()
            for step, batch in enumerate(self.train_loader):
                if max_steps is not None and state.global_step >= max_steps:
                    break
                state.step = step
                state.global_step += 1
                batch = self._move_to_device(batch)
                callbacks.on_batch_start(state, batch)
                amp_enabled = self.use_amp and self.device.type == "cuda"
                autocast_ctx = torch.amp.autocast("cuda") if amp_enabled else contextlib.nullcontext()
                with autocast_ctx:
                    loss, logits, labels, n_tokens = self._compute_loss(batch)
                if not torch.isfinite(loss):
                    loss = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
                    n_tokens = 0
                loss_for_backward = loss / self.grad_accum_steps
                loss_for_backward = callbacks.on_before_backward(state, loss_for_backward)
                if scaler is not None:
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()
                self._sanitize_gradients()
                callbacks.on_after_backward(state)
                state.extra["last_batch_token_count"] = int(n_tokens)
                if state.global_step % self.grad_accum_steps == 0:
                    callbacks.on_before_optimizer_step(state)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    self._sanitize_parameters()
                    if scheduler is not None:
                        scheduler.step()
                    callbacks.on_after_optimizer_step(state)
                loss_val = float(loss.detach().item())
                train_loss_meter.update(loss_val, weight=max(n_tokens, 1))
                callbacks.on_batch_end(state, loss_val)
                if (step + 1) % self.log_interval == 0 or (step + 1) == steps_per_epoch:
                    now = time.time()
                    pct = (step + 1) / steps_per_epoch
                    bar_len = 30
                    filled = int(bar_len * pct)
                    bar = "█" * filled + "-" * (bar_len - filled)
                    elapsed_epoch = now - epoch_start
                    if step > 0:
                        eta_epoch = elapsed_epoch / (step + 1) * (steps_per_epoch - step - 1)
                    else:
                        eta_epoch = 0.0
                    elapsed_total_min = (now - global_start) / 60.0
                    print(
                        f"\rEpoch {epoch+1}/{num_epochs} [{bar}] {pct*100:5.1f}% "
                        f"loss={loss_val:.4f} "
                        f"epoch_eta={eta_epoch/60:.1f}m "
                        f"total_elapsed={elapsed_total_min:.1f}m",
                        end="",
                    )
            print("")
            epoch_train_loss = float(train_loss_meter.compute())
            state.extra["train_loss"] = epoch_train_loss
            val_metrics: Dict[str, float] = {}
            if self.val_loader is not None:
                callbacks.on_validation_start(state)
                val_metrics = self._run_validation(state)
                metrics_for_cb = dict(val_metrics)
                metrics_for_cb["train_loss"] = epoch_train_loss
                callbacks.on_validation_end(state, metrics_for_cb)
            else:
                metrics_for_cb = {"train_loss": epoch_train_loss}
                callbacks.on_validation_end(state, metrics_for_cb)
            epoch_time_min = (time.time() - epoch_start) / 60.0
            print(f"Epoch {epoch+1}/{num_epochs} finished in {epoch_time_min:.2f} minutes")
            if state.extra.get("stop_training"):
                print("Early stopping triggered.")
                break
        total_min = (time.time() - global_start) / 60.0
        print(f"Training finished in {total_min:.2f} minutes, total_steps={state.global_step}")
        callbacks.on_fit_end(state)
        out: Dict[str, Any] = {
            "state": state,
            "train_loss": float(state.extra.get("train_loss", float("nan"))),
            "val_metrics": state.extra.get("val_metrics", {}),
        }
        return out


__all__ = [
    "DecoderOnlyDataset",
    "EncoderOnlyDataset",
    "Seq2SeqDataset",
    "decoder_only_collate_fn",
    "encoder_only_collate_fn",
    "seq2seq_collate_fn",
    "build_dataloader",
    "HysoTrainer",
]