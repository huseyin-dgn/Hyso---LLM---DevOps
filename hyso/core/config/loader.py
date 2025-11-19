# Config yükleme, birleştirme ve override işlemleri

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional
import json
import ast

try:
    import yaml  # type: ignore[import]
except ImportError:
    yaml = None  # type: ignore[assignment]


@dataclass(slots=True)
class Config:
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return self.data

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __getattr__(self, item: str) -> Any:
        try:
            return self.data[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def load_config(path: str | Path) -> Config:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    suffix = p.suffix.lower()
    if suffix in (".yml", ".yaml"):
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configs")
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    elif suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raise ValueError(f"Unsupported config extension: {suffix}")
    if raw is None:
        data: Dict[str, Any] = {}
    elif isinstance(raw, Mapping):
        data = dict(raw)
    else:
        raise ValueError("Config root must be a mapping")
    return Config(data=data)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], Mapping) and isinstance(v, Mapping):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def merge_config(base: Config | Mapping[str, Any], override: Mapping[str, Any]) -> Config:
    if isinstance(base, Config):
        base_dict = base.data
    else:
        base_dict = dict(base)
    merged = _deep_merge(base_dict, override)
    return Config(data=merged)


def _parse_override_value(raw: str) -> Any:
    s = raw.strip()
    if not s:
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        lowered = s.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"null", "none"}:
            return None
        return s


def _set_by_path(root: MutableMapping[str, Any], path: str, value: Any) -> None:
    parts = [p for p in path.split(".") if p]
    if not parts:
        return
    curr: MutableMapping[str, Any] = root
    for key in parts[:-1]:
        if key not in curr or not isinstance(curr[key], MutableMapping):
            curr[key] = {}
        child = curr[key]
        if not isinstance(child, MutableMapping):
            new_child: Dict[str, Any] = {}
            curr[key] = new_child
            curr = new_child
        else:
            curr = child
    curr[parts[-1]] = value


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = _parse_override_value(raw)
        _set_by_path(overrides, key, value)
    return overrides


def load_config_with_overrides(
    path: str | Path,
    override_pairs: Optional[List[str]] = None,
) -> Config:
    cfg = load_config(path)
    if override_pairs:
        overrides = parse_overrides(override_pairs)
        cfg = merge_config(cfg, overrides)
    return cfg


def save_config(path: str | Path, config: Config | Mapping[str, Any]) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(config, Config):
        payload = config.to_dict()
    else:
        payload = dict(config)
    suffix = p.suffix.lower()
    if suffix in (".yml", ".yaml"):
        if yaml is None:
            raise ImportError("PyYAML is required to save YAML configs")
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    elif suffix == ".json":
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        raise ValueError(f"Unsupported config extension for save: {suffix}")