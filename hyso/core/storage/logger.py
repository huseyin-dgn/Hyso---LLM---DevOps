# Konsol ve dosya tabanlı logging için yapılandırma

from __future__ import annotations

import logging
import logging.handlers
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal


@dataclass(slots=True)
class LoggingConfig:
    name: str = "train"
    level: int = logging.INFO
    log_dir: Optional[Path] = None
    filename: str = "train.log"
    rotation: Literal["none", "size", "time"] = "none"
    max_bytes: int = 10_000_000
    backup_count: int = 5
    when: str = "D"
    interval: int = 1
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


def configure_logger(config: LoggingConfig) -> logging.Logger:
    logger = logging.getLogger(config.name)
    if logger.handlers:
        return logger

    logger.setLevel(config.level)

    formatter = logging.Formatter(config.fmt, datefmt=config.datefmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if config.log_dir is not None:
        log_dir = config.log_dir.expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / config.filename

        if config.rotation == "size":
            file_handler: logging.Handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
        elif config.rotation == "time":
            file_handler = logging.handlers.TimedRotatingFileHandler(
                file_path,
                when=config.when,
                interval=config.interval,
                backupCount=config.backup_count,
                encoding="utf-8",
                utc=False,
            )
        else:
            file_handler = logging.FileHandler(file_path, encoding="utf-8")

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(
    name: str = "train",
    log_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
    filename: str = "train.log",
    rotation: Literal["none", "size", "time"] = "none",
) -> logging.Logger:
    dir_path: Optional[Path]
    if log_dir is None:
        dir_path = None
    else:
        dir_path = Path(log_dir)
    config = LoggingConfig(
        name=name,
        level=level,
        log_dir=dir_path,
        filename=filename,
        rotation=rotation,
    )
    return configure_logger(config)
