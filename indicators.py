"""Append-only CSV writer used by every dataset exporter. Never overwrites data:
if the target file already exists, it is copied into outputs/backups/ (timestamped)
before the new rows are appended, and the actual write happens via a temp-file +
atomic rename so a crash mid-write can't corrupt the target file.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.logger import get_logger

logger = get_logger("datasets.updater")


def _backup(target_path: Path, backups_dir: Path) -> None:
    backups_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    backup_path = backups_dir / f"{target_path.stem}_{timestamp}{target_path.suffix}"
    shutil.copy2(target_path, backup_path)
    logger.info("Backed up %s -> %s", target_path, backup_path)


def append_rows(target_path: Path, new_rows: pd.DataFrame, columns: list[str], backups_dir: Path) -> None:
    """Reindex `new_rows` to `columns` and append them to `target_path`, backing
    up the prior file first. Creates `target_path` (with header) if it doesn't
    exist yet.
    """
    new_rows = new_rows.reindex(columns=columns)

    if target_path.exists():
        _backup(target_path, backups_dir)
        existing = pd.read_csv(target_path)
        existing = existing.reindex(columns=columns)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_rows

    fd, tmp_path_str = tempfile.mkstemp(dir=str(target_path.parent), suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_path_str)
    try:
        combined.to_csv(tmp_path, index=False)
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    logger.info("Appended %d row(s) to %s (total %d)", len(new_rows), target_path, len(combined))
