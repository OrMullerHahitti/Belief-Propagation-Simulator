"""CSV backup helpers for AAAI experiment scripts."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


def backup_existing_csvs(
    data_dir: Path,
    *,
    label: str = "csv_backup",
    backup_root: Path | None = None,
) -> Path | None:
    """Copy existing CSV files from ``data_dir`` into a timestamped backup folder.

    Returns the backup directory when files were copied, or ``None`` when there
    are no current CSVs to protect.
    """
    data_dir = Path(data_dir)
    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        return None

    root = Path(backup_root) if backup_root is not None else data_dir.parent / "backups"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = root / f"{label}_{timestamp}"
    suffix = 2
    while backup_dir.exists():
        backup_dir = root / f"{label}_{timestamp}_{suffix}"
        suffix += 1

    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in csv_paths:
        shutil.copy2(path, backup_dir / path.name)
    return backup_dir
