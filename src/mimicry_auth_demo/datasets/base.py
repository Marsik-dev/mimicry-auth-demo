from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Recording:
    path: Path
    subject_id: str
    emotion: str
    label: str  # отображаемое имя в UI
