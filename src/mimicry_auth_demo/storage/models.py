from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProfileInfo:
    user_id: str
    created_at: datetime
    n_own_samples: int
    quality_mean_stability: float
    predicted_far: float
