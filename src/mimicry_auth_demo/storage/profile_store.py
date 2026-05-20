from __future__ import annotations

from datetime import datetime
from pathlib import Path

from npbk import NBKContainer

from .models import ProfileInfo


class ProfileStore:
    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str) -> Path:
        return self.store_dir / f"{user_id}.nbk"

    def save_profile(self, user_id: str, container: NBKContainer) -> None:
        container.save(self._path(user_id))

    def load_profile(self, user_id: str) -> NBKContainer:
        p = self._path(user_id)
        if not p.exists():
            raise FileNotFoundError(f"Profile '{user_id}' not found")
        return NBKContainer.load(p)

    def list_profiles(self) -> list[str]:
        return sorted(p.stem for p in self.store_dir.glob("*.nbk"))

    def delete_profile(self, user_id: str) -> None:
        self._path(user_id).unlink(missing_ok=True)

    def profile_exists(self, user_id: str) -> bool:
        return self._path(user_id).exists()

    def get_info(self, user_id: str) -> ProfileInfo:
        c = self.load_profile(user_id)
        return ProfileInfo(
            user_id=user_id,
            created_at=datetime.fromisoformat(c.created_at),
            n_own_samples=c.quality.n_own_samples,
            quality_mean_stability=c.quality.mean_stability,
            predicted_far=c.quality.predicted_far,
        )
