"""
CREMA-D adapter (Crowd-sourced Emotional Multimodal Actors Dataset).

Структура: cremad/VideoFlash/{actor_id}_{sentence_id}_{emotion}_{level}.flv
  Или:     cremad/VideoMP4/{actor_id}_{sentence_id}_{emotion}_{level}.mp4

91 актор, 6 эмоций, 12 предложений → 12 записей/актора/эмоцию (без аугментации).
Лицензия: ODbL. GitHub: CheyneyComputerScience/CREMA-D

Коды эмоций в именах файлов:
  ANG=angry, DIS=disgust, FEA=fear, HAP=happy, NEU=neutral, SAD=sad
"""
from __future__ import annotations

from pathlib import Path

from .base import Recording

_EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


class CREMADAdapter:
    EMOTIONS = sorted(set(_EMOTION_MAP.values()))

    def __init__(self, root: Path) -> None:
        self.root = root / "cremad"
        # Поддерживаем как .flv (VideoFlash), так и .mp4 (VideoMP4)
        self._video_dir: Path | None = None
        for sub in ("VideoMP4", "VideoFlash", ""):
            candidate = self.root / sub if sub else self.root
            if candidate.is_dir():
                self._video_dir = candidate
                break

    @property
    def available(self) -> bool:
        if self._video_dir is None or not self._video_dir.is_dir():
            return False
        return any(
            next(self._video_dir.glob(f"*.{ext}"), None) is not None
            for ext in ("mp4", "flv", "avi")
        )

    def subjects(self) -> list[str]:
        if not self.available or self._video_dir is None:
            return []
        ids: set[str] = set()
        for f in self._video_dir.glob("*"):
            if f.suffix in (".mp4", ".flv", ".avi"):
                ids.add(f.stem.split("_")[0])
        return sorted(ids)

    def emotions(self) -> list[str]:
        return self.EMOTIONS

    def recordings(self, subject: str, emotion: str) -> list[Recording]:
        if not self.available or self._video_dir is None:
            return []
        emo_code = next(
            (k for k, v in _EMOTION_MAP.items() if v == emotion), None
        )
        if emo_code is None:
            return []

        result = []
        for ext in ("*.mp4", "*.flv", "*.avi"):
            for vid in sorted(self._video_dir.glob(ext)):
                parts = vid.stem.split("_")
                if len(parts) < 3:
                    continue
                if parts[0] != subject:
                    continue
                if parts[2] != emo_code:
                    continue
                result.append(
                    Recording(
                        path=vid,
                        subject_id=subject,
                        emotion=emotion,
                        label=vid.name,
                    )
                )
        return result
