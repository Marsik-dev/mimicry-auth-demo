"""
MEAD adapter (Multimodal Emotional Apprisal Dataset).

Структура:
  mead/{actor}/video/front/{emotion}/level_{1,2,3}/{clip}.mp4

60 акторов, 8 эмоций, 3 уровня интенсивности.
На каждую комбинацию (актор × эмоция × уровень) — несколько клипов.
Итого: 10+ записей на актора на эмоцию → НПБК обучается без аугментации.

Рекомендуемые уровни для НПБК: level_2 и level_3 (выраженная мимика).
"""
from __future__ import annotations

from pathlib import Path

from .base import Recording

_RECOMMENDED_LEVELS = ("level_2", "level_3")


class MEADAdapter:
    EMOTIONS = (
        "neutral",
        "angry",
        "contempt",
        "disgusted",
        "fear",
        "happy",
        "sad",
        "surprised",
    )

    def __init__(self, root: Path) -> None:
        self.root = root / "mead"

    @property
    def available(self) -> bool:
        return self.root.is_dir()

    def subjects(self) -> list[str]:
        return sorted(
            d.name
            for d in self.root.iterdir()
            if d.is_dir() and (d / "video" / "front").is_dir()
        )

    def emotions(self) -> list[str]:
        return list(self.EMOTIONS)

    def recordings(
        self,
        subject: str,
        emotion: str,
        levels: tuple[str, ...] = _RECOMMENDED_LEVELS,
    ) -> list[Recording]:
        """
        Args:
            subject: 'M003', 'W009' и т.д.
            emotion: одна из MEAD.EMOTIONS
            levels:  ('level_1','level_2','level_3') или подмножество.
                     По умолчанию level_2 + level_3 (рекомендуется для НПБК).
        """
        front_dir = self.root / subject / "video" / "front" / emotion
        if not front_dir.is_dir():
            return []

        result = []
        for level in levels:
            level_dir = front_dir / level
            if not level_dir.is_dir():
                continue
            for vid in sorted(level_dir.glob("*.mp4")) + sorted(level_dir.glob("*.avi")):
                result.append(
                    Recording(
                        path=vid,
                        subject_id=subject,
                        emotion=emotion,
                        label=f"{subject}/{emotion}/{level}/{vid.name}",
                    )
                )
        return result
