"""
OULU-CASIA adapter.

Структура: oulu/Subject_NN/{Emotion}/*.avi (или *.mp4)
"""
from __future__ import annotations

from pathlib import Path

from .base import Recording


class OULUAdapter:
    def __init__(self, root: Path) -> None:
        self.root = root / "oulu"

    @property
    def available(self) -> bool:
        return self.root.is_dir() and any(self.root.glob("Subject_*"))

    def subjects(self) -> list[str]:
        return sorted(d.name for d in self.root.glob("Subject_*") if d.is_dir())

    def emotions(self) -> list[str]:
        emotions: set[str] = set()
        for subj in self.root.glob("Subject_*"):
            for emo in subj.iterdir():
                if emo.is_dir():
                    emotions.add(emo.name)
        return sorted(emotions)

    def recordings(self, subject: str, emotion: str) -> list[Recording]:
        emo_dir = self.root / subject / emotion
        if not emo_dir.is_dir():
            return []
        result = []
        for vid in sorted(emo_dir.glob("*.avi")) + sorted(emo_dir.glob("*.mp4")):
            result.append(
                Recording(
                    path=vid,
                    subject_id=subject,
                    emotion=emotion,
                    label=vid.name,
                )
            )
        return result
