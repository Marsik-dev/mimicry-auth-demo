"""
RAVDESS adapter.

Структура: ravdess/videos/Actor_NN/{modality}-{vocal}-{emotion}-{intensity}-{stmt}-{rep}-{actor}.mp4

Имена полей (1-индексированные):
  [0] modality:     01=AV, 02=video-only
  [1] vocal_channel: 01=speech, 02=song
  [2] emotion:      01=neutral,02=calm,03=happy,04=sad,05=angry,06=fear,07=disgust,08=surprise
  [3] intensity:    01=normal, 02=strong
  [4] statement:    01 или 02
  [5] repetition:   01 или 02
  [6] actor:        01-24

Для обучения НПБК берём ТОЛЬКО speech (vocal_channel=01) —
это даёт 8 записей на актора на эмоцию (2 intensity × 2 stmt × 2 rep).
Музыкальные (song) записи не используются: мимика при пении другая.

FIX: старый код фильтровал parts[0]=="02" (video-only), которых в наборе нет.
     Правильный фильтр — parts[1]=="01" (speech).
"""
from __future__ import annotations

from pathlib import Path

from .base import Recording


class RAVDESSAdapter:
    EMOTIONS = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprise",
    }
    # Эмоции, доступные только в speech (нет song-версии)
    SPEECH_ONLY = {"neutral", "calm"}

    def __init__(self, root: Path) -> None:
        self.root = root / "ravdess" / "videos"

    @property
    def available(self) -> bool:
        return self.root.is_dir()

    def subjects(self) -> list[str]:
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())

    def emotions(self) -> list[str]:
        return sorted(set(self.EMOTIONS.values()))

    def recordings(
        self,
        subject: str,
        emotion: str,
        speech_only: bool = True,
    ) -> list[Recording]:
        """
        Список записей для данного субъекта и эмоции.

        Args:
            subject:     'Actor_01' и т.д.
            emotion:     'happy', 'angry', ...
            speech_only: если True (по умолчанию) — только vocal_channel=01.
                         8 записей/актора/эмоцию. Рекомендуется для НПБК.
        """
        subj_dir = self.root / subject
        if not subj_dir.is_dir():
            return []
        emo_code = next(
            (k for k, v in self.EMOTIONS.items() if v == emotion), None
        )
        if emo_code is None:
            return []

        result = []
        for mp4 in sorted(subj_dir.glob("*.mp4")):
            parts = mp4.stem.split("-")
            if len(parts) != 7:
                continue
            if parts[2] != emo_code:
                continue
            if speech_only and parts[1] != "01":
                continue
            result.append(
                Recording(
                    path=mp4,
                    subject_id=subject,
                    emotion=emotion,
                    label=mp4.name,
                )
            )
        return result
