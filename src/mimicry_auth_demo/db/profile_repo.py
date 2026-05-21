"""
ProfileRepo — замена файлового ProfileStore на PostgreSQL.

Хранит:
  - NBKContainer (bytes) в поле container
  - зашифрованную пассфразу (enc_secret + kdf_salt)
  - ссылку на нормализатор
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from npbk import NBKContainer

from .models import AuthLog, NormalizerRow, Profile
from .session import get_session


@dataclass
class ProfileInfo:
    user_id: str
    created_at: datetime
    emotion: str
    n_own_samples: int
    mean_stability: float
    empirical_far: float
    code_length: int
    hamming_threshold: int


class ProfileRepo:
    """CRUD-операции с профилями и нормализаторами."""

    # ------------------------------------------------------------------ profiles

    def save_profile(
        self,
        user_id: str,
        container: NBKContainer,
        emotion: str,
        enc_secret: bytes | None = None,
        kdf_salt: bytes | None = None,
        normalizer_id: int | None = None,
    ) -> None:
        with get_session() as session:
            row = session.get(Profile, user_id)
            if row is None:
                row = Profile(user_id=user_id)
                session.add(row)

            row.emotion = emotion
            row.feature_dim = container.feature_dim
            row.code_length = int(len(container.reference_code))
            row.hamming_threshold = container.hamming_threshold
            row.container = container.to_bytes()
            row.enc_secret = enc_secret
            row.kdf_salt = kdf_salt
            row.normalizer_id = normalizer_id
            row.mean_stability = float(container.quality.mean_stability)
            row.empirical_far = float(container.quality.predicted_far)
            row.n_own_samples = container.quality.n_own_samples

            session.commit()

    def load_profile(self, user_id: str) -> NBKContainer:
        with get_session() as session:
            row = session.get(Profile, user_id)
            if row is None:
                raise KeyError(f"Profile not found: {user_id!r}")
            return NBKContainer.from_bytes(row.container)

    def load_profile_row(self, user_id: str) -> Profile:
        with get_session() as session:
            row = session.get(Profile, user_id)
            if row is None:
                raise KeyError(f"Profile not found: {user_id!r}")
            session.expunge(row)
            return row

    def list_profiles(self) -> list[str]:
        with get_session() as session:
            rows = session.query(Profile.user_id).order_by(Profile.created_at).all()
            return [r[0] for r in rows]

    def list_profile_infos(self) -> list[ProfileInfo]:
        with get_session() as session:
            rows = session.query(Profile).order_by(Profile.created_at).all()
            return [
                ProfileInfo(
                    user_id=r.user_id,
                    created_at=r.created_at,
                    emotion=r.emotion,
                    n_own_samples=r.n_own_samples,
                    mean_stability=r.mean_stability,
                    empirical_far=r.empirical_far,
                    code_length=r.code_length,
                    hamming_threshold=r.hamming_threshold,
                )
                for r in rows
            ]

    def profile_exists(self, user_id: str) -> bool:
        with get_session() as session:
            return session.get(Profile, user_id) is not None

    def delete_profile(self, user_id: str) -> None:
        with get_session() as session:
            row = session.get(Profile, user_id)
            if row:
                session.delete(row)
                session.commit()

    # ---------------------------------------------------------------- normalizers

    def save_normalizer(
        self,
        emotion: str,
        normalizer_obj: object,
        n_vectors: int,
        input_dim: int,
        output_dim: int,
    ) -> int:
        """Сохранить нормализатор, вернуть его id."""
        data = pickle.dumps(normalizer_obj)
        with get_session() as session:
            row = NormalizerRow(
                emotion=emotion,
                n_vectors=n_vectors,
                input_dim=input_dim,
                output_dim=output_dim,
                scaler=data,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def load_normalizer(self, normalizer_id: int) -> object:
        with get_session() as session:
            row = session.get(NormalizerRow, normalizer_id)
            if row is None:
                raise KeyError(f"Normalizer not found: {normalizer_id}")
            return pickle.loads(row.scaler)  # noqa: S301

    def list_normalizers(self, emotion: str | None = None) -> list[NormalizerRow]:
        with get_session() as session:
            q = session.query(NormalizerRow)
            if emotion:
                q = q.filter(NormalizerRow.emotion == emotion)
            rows = q.order_by(NormalizerRow.created_at.desc()).all()
            for r in rows:
                session.expunge(r)
            return rows

    def latest_normalizer_id(self, emotion: str) -> int | None:
        with get_session() as session:
            row = (
                session.query(NormalizerRow)
                .filter(NormalizerRow.emotion == emotion)
                .order_by(NormalizerRow.created_at.desc())
                .first()
            )
            return row.id if row else None

    # ------------------------------------------------------------------- auth log

    def log_auth(
        self,
        user_id: str | None,
        accepted: bool,
        hamming: int | None = None,
        confidence: float | None = None,
    ) -> None:
        with get_session() as session:
            session.add(
                AuthLog(
                    user_id=user_id,
                    accepted=accepted,
                    hamming=hamming,
                    confidence=confidence,
                )
            )
            session.commit()
