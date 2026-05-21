"""SQLAlchemy ORM models."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, Float, ForeignKey, Integer, LargeBinary, Text
from sqlalchemy.dialects import sqlite as _sqlite_dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class NormalizerRow(Base):
    """Обученный нормализатор (MinMax + PCA) для конкретной эмоции."""

    __tablename__ = "normalizers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    emotion: Mapped[str] = mapped_column(Text, nullable=False)
    n_vectors: Mapped[int] = mapped_column(Integer, nullable=False)
    input_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    output_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    scaler: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)  # pickle

    profiles: Mapped[list["Profile"]] = relationship(back_populates="normalizer")


class Profile(Base):
    """Зарегистрированный пользователь: НПБК + зашифрованная пассфраза."""

    __tablename__ = "profiles"

    user_id: Mapped[str] = mapped_column(Text, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    emotion: Mapped[str] = mapped_column(Text, nullable=False)
    feature_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    code_length: Mapped[int] = mapped_column(Integer, nullable=False)
    hamming_threshold: Mapped[int] = mapped_column(Integer, nullable=False)

    # НПБК контейнер (сериализованный)
    container: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Кузнечик: зашифрованная пассфраза пользователя
    enc_secret: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    kdf_salt: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    # Ссылка на нормализатор
    normalizer_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("normalizers.id"), nullable=True
    )
    normalizer: Mapped["NormalizerRow | None"] = relationship(back_populates="profiles")

    # Метрики качества (денормализованы для быстрого отображения)
    mean_stability: Mapped[float] = mapped_column(Float, default=0.0)
    empirical_far: Mapped[float] = mapped_column(Float, default=1.0)
    n_own_samples: Mapped[int] = mapped_column(Integer, default=0)

    auth_logs: Mapped[list["AuthLog"]] = relationship(back_populates="profile")


class AuthLog(Base):
    """Лог попыток аутентификации."""

    __tablename__ = "auth_log"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    user_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("profiles.user_id"), nullable=True
    )
    ts: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    hamming: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    profile: Mapped["Profile | None"] = relationship(back_populates="auth_logs")


class EvalRun(Base):
    """Результаты оценки FAR/FRR на датасете RAVDESS."""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    run_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    run_index: Mapped[int] = mapped_column(Integer, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    registered_actor: Mapped[str] = mapped_column(Text, nullable=False)
    registered_emotion: Mapped[str] = mapped_column(Text, nullable=False)
    test_actor: Mapped[str] = mapped_column(Text, nullable=False)
    test_emotion: Mapped[str] = mapped_column(Text, nullable=False)
    test_recording: Mapped[str] = mapped_column(Text, nullable=False)
    is_genuine: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_wrong_emotion: Mapped[bool] = mapped_column(Boolean, nullable=False)
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    hamming_distance: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    code_length: Mapped[int] = mapped_column(Integer, nullable=False)
    hamming_threshold: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
