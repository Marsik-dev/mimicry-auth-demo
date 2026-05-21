"""
Тесты слоя БД (SQLite в памяти — не требует PostgreSQL).
"""
import os
import pickle

import contextlib

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mimicry_auth_demo.db.models import Base
from mimicry_auth_demo.db import session as session_module
from mimicry_auth_demo.db import profile_repo as repo_module


@pytest.fixture(autouse=True)
def sqlite_db(monkeypatch, tmp_path):
    """Подменяем движок на SQLite in-memory (без PostgreSQL)."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)

    monkeypatch.setattr(session_module, "_engine", engine)
    monkeypatch.setattr(session_module, "_SessionFactory", SessionFactory)

    @contextlib.contextmanager
    def mock_session():
        session = SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(session_module, "get_session", mock_session)
    # profile_repo импортирует get_session из session модуля — патчим там же
    monkeypatch.setattr(repo_module, "get_session", mock_session)


def _make_container():
    """Создаём минимальный NBKContainer для тестов."""
    from npbk import NBKContainer, NPBKTrainer, TrainingSet

    rng = np.random.default_rng(0)
    own = rng.normal(1.0, 0.2, (15, 32))
    other = rng.normal(0.0, 0.5, (80, 32))
    result = NPBKTrainer().train(TrainingSet(own, other))
    return NBKContainer.from_result(result, feature_dim=32)


def test_save_load_profile():
    from mimicry_auth_demo.db.profile_repo import ProfileRepo

    repo = ProfileRepo()
    container = _make_container()
    repo.save_profile("user1", container, emotion="happy")
    assert repo.profile_exists("user1")
    loaded = repo.load_profile("user1")
    assert np.array_equal(loaded.reference_code, container.reference_code)
    assert loaded.hamming_threshold == container.hamming_threshold


def test_list_profiles():
    from mimicry_auth_demo.db.profile_repo import ProfileRepo

    repo = ProfileRepo()
    container = _make_container()
    repo.save_profile("alice", container, emotion="happy")
    repo.save_profile("bob", container, emotion="angry")
    profiles = repo.list_profiles()
    assert "alice" in profiles
    assert "bob" in profiles


def test_delete_profile():
    from mimicry_auth_demo.db.profile_repo import ProfileRepo

    repo = ProfileRepo()
    container = _make_container()
    repo.save_profile("to_delete", container, emotion="happy")
    assert repo.profile_exists("to_delete")
    repo.delete_profile("to_delete")
    assert not repo.profile_exists("to_delete")


def test_save_load_normalizer():
    from mimicry_auth_demo.db.profile_repo import ProfileRepo
    from sklearn.preprocessing import MinMaxScaler

    repo = ProfileRepo()
    scaler = MinMaxScaler()
    rng = np.random.default_rng(0)
    data = rng.random((50, 32))
    scaler.fit(data)

    nid = repo.save_normalizer(
        emotion="happy",
        normalizer_obj=scaler,
        n_vectors=50,
        input_dim=32,
        output_dim=32,
    )
    loaded_scaler = repo.load_normalizer(nid)
    test_vec = rng.random((1, 32))
    np.testing.assert_allclose(scaler.transform(test_vec), loaded_scaler.transform(test_vec))


def test_log_auth():
    from mimicry_auth_demo.db.profile_repo import ProfileRepo

    repo = ProfileRepo()
    # Лог без привязанного профиля
    repo.log_auth(user_id=None, accepted=False, hamming=15, confidence=0.7)
