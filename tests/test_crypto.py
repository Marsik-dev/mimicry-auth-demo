"""Тесты Кузнечик-шифрования."""
import os

import numpy as np
import pytest

from mimicry_auth_demo.crypto import decrypt_passphrase, encrypt_passphrase


@pytest.fixture
def reference_code() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=64, dtype=np.int8)


@pytest.fixture
def salt() -> bytes:
    return os.urandom(16)


def test_roundtrip(reference_code, salt):
    passphrase = "тест_пароль_2024!"
    blob = encrypt_passphrase(reference_code, passphrase, salt)
    assert decrypt_passphrase(reference_code, blob, salt) == passphrase


def test_roundtrip_ascii(reference_code, salt):
    passphrase = "my_secret_ssh_key_abc123"
    blob = encrypt_passphrase(reference_code, passphrase, salt)
    assert decrypt_passphrase(reference_code, blob, salt) == passphrase


def test_different_salt_gives_different_blob(reference_code):
    passphrase = "same_pass"
    blob1 = encrypt_passphrase(reference_code, passphrase, os.urandom(16))
    blob2 = encrypt_passphrase(reference_code, passphrase, os.urandom(16))
    assert blob1 != blob2  # random IV + different salt


def test_wrong_reference_code_raises(reference_code, salt):
    passphrase = "secret"
    blob = encrypt_passphrase(reference_code, passphrase, salt)

    rng = np.random.default_rng(99)
    wrong_code = rng.integers(0, 2, size=64, dtype=np.int8)
    with pytest.raises(ValueError, match="биометрический ключ"):
        decrypt_passphrase(wrong_code, blob, salt)


def test_wrong_salt_raises(reference_code):
    passphrase = "secret"
    salt_enc = os.urandom(16)
    salt_dec = os.urandom(16)
    blob = encrypt_passphrase(reference_code, passphrase, salt_enc)
    with pytest.raises(ValueError, match="биометрический ключ"):
        decrypt_passphrase(reference_code, blob, salt_dec)


def test_blob_has_iv_prefix(reference_code, salt):
    blob = encrypt_passphrase(reference_code, "x", salt)
    # Blob = 8 bytes IV + ciphertext
    assert len(blob) > 8


def test_short_blob_raises(reference_code, salt):
    with pytest.raises(ValueError):
        decrypt_passphrase(reference_code, b"short", salt)
