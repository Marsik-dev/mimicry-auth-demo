"""
Шифрование пользовательского секрета по схеме:
  reference_code (НПБК) → KDF (PBKDF2-HMAC-SHA256) → 32-байтовый ключ
  Кузнечик-CTR (ГОСТ Р 34.12-2015): encrypt(key, passphrase) → ciphertext

Кузнечик требует:
  - ключ 256 бит (32 байта)
  - IV  64 бита (8 байт) в режиме CTR

Формат ciphertext-блоба (передаётся в БД):
  [8 байт IV][остаток — шифротекст]
"""
from __future__ import annotations

import hashlib
import os

import gostcrypto
import numpy as np


_KDF_ITERATIONS = 100_000
_KDF_DKLEN = 32        # 256 бит для Кузнечика
_CTR_IV_LEN = 8        # 64-битный IV для режима CTR Кузнечика
_ENCODING = "utf-8"
_MAGIC = b"NPBK\x00"  # маркер в начале plaintext для детекции неверного ключа


def _derive_key(reference_code: np.ndarray, salt: bytes) -> bytes:
    """PBKDF2-HMAC-SHA256 из байтов reference_code."""
    code_bytes = np.packbits(reference_code.astype(np.uint8)).tobytes()
    return hashlib.pbkdf2_hmac(
        "sha256",
        code_bytes,
        salt,
        _KDF_ITERATIONS,
        dklen=_KDF_DKLEN,
    )


def encrypt_passphrase(
    reference_code: np.ndarray,
    passphrase: str,
    salt: bytes,
) -> bytes:
    """
    Зашифровать пассфразу ключом, производным от reference_code НПБК.

    Args:
        reference_code: бинарный код из NBKContainer.reference_code (K,)
        passphrase:     строка пользователя (пароль/секрет)
        salt:           случайная соль (16 байт, хранится в БД рядом с ciphertext)

    Returns:
        blob = IV (8 байт) || ciphertext
    """
    key = _derive_key(reference_code, salt)
    iv = os.urandom(_CTR_IV_LEN)
    plaintext = bytearray(_MAGIC + passphrase.encode(_ENCODING))

    cipher = gostcrypto.gostcipher.new(
        "kuznechik",
        bytearray(key),
        gostcrypto.gostcipher.MODE_CTR,
        init_vect=bytearray(iv),
    )
    ciphertext = bytes(cipher.encrypt(plaintext))
    return iv + ciphertext


def decrypt_passphrase(
    reference_code: np.ndarray,
    ciphertext_blob: bytes,
    salt: bytes,
) -> str:
    """
    Расшифровать пассфразу тем же ключом из reference_code НПБК.

    Args:
        reference_code:  тот же бинарный код, что при шифровании
        ciphertext_blob: blob из БД (IV || ciphertext)
        salt:            та же соль, что при шифровании

    Returns:
        исходная пассфраза пользователя
    """
    if len(ciphertext_blob) <= _CTR_IV_LEN:
        raise ValueError("ciphertext_blob слишком короткий")

    iv = ciphertext_blob[:_CTR_IV_LEN]
    ciphertext = bytearray(ciphertext_blob[_CTR_IV_LEN:])

    key = _derive_key(reference_code, salt)
    cipher = gostcrypto.gostcipher.new(
        "kuznechik",
        bytearray(key),
        gostcrypto.gostcipher.MODE_CTR,
        init_vect=bytearray(iv),
    )
    plaintext = bytes(cipher.decrypt(ciphertext))
    if not plaintext.startswith(_MAGIC):
        raise ValueError(
            "Расшифрование не удалось: неверный биометрический ключ (НПБК не принял)"
        )
    return plaintext[len(_MAGIC):].decode(_ENCODING)
