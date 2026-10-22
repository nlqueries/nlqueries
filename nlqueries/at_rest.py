# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.at_rest
~~~~~~~~~~~~~~~~~
Optional encryption of the text this application keeps in Qdrant: a cached
answer, and a document chunk's text.

Qdrant holds what the database and the documents said, in the clear, for anyone
who can read its storage or its API. With :data:`~nlqueries.config.CACHE_ENCRYPTION`
on, those two fields are sealed with AES-256-GCM before they are stored. Vectors,
questions, SQL and the other payload fields are not: they are what the store
searches and filters on, or what the cache verifies.

A sealed value is a string, so it takes the place of the plaintext in the same
payload key and nothing that reads payloads needs to know about new keys::

    nlqenc:v1:<urlsafe base64 of a 12-byte nonce, then the ciphertext and tag>

The setting controls writing only. Reading takes either kind: plaintext is
returned as it is, so documents ingested before the setting was turned on keep
answering, and a sealed value is opened whenever a key is available, whatever
the setting. One that cannot be opened (no key, another key, damaged) reads as
``None``, which the cache treats as a miss and the document store skips.

The key comes from ``NLQ_CACHE_ENCRYPTION_KEY``, from the file
``NLQ_CACHE_ENCRYPTION_KEY_FILE`` names, or is generated in the state directory.
Any string serves; the AES key is derived from it with HKDF-SHA256. A generated
key is per state directory, so processes that share a Qdrant but not a state
directory must be given the same key, as for the cache signing key.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import functools
import logging
import os
import secrets
import tempfile
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger(__name__)

#: Starts every sealed value. The version is the format: the derivation, the
#: cipher and the layout after the marker.
MARKER = "nlqenc:v1:"

#: Associated data for each kind of value, so a sealed answer cannot be moved
#: into a chunk's text, or the reverse, and still open.
CACHE_ANSWER = b"nlqueries cache answer"
CHUNK_TEXT = b"nlqueries document chunk text"

KEY_ENV = "NLQ_CACHE_ENCRYPTION_KEY"
KEY_FILE_ENV = "NLQ_CACHE_ENCRYPTION_KEY_FILE"

_NONCE_BYTES = 12
_HKDF_INFO = b"nlqueries at-rest encryption v1"

#: Kinds of value already reported as unreadable in this process, so a cache or
#: a collection full of them warns once rather than on every read.
_warned: set[bytes] = set()


def key_path() -> Path:
    """Where a generated key is kept: the state directory, beside the signing key."""
    from nlqueries import config  # noqa: PLC0415

    return config.STATE_DIR / "cache_encryption_key"


def _configured() -> bytes | None:
    """The key from the environment or a mounted file, if one is configured.

    A file that is configured but cannot be read, or is empty, is a deployment
    error and raises: falling back to a generated key would seal with a key no
    other process holds, and read nothing sealed by the right one.
    """
    value = os.getenv(KEY_ENV, "").strip()
    if value:
        return value.encode("utf-8")
    key_file = os.getenv(KEY_FILE_ENV, "").strip()
    if not key_file:
        return None
    try:
        material = Path(key_file).read_bytes().strip()
    except OSError as exc:
        raise RuntimeError(f"{KEY_FILE_ENV} is set to {key_file}, which cannot be read") from exc
    if not material:
        raise RuntimeError(f"{KEY_FILE_ENV} is set to {key_file}, which is empty")
    return material


def _stored() -> bytes | None:
    path = key_path()
    try:
        material = path.read_bytes().strip()
    except FileNotFoundError:
        return None
    return material or None


def _generated() -> bytes:
    """A new key, written to :func:`key_path` -- or the one another process wrote first.

    Written to a temporary file at 0600 and then linked into place, which fails
    if a key is already there: two processes starting together end up with the
    same key, where a rename would let the second replace the first and leave
    whatever the first sealed unreadable. Not held in memory when it cannot be
    written, unlike the signing key: a document sealed with a key that is gone
    after a restart is lost until it is ingested again.
    """
    from nlqueries.state_files import private_dir  # noqa: PLC0415

    path = key_path()
    key = secrets.token_hex(32).encode("ascii")
    try:
        private_dir(path.parent)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(key)
                fh.flush()
                os.fsync(fh.fileno())
            try:
                try:
                    os.link(tmp_name, path)
                except FileExistsError:
                    raise
                except OSError:
                    # A filesystem without hard links. Created exclusively
                    # instead: still one winner, though a reader could see the
                    # file before its 64 bytes are in it.
                    with open(path, "xb") as fh:
                        fh.write(key)
                    with contextlib.suppress(OSError):
                        path.chmod(0o600)
            except FileExistsError:
                existing = _stored()
                if existing is None:
                    raise
                return existing
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
    except OSError as exc:
        raise RuntimeError(
            f"NLQ_CACHE_ENCRYPTION is on, and no key could be written to {path} ({exc}). "
            f"Set {KEY_ENV} or {KEY_FILE_ENV}."
        ) from exc
    logger.info(
        "Generated an encryption key for cached answers and document text at %s. Set %s "
        "or %s to share one key across processes that share a Qdrant.",
        path,
        KEY_ENV,
        KEY_FILE_ENV,
    )
    return key


@functools.lru_cache(maxsize=4)
def _cipher(material: bytes) -> AESGCM:
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_HKDF_INFO).derive(material)
    return AESGCM(key)


def is_sealed(value: object) -> bool:
    return isinstance(value, str) and value.startswith(MARKER)


def seal(text: str, purpose: bytes) -> str:
    """*text* sealed for *purpose* when encryption is on; *text* unchanged when it is off."""
    from nlqueries import config  # noqa: PLC0415

    if not config.CACHE_ENCRYPTION:
        return text
    material = _configured() or _stored() or _generated()
    nonce = os.urandom(_NONCE_BYTES)
    sealed = _cipher(material).encrypt(nonce, text.encode("utf-8"), purpose)
    return MARKER + base64.urlsafe_b64encode(nonce + sealed).decode("ascii")


def unseal(value: str, purpose: bytes) -> str | None:
    """The plaintext of *value*: itself if it was never sealed, ``None`` if it cannot be opened."""
    if not value.startswith(MARKER):
        return value
    try:
        material = _configured() or _stored()
    except RuntimeError as exc:
        _report(purpose, str(exc))
        return None
    if material is None:
        _report(
            purpose, f"no key is configured ({KEY_ENV}, {KEY_FILE_ENV}) or stored at {key_path()}"
        )
        return None
    try:
        raw = base64.urlsafe_b64decode(value[len(MARKER) :].encode("ascii"))
        if len(raw) <= _NONCE_BYTES:
            raise ValueError("too short")
        plain = _cipher(material).decrypt(raw[:_NONCE_BYTES], raw[_NONCE_BYTES:], purpose)
        return plain.decode("utf-8")
    except (InvalidTag, ValueError, binascii.Error, UnicodeError):
        _report(purpose, "it was sealed with another key, or is damaged")
        return None


def _report(purpose: bytes, why: str) -> None:
    if purpose in _warned:
        logger.debug("An encrypted %s could not be read: %s", purpose.decode(), why)
        return
    _warned.add(purpose)
    logger.warning(
        "An encrypted %s could not be read: %s. Such entries are treated as absent; this "
        "is reported once per process.",
        purpose.decode(),
        why,
    )
