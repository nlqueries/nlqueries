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
import string
import tempfile
import time
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


def chunk_text_purpose(source_id: str) -> bytes:
    """The associated data for a chunk of *source_id*'s text.

    Bound to the document identifier as well as to the kind of value: chunk
    payloads are not signed, so without it a sealed text given another
    document's ``source_id`` would open as that document's. A point copied
    whole, ``source_id`` and all, still opens: the collection is not part of
    it, so that a snapshot restored under another name still reads.
    """
    return CHUNK_TEXT + b"\x00" + source_id.encode("utf-8")


KEY_ENV = "NLQ_CACHE_ENCRYPTION_KEY"
KEY_FILE_ENV = "NLQ_CACHE_ENCRYPTION_KEY_FILE"

_NONCE_BYTES = 12
_HKDF_INFO = b"nlqueries at-rest encryption v1"

#: Kinds of value already reported as unreadable in this process, so a cache or
#: a collection full of them warns once rather than on every read.
_warned: set[bytes] = set()

#: A configured key shorter than this is warned about. The AES key is derived
#: with HKDF, which has no work factor, so a guessable key is guessable offline
#: against a copy of Qdrant's data. A generated key is 64 characters.
_SHORT_KEY = 32
_warned_short = False

#: A generated key: 32 random bytes as hex. A stored file holding anything else
#: is not taken for one, since it may be a key another process is still writing.
_GENERATED_KEY_LENGTH = 64

#: How long a process that lost the race to create the key waits for the
#: winner's to be complete.
_KEY_WAIT_SECONDS = 2.0

#: Set once a wait for an incomplete key has run out, so that a file left half
#: written by a process that died is not waited on again by every later write.
_gave_up_waiting = False


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
        return _checked(value.encode("utf-8"))
    key_file = os.getenv(KEY_FILE_ENV, "").strip()
    if not key_file:
        return None
    try:
        material = Path(key_file).read_bytes().strip()
    except OSError as exc:
        raise RuntimeError(f"{KEY_FILE_ENV} is set to {key_file}, which cannot be read") from exc
    if not material:
        raise RuntimeError(f"{KEY_FILE_ENV} is set to {key_file}, which is empty")
    return _checked(material)


def _checked(material: bytes) -> bytes:
    """*material*, after warning once if it is short enough to guess."""
    global _warned_short  # noqa: PLW0603 - once per process
    if len(material) < _SHORT_KEY and not _warned_short:
        _warned_short = True
        logger.warning(
            "The configured encryption key is %d characters long. Use a random one of at "
            "least %d, such as the output of `openssl rand -hex 32`: anyone with a copy of "
            "Qdrant's data can try guesses against it offline.",
            len(material),
            _SHORT_KEY,
        )
    return material


def _stored() -> bytes | None:
    """The generated key, or None if there is no complete one this process can read.

    Unreadable counts as none -- a key written by another user at 0600, say --
    so that reading reports it as a missing key and writing fails with the
    message naming the settings, rather than either raising the bare error.
    So does anything but a whole generated key: on a filesystem without hard
    links the key is written in place, and a key read half-written would seal
    text no other process can open.
    """
    return _key_file()[0]


def _key_file() -> tuple[bytes | None, bool]:
    """The stored key, if whole, and whether the file may still be being written.

    Only a readable file of fewer hex characters than a key may be one another
    process is writing. One that cannot be read, or holds anything else, will
    not become a key by waiting.
    """
    try:
        material = key_path().read_bytes().strip()
    except OSError:
        return None, False
    hex_only = all(chr(b) in string.hexdigits for b in material)
    if hex_only and len(material) == _GENERATED_KEY_LENGTH:
        return material, False
    return None, hex_only and len(material) < _GENERATED_KEY_LENGTH


def _generated() -> bytes:
    """A new key, written to :func:`key_path` -- or the one another process wrote first.

    Written to a temporary file at 0600 and then linked into place, which fails
    if a key is already there: two processes starting together end up with the
    same key, where a rename would let the second replace the first and leave
    whatever the first sealed unreadable. Not held in memory when it cannot be
    written, unlike the signing key: a document sealed with a key that is gone
    after a restart is lost until it is ingested again.
    """
    path = key_path()
    key = secrets.token_hex(32).encode("ascii")
    try:
        # Created if missing, and otherwise left as it is: the key's own file
        # is 0600, and the directory is the whole state directory, whose mode
        # is not this feature's to change.
        path.parent.mkdir(parents=True, exist_ok=True)
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
                    # instead: still one winner, though a reader can see the
                    # file before its 64 bytes are in it, which `_stored`
                    # does not take for a key.
                    # Created at 0600, not chmodded after: the key goes in
                    # before a chmod would, readable to others meanwhile.
                    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
                    with os.fdopen(os.open(path, flags, 0o600), "wb") as fh:
                        fh.write(key)
                        # On disk before it is closed, as the linked file is: a
                        # file left empty by a power cut would stop every write.
                        fh.flush()
                        os.fsync(fh.fileno())
            except FileExistsError:
                return _winners_key()
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


def _winners_key() -> bytes:
    """The key another process created first, once it is complete.

    Waited for only while the file may still be being written, and only until
    one wait has run out in this process: a file that cannot be read, holds
    something else, or was left half written will not change, and every write
    would otherwise wait the full time for it.
    """
    global _gave_up_waiting  # noqa: PLW0603 - once per process
    deadline = time.monotonic() + _KEY_WAIT_SECONDS
    while True:
        existing, in_progress = _key_file()
        if existing is not None:
            return existing
        if not in_progress or _gave_up_waiting or time.monotonic() >= deadline:
            if in_progress:
                _gave_up_waiting = True
            raise FileExistsError(f"{key_path()} exists but holds no complete key")
        time.sleep(0.05)


@functools.lru_cache(maxsize=4)
def _cipher(material: bytes) -> AESGCM:
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_HKDF_INFO).derive(material)
    return AESGCM(key)


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
            purpose, f"no key is configured ({KEY_ENV}, {KEY_FILE_ENV}) or readable at {key_path()}"
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
