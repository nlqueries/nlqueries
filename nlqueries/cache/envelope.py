# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.cache.envelope
~~~~~~~~~~~~~~~~~~~~~~~~
Signing and verification of semantic-cache entries.

A cache entry carries SQL that is executed on a hit, with no model between the
entry and the database. The SQL policy establishes that a statement is safe to
run; it does not establish that NLQueries produced it. Anything able to write to
Qdrant could therefore store an ordinary ``SELECT`` against a table the question
did not mention and have it executed (SEC-09).

An entry is signed with HMAC-SHA256 over its own contents and the context it was
produced in: agent, connector, dialect, schema fingerprint and policy version.
Verification recomputes the tag, so an entry that was altered, or that is
replayed against a different agent, connector, dialect or schema, does not
verify. An unsigned entry does not verify either, which makes entries written
before this change cache misses rather than trusted input.

The key is held outside Qdrant. An attacker with write access to the vector
store but not to the key cannot produce a tag that verifies.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import os
import secrets
import stat
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Incremented when the signed representation changes. Entries carrying a
#: different version do not verify, so a change to the format expires the cache
#: rather than producing tags that disagree.
ENVELOPE_VERSION = "1"

#: Payload keys holding the signature and its version.
SIGNATURE_KEY = "signature"
VERSION_KEY = "envelope_version"

#: Payload keys covered by the signature. Listed rather than derived from the
#: payload, so a key added later is not silently signed or silently ignored.
#: ``hit_count`` is excluded: it is updated in place on every hit and is not
#: part of what the entry asserts.
SIGNED_FIELDS = (
    "question",
    "resolved_question",
    "agent_type",
    "answer",
    "sql",
    "created_at",
    "kind",
)

#: Keys the cache writes itself. Everything else in a payload came from the
#: caller's cache context. Mirrors `semantic_cache._RESERVED_PAYLOAD_KEYS`,
#: which a test pins to this; duplicated rather than imported to avoid a cycle.
#: `_ctx` is the context digest `semantic_cache` writes so the context can be
#: matched inside Qdrant. Reserved here too, or it would count as context, be
#: signed as part of it, and change the very digest it is derived from.
_RESERVED_PAYLOAD_KEYS_FOR_SIGNING: frozenset[str] = frozenset(
    {*SIGNED_FIELDS, "hit_count", "_ctx", SIGNATURE_KEY, VERSION_KEY}
)


_KEY_ENV = "NLQ_CACHE_SIGNING_KEY"

#: A file holding the key, for deployments that mount secrets rather than pass
#: them in the environment.
_KEY_FILE_ENV = "NLQ_CACHE_SIGNING_KEY_FILE"

#: Used when the key cannot be persisted. Held for the life of the process so
#: that a payload signed by this process still verifies in it.
_ephemeral_key: bytes | None = None

#: The shortest key accepted, in bytes. The generated key is 64 hex characters.
#: An empty key file (a full disk, a crash between creating and writing it, an
#: operator's `touch`) used to be accepted, and every entry signed with it
#: verified for anyone who knew the algorithm.
MIN_KEY_BYTES = 32

#: Set once a key problem has been logged on the read path.
_KEY_ERROR_LOGGED: list[bool] = []


class SigningKeyError(RuntimeError):
    """The configured or stored signing key is unusable."""


def _checked(material: bytes, where: str) -> bytes:
    if len(material) < MIN_KEY_BYTES:
        raise SigningKeyError(
            f"The semantic-cache signing key from {where} is {len(material)} bytes; at "
            f"least {MIN_KEY_BYTES} are required. Generate one with: openssl rand -hex 32"
        )
    return material


@dataclass(frozen=True)
class CacheBinding:
    """The context an entry was produced in.

    An entry that verifies under one binding does not verify under another, so a
    cached answer cannot be replayed against a different agent, connector,
    dialect or schema.
    """

    agent_id: str
    connector_fingerprint: str
    dialect: str
    schema_fingerprint: str
    policy_version: str

    def canonical(self) -> str:
        return json.dumps(
            {
                "agent_id": self.agent_id,
                "connector_fingerprint": self.connector_fingerprint,
                "dialect": self.dialect.lower(),
                "schema_fingerprint": self.schema_fingerprint,
                "policy_version": self.policy_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def _key_path() -> Path:
    """Where the signing key lives when it is not supplied by env or a mount.

    Under `config.STATE_DIR`, which is what this function's own docstring above
    already called "the state directory" -- it just had no name in code, so a
    deployment that relocated everything else still had this one pinned to a
    home directory it might not be able to write.
    """
    from nlqueries import config  # noqa: PLC0415

    return config.STATE_DIR / "cache_signing_key"


def _publish_key(path: Path, key: bytes) -> bytes | None:
    """Write *key* to *path* atomically; the key already there if one won the race.

    Written whole to a private temporary file first (created with ``O_EXCL`` and
    mode 0600, so it is never readable by others, not even briefly), then linked
    into place, which fails rather than overwrites if another process got there
    first. A reader therefore sees either no file or a complete key, never the
    empty or half-written one a crash between create and write used to leave.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(key)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return path.read_bytes().strip()
        except OSError:
            # No hard links on this filesystem (FAT and exFAT, some SMB and
            # FUSE mounts, all valid places for the state directory). A rename
            # is atomic too, so a reader still never sees a partial key; it
            # can replace a key another process published between this check
            # and the rename, a narrower window than linking leaves.
            if path.exists():
                return path.read_bytes().strip()
            os.replace(tmp, path)
        return None
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def report_key_error(exc: SigningKeyError) -> None:
    """Log, once per process, that the cache is off because of *exc*.

    For the query path, where a bad key skips the cache rather than failing the
    question: reads miss and writes are skipped, and saying so on every one of
    them would bury the message it repeats.
    """
    if not _KEY_ERROR_LOGGED:
        _KEY_ERROR_LOGGED.append(True)
        logger.error("Semantic cache disabled: %s", exc)


def check_signing_key() -> None:
    """Raise :class:`SigningKeyError` now if the signing key is unusable.

    For startup. On the read path a bad key turns every cache lookup into a
    miss, which keeps queries working but hides the problem; a server that
    refuses to start says what is wrong.
    """
    signing_key()


def signing_key() -> bytes:
    """The HMAC key: from the environment, a mounted file, or the state directory.

    Generated on first use when none is present. Generating rather than
    requiring configuration keeps caching working on upgrade; the key is still
    outside Qdrant, which is what the signature depends on.

    Every process that shares a cache must arrive at the same key. A generated
    one is written under the state directory, which a second container does not
    share, so a deployment running more than one process must set
    :data:`_KEY_ENV` or :data:`_KEY_FILE_ENV`. Where it does not, entries
    written by one process do not verify in another and are treated as misses.

    Raises :class:`SigningKeyError` for a key shorter than :data:`MIN_KEY_BYTES`,
    wherever it came from. :func:`check_signing_key` raises it at startup;
    :func:`verify` turns it into a miss so a query is still answered.
    """
    configured = os.getenv(_KEY_ENV, "").strip()
    if configured:
        return _checked(configured.encode("utf-8"), _KEY_ENV)

    key_file = os.getenv(_KEY_FILE_ENV, "").strip()
    if key_file:
        # Configured but unreadable is a deployment error. Falling back would
        # produce a key this process alone holds, and the cache would go quiet
        # rather than report the misconfiguration.
        try:
            material = Path(key_file).read_bytes().strip()
        except OSError as exc:
            raise SigningKeyError(
                f"{_KEY_FILE_ENV} is set to {key_file}, which cannot be read"
            ) from exc
        if not material:
            raise SigningKeyError(f"{_KEY_FILE_ENV} is set to {key_file}, which is empty")
        return _checked(material, key_file)

    path = _key_path()
    if path.exists():
        try:
            stored = path.read_bytes().strip()
        except OSError as exc:
            # A file another user wrote, typically: a container that created
            # it as root and now runs as someone else. The same condition as
            # an unreadable mounted key, and handled the same way.
            raise SigningKeyError(
                f"The semantic-cache signing key at {path} cannot be read"
            ) from exc
        return _checked(stored, f"{path} (delete it to generate a new one)")

    global _ephemeral_key  # noqa: PLW0603 - one key per process, by design
    if _ephemeral_key is not None:
        return _ephemeral_key

    key = secrets.token_hex(32).encode("ascii")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = _publish_key(path, key)
        if existing is not None:
            # Another process created it first; every process must share one.
            return _checked(existing, str(path))
    except OSError as exc:
        # A read-only root filesystem reaches here. Signing still works, but
        # only within this process and only until it restarts.
        _ephemeral_key = key
        logger.warning(
            "Could not write a semantic-cache signing key to %s (%s). Using a "
            "key held in memory: cache entries will not be shared with other "
            "processes and will not survive a restart. Set %s or %s to fix this.",
            path,
            exc,
            _KEY_ENV,
            _KEY_FILE_ENV,
        )
        return key

    logger.info(
        "Generated a semantic-cache signing key at %s. Set %s or %s to share one "
        "key across processes; otherwise each verifies only its own entries.",
        path,
        _KEY_ENV,
        _KEY_FILE_ENV,
    )
    return key


def _context_fields(payload: dict[str, Any]) -> dict[str, str]:
    """The caller's cache context: every key the cache does not write itself."""
    return {
        str(k): str(v) for k, v in payload.items() if k not in _RESERVED_PAYLOAD_KEYS_FOR_SIGNING
    }


def _message(payload: dict[str, Any], binding: CacheBinding) -> bytes:
    fields = {name: payload.get(name) for name in SIGNED_FIELDS}
    body = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    message = f"{ENVELOPE_VERSION}\n{binding.canonical()}\n{body}"

    # The caller's cache context is covered too, appended only when there is one
    # -- so an entry written without a context produces byte-identical output to
    # before this existed and keeps verifying across the upgrade. Only
    # context-carrying entries pay a one-off miss.
    #
    # Without this the context was the one part of a partition boundary the HMAC
    # did not cover: anyone with write access to Qdrant and no access to the key
    # could take a valid entry, change `tenant` from `a` to `b`, and have it
    # verify and be served under the other context -- the threat this module's
    # docstring says it defends against, reached by relabelling a good tag
    # rather than forging one.
    #
    # Conditional inclusion is sound in both directions. Stripping the context
    # from a scoped entry makes the verifier build the short message, which does
    # not match a tag computed over the long one; adding a context to an
    # unscoped entry makes it build the long message against a tag computed over
    # the short one. Both fail.
    context = _context_fields(payload)
    if context:
        message += "\n" + json.dumps(context, sort_keys=True, separators=(",", ":"), default=str)
    return message.encode()


def sign(
    payload: dict[str, Any], binding: CacheBinding, key: bytes | None = None
) -> dict[str, Any]:
    """Return *payload* with a signature and envelope version added."""
    material = key if key is not None else signing_key()
    tag = hmac.new(material, _message(payload, binding), hashlib.sha256).hexdigest()
    return {**payload, SIGNATURE_KEY: tag, VERSION_KEY: ENVELOPE_VERSION}


def verify(payload: dict[str, Any], binding: CacheBinding, key: bytes | None = None) -> bool:
    """Whether *payload* was signed for *binding* with this key.

    False for an unsigned payload, one carrying a different envelope version,
    one whose signed fields have changed, and one signed for a different
    binding.
    """
    tag = payload.get(SIGNATURE_KEY)
    if not isinstance(tag, str) or not tag:
        return False
    if str(payload.get(VERSION_KEY, "")) != ENVELOPE_VERSION:
        return False

    if key is not None:
        material = key
    else:
        try:
            material = signing_key()
        except SigningKeyError as exc:
            # A miss, not an error: the question is answered without the
            # cache. `check_signing_key` refuses the same key at startup.
            report_key_error(exc)
            return False
    expected = hmac.new(material, _message(payload, binding), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, tag)


# ---------------------------------------------------------------------------
# Verified question/SQL pairs
# ---------------------------------------------------------------------------

#: Incremented when the signed form of a verified pair changes.
PAIR_VERSION = "1"

#: Payload key holding a pair's signature version. The tag itself goes under
#: :data:`SIGNATURE_KEY`, as for cache entries.
PAIR_VERSION_KEY = "pair_version"

#: The parts of a verified pair its signature covers: everything the prompt
#: shows or the reader filters on.
PAIR_SIGNED_FIELDS = ("agent_id", "question", "sql", "verified")


def _pair_message(payload: dict[str, Any]) -> bytes:
    fields = {name: payload.get(name) for name in PAIR_SIGNED_FIELDS}
    body = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    # The leading label keeps a pair's tag from ever being valid as a cache
    # entry's, whose message starts with the envelope version, or the reverse.
    return f"verified-pair\n{PAIR_VERSION}\n{body}".encode()


def sign_pair(payload: dict[str, Any], key: bytes | None = None) -> dict[str, Any]:
    """Return a verified pair's *payload* with a signature added.

    Verified pairs are shown to the model as user-confirmed examples, and they
    live in Qdrant, which is an input rather than a store this process controls.
    The tag is made with the cache's signing key, held outside Qdrant, so a
    writer to Qdrant cannot add a pair the reader will accept. Raises
    :class:`SigningKeyError` when the key is unusable.
    """
    material = key if key is not None else signing_key()
    tag = hmac.new(material, _pair_message(payload), hashlib.sha256).hexdigest()
    return {**payload, SIGNATURE_KEY: tag, PAIR_VERSION_KEY: PAIR_VERSION}


def verify_pair(payload: dict[str, Any], key: bytes | None = None) -> bool:
    """Whether *payload* is a verified pair signed with this key.

    False for an unsigned pair (one promoted before pairs were signed), one of
    a different version, one whose signed fields have changed, and, with an
    unusable key, every pair.
    """
    tag = payload.get(SIGNATURE_KEY)
    if not isinstance(tag, str) or not tag:
        return False
    if str(payload.get(PAIR_VERSION_KEY, "")) != PAIR_VERSION:
        return False
    if key is not None:
        material = key
    else:
        try:
            material = signing_key()
        except SigningKeyError as exc:
            report_key_error(exc)
            return False
    expected = hmac.new(material, _pair_message(payload), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, tag)


# ---------------------------------------------------------------------------
# Query capsules
# ---------------------------------------------------------------------------

#: Incremented when the signed form of a query capsule changes.
CAPSULE_VERSION = "1"

#: Payload key holding a capsule's signature version.
CAPSULE_VERSION_KEY = "capsule_version"

#: The parts of a capsule its signature covers: everything the prompt shows or
#: the reader filters on.
CAPSULE_SIGNED_FIELDS = ("agent_id", "auto_description", "intent", "template_sql", "type")


def _capsule_message(payload: dict[str, Any]) -> bytes:
    fields = {name: payload.get(name) for name in CAPSULE_SIGNED_FIELDS}
    body = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    # Its own label, so a capsule's tag is never valid as a verified pair's or
    # a cache entry's, or the reverse.
    return f"query-capsule\n{CAPSULE_VERSION}\n{body}".encode()


def sign_capsule(payload: dict[str, Any], key: bytes | None = None) -> dict[str, Any]:
    """Return a query capsule's *payload* with a signature added.

    Capsules are shown to the model as example queries, from Qdrant, which a
    process other than this one can write to. As for verified pairs, the tag is
    made with the cache's signing key, held outside Qdrant. Raises
    :class:`SigningKeyError` when the key is unusable.
    """
    material = key if key is not None else signing_key()
    tag = hmac.new(material, _capsule_message(payload), hashlib.sha256).hexdigest()
    return {**payload, SIGNATURE_KEY: tag, CAPSULE_VERSION_KEY: CAPSULE_VERSION}


def verify_capsule(payload: dict[str, Any], key: bytes | None = None) -> bool:
    """Whether *payload* is a query capsule signed with this key.

    False for an unsigned capsule (one embedded before capsules were signed),
    one of a different version, one whose signed fields have changed, and, with
    an unusable key, every capsule.
    """
    tag = payload.get(SIGNATURE_KEY)
    if not isinstance(tag, str) or not tag:
        return False
    if str(payload.get(CAPSULE_VERSION_KEY, "")) != CAPSULE_VERSION:
        return False
    if key is not None:
        material = key
    else:
        try:
            material = signing_key()
        except SigningKeyError as exc:
            report_key_error(exc)
            return False
    expected = hmac.new(material, _capsule_message(payload), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, tag)


@lru_cache(maxsize=32)
def _hash_file(path: str, mtime_ns: int, size: int) -> str:
    """SHA-256 of a file, memoised on its identity and last modification.

    The binding is computed once per query, so the knowledge base would
    otherwise be re-hashed on every one. Arguments beyond *path* exist to key
    the cache; they are not read.
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _file_fingerprint(path: Path) -> str:
    """SHA-256 of *path*, or an empty string when it cannot be read.

    An empty value is consistent between signing and verification on the same
    host, so a missing file weakens the binding rather than breaking the cache.
    """
    try:
        info = path.stat()
    except OSError:
        return ""
    return _hash_file(str(path), info.st_mtime_ns, info.st_size)


def binding_for_agent(agent_id: str, dialect: str) -> CacheBinding:
    """The binding for entries produced by *agent_id* against *dialect*.

    The schema fingerprint comes from the agent's knowledge-base file, so an
    entry stops verifying once the schema it was generated against changes. The
    connector fingerprint covers the connector configuration file and the
    connector this agent resolves to, so a repointed connector does the same.

    The connector fingerprint deliberately excludes the stored password. The
    loader's own fingerprint includes it, which requires a keyring lookup; this
    is computed once per query and a keyring call on that path is not worth the
    additional binding.
    """
    from nlqueries import config  # noqa: PLC0415
    from nlqueries.sql_policy import POLICY_VERSION  # noqa: PLC0415
    from nlqueries.state_files import agent_storage_id  # noqa: PLC0415

    # The file name `export-kb` writes, by the rule it uses rather than a copy
    # of it. A copy once dropped dots instead, so for any agent id containing
    # one (the shape `connect` produces for a hostname) it fingerprinted a file
    # that does not exist, and the binding to the schema was empty.
    schema_fp = _file_fingerprint(config.KB_PATH / f"{agent_storage_id(agent_id)}.yaml")

    connector_fp = ""
    try:
        from nlqueries.connectors.loader import _find_connector_id  # noqa: PLC0415

        connector_id = _find_connector_id(agent_id) or ""
        connectors_fp = _file_fingerprint(config.CONNECTORS_FILE)
        connector_fp = hashlib.sha256(f"{connector_id}|{connectors_fp}".encode()).hexdigest()
    except Exception:  # noqa: BLE001 - an unreadable configuration weakens the binding only
        connector_fp = ""

    return CacheBinding(
        agent_id=agent_id,
        connector_fingerprint=connector_fp,
        dialect=dialect,
        schema_fingerprint=schema_fp,
        policy_version=POLICY_VERSION,
    )
