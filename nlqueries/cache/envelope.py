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

_KEY_ENV = "NLQ_CACHE_SIGNING_KEY"


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
    return Path.home() / ".nlqueries" / "cache_signing_key"


def signing_key() -> bytes:
    """The HMAC key, from the environment or a file in the state directory.

    Generated on first use when neither is present. Generating rather than
    requiring configuration keeps caching working on upgrade; the key is still
    outside Qdrant, which is what the signature depends on.

    A deployment running more than one host must set :data:`_KEY_ENV` so the
    hosts agree. Where they do not, entries written by one host fail to verify
    on another and are treated as misses.
    """
    configured = os.getenv(_KEY_ENV, "").strip()
    if configured:
        return configured.encode("utf-8")

    path = _key_path()
    if path.exists():
        return path.read_bytes().strip()

    key = secrets.token_hex(32).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    # Readable only by the owner. The key is the whole of the protection.
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    logger.info(
        "Generated a semantic-cache signing key at %s. Set %s to share one key "
        "across hosts; otherwise each host verifies only its own entries.",
        path,
        _KEY_ENV,
    )
    return key


def _message(payload: dict[str, Any], binding: CacheBinding) -> bytes:
    fields = {name: payload.get(name) for name in SIGNED_FIELDS}
    body = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    return f"{ENVELOPE_VERSION}\n{binding.canonical()}\n{body}".encode()


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

    material = key if key is not None else signing_key()
    expected = hmac.new(material, _message(payload, binding), hashlib.sha256).hexdigest()
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

    safe_id = "".join(c for c in agent_id if c.isalnum() or c in "-_")
    schema_fp = _file_fingerprint(config.KB_PATH / f"{safe_id}.yaml")

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
