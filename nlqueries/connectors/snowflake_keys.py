"""
nlqueries.connectors.snowflake_keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The private key a Snowflake connector signs in with, for key-pair
authentication.

Snowflake is retiring single-factor password sign-in for users, and a service
user signs in with a key pair instead: the connector proves it holds the
private key whose public half is set on the user (``ALTER USER ... SET
RSA_PUBLIC_KEY``). The driver takes the key as unencrypted PKCS#8 DER bytes, so
the PEM a user has -- encrypted with a passphrase or not -- is decoded here.

Where the key is kept (owner, 2026-10-27): its contents in the OS keychain,
under the connector's id, as a password is; on a host with no keychain, the
path of the key file, read at each connect.
"""

from __future__ import annotations

import contextlib
import secrets
from typing import Any

from cryptography.hazmat.primitives import serialization

#: The keychain service a connector's secrets are stored under, as passwords are.
KEYCHAIN_SERVICE = "nlqueries"


class PrivateKeyError(ValueError):
    """The private key could not be read: not a PEM key, or the wrong passphrase."""


class EncryptedKeyError(PrivateKeyError):
    """The private key is encrypted, and no passphrase was given."""


def private_key_entry(connector_id: str) -> str:
    """The keychain entry holding *connector_id*'s private key."""
    return f"{connector_id}:snowflake_private_key"


def passphrase_entry(connector_id: str) -> str:
    """The keychain entry holding the passphrase of *connector_id*'s private key."""
    return f"{connector_id}:snowflake_private_key_passphrase"


def private_key_der(pem: str | bytes, passphrase: str | None = None) -> bytes:
    """*pem*, decrypted with *passphrase* if it is encrypted, as the
    unencrypted PKCS#8 DER the Snowflake driver signs with.

    Raises:
        PrivateKeyError: not a PEM private key; encrypted and no passphrase
            given; or a passphrase that does not open it.
    """
    data = pem.encode() if isinstance(pem, str) else pem
    secret = passphrase.encode() if passphrase else None
    try:
        key = serialization.load_pem_private_key(data, password=secret)
    except TypeError as exc:
        # The library's way of saying the key's encryption and the passphrase
        # disagree: an encrypted key with none, or a plain one with one.
        if secret is None:
            raise EncryptedKeyError("The private key is encrypted: give its passphrase.") from exc
        raise PrivateKeyError("The private key is not encrypted: give no passphrase.") from exc
    except ValueError as exc:
        raise PrivateKeyError("Not a PEM private key, or the passphrase does not open it.") from exc
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


#: Characters per keychain entry. Windows Credential Manager holds at most 2,560
#: bytes per credential, and keyring stores the value as UTF-16, so a PEM --
#: 1,700 characters for a 2048-bit key -- does not fit in one entry: CredWrite
#: refuses it ("The stub received bad data"). Measured, 1,270 characters fit;
#: 1,000 leaves room. The key is split across entries, and the connector's own
#: entry says which generation of them is in use, and how many there are (from
#: #347's review).
_CHUNK = 1000


class KeychainKeyError(RuntimeError):
    """A connector's private key is not in the keychain, or not all of it."""


def _generation_entries(connector_id: str, generation: str, count: int) -> list[str]:
    """The entries one stored key uses: its parts, then its passphrase."""
    entry = private_key_entry(connector_id)
    parts = [f"{entry}:{generation}:{i}" for i in range(count)]
    return [*parts, f"{passphrase_entry(connector_id)}:{generation}"]


def _parse_pointer(value: str | None) -> tuple[str, int] | None:
    """The generation and part count the connector's own entry holds, or None."""
    generation, _, count = (value or "").partition(":")
    if not generation.isalnum() or not count.isdigit() or int(count) == 0:
        return None
    return generation, int(count)


def store_in_keychain(keyring: Any, connector_id: str, pem: str, passphrase: str | None) -> None:
    """Put *pem*, and *passphrase*, in the keychain for *connector_id*.

    *keyring* is the ``keyring`` module, or a stand-in. Raises what it raises
    when the keychain will not take a value.

    Each key goes under a fresh generation, its parts and passphrase alike, and
    the connector's own entry -- ``<generation>:<count>`` -- switches to it
    last. So a write that fails part-way leaves the previous key and its
    passphrase as they were (from #347's review): the new parts never overwrite
    the ones in use. The previous generation is removed after the switch.
    """
    entry = private_key_entry(connector_id)
    previous = None
    with contextlib.suppress(Exception):
        previous = _parse_pointer(keyring.get_password(KEYCHAIN_SERVICE, entry))
    generation = secrets.token_hex(4)
    while previous is not None and generation == previous[0]:
        generation = secrets.token_hex(4)
    chunks = [pem[i : i + _CHUNK] for i in range(0, len(pem), _CHUNK)]
    *part_entries, passphrase_name = _generation_entries(connector_id, generation, len(chunks))
    written: list[str] = []
    try:
        for name, chunk in zip(part_entries, chunks, strict=True):
            keyring.set_password(KEYCHAIN_SERVICE, name, chunk)
            written.append(name)
        if passphrase:
            keyring.set_password(KEYCHAIN_SERVICE, passphrase_name, passphrase)
            written.append(passphrase_name)
        keyring.set_password(KEYCHAIN_SERVICE, entry, f"{generation}:{len(chunks)}")
    except Exception:
        _delete_entries(keyring, written)
        raise
    if previous is not None:
        _delete_entries(keyring, _generation_entries(connector_id, *previous))


def _delete_entries(keyring: Any, names: list[str]) -> None:
    """Remove *names* from the keychain, as far as it lets us; what stays is unused."""
    for name in names:
        with contextlib.suppress(Exception):
            keyring.delete_password(KEYCHAIN_SERVICE, name)


def load_from_keychain(keyring: Any, connector_id: str) -> tuple[str, str | None]:
    """*connector_id*'s private key and passphrase, from the keychain.

    Raises:
        KeychainKeyError: the key is not there, or not all of it -- the keychain
            is locked, nlqueries runs as another account, or the entry was
            removed. Said, rather than left to Snowflake to refuse a sign-in
            with no key and no password.
    """
    entry = private_key_entry(connector_id)
    missing = KeychainKeyError(
        f"The private key for connector {connector_id!r} could not be read from the OS "
        "keychain (it may be locked, nlqueries may be running as another account, or the "
        "entry was removed). Run `nlqueries connect snowflake ... --private-key-file` "
        "again to store it."
    )
    try:
        pointer = _parse_pointer(keyring.get_password(KEYCHAIN_SERVICE, entry))
        if pointer is None:
            raise missing
        *part_entries, passphrase_name = _generation_entries(connector_id, *pointer)
        parts = [keyring.get_password(KEYCHAIN_SERVICE, name) for name in part_entries]
        passphrase = keyring.get_password(KEYCHAIN_SERVICE, passphrase_name)
    except KeychainKeyError:
        raise
    except Exception as exc:  # the keychain itself failed
        raise missing from exc
    if any(part is None for part in parts):
        raise missing
    return "".join(str(part) for part in parts), passphrase
