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
