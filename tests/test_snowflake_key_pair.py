"""Snowflake key-pair authentication (owner, 2026-10-27).

A Snowflake service user signs in with a private key instead of a password.
``nlqueries connect snowflake --private-key-file`` keeps the key's contents in
the OS keychain, as a password is, and records the key file's path on a host
with no keychain. The connector hands the driver the key as PKCS#8 DER with
``SNOWFLAKE_JWT``, and no password.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from nlqueries.cli.main import cli
from nlqueries.connectors import loader
from nlqueries.connectors.snowflake import SnowflakeConnector
from nlqueries.connectors.snowflake_keys import (
    EncryptedKeyError,
    PrivateKeyError,
    passphrase_entry,
    private_key_der,
    private_key_entry,
)

_PASSPHRASE = "correct horse battery staple"


def _pem(passphrase: str | None = None) -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    encryption: serialization.KeySerializationEncryption = (
        serialization.BestAvailableEncryption(passphrase.encode())
        if passphrase
        else serialization.NoEncryption()
    )
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, encryption
    ).decode()


def _is_der_key(data: bytes) -> bool:
    return serialization.load_der_private_key(data, password=None) is not None


# -- the key itself -----------------------------------------------------------


def test_a_plain_key_becomes_der() -> None:
    assert _is_der_key(private_key_der(_pem()))


def test_an_encrypted_key_opens_with_its_passphrase() -> None:
    assert _is_der_key(private_key_der(_pem(_PASSPHRASE), _PASSPHRASE))


def test_an_encrypted_key_without_a_passphrase_asks_for_one() -> None:
    with pytest.raises(EncryptedKeyError):
        private_key_der(_pem(_PASSPHRASE))


@pytest.mark.parametrize(
    ("pem", "passphrase"),
    [("not a key", None), (_pem(_PASSPHRASE), "wrong"), (_pem(), "unneeded")],
    ids=["not a key", "wrong passphrase", "passphrase for a plain key"],
)
def test_a_key_that_will_not_open_says_so(pem: str, passphrase: str | None) -> None:
    with pytest.raises(PrivateKeyError):
        private_key_der(pem, passphrase)


# -- the connector -----------------------------------------------------------

_SITE = {"account": "acme-prod", "user": "svc_nlq", "warehouse": "WH", "database": "PROD"}


@patch("nlqueries.connectors.snowflake.snowflake.connector.connect")
def test_a_key_signs_in_with_jwt_and_no_password(mock_connect: Any) -> None:
    SnowflakeConnector().connect({**_SITE, "private_key": _pem(), "password": ""})

    kwargs = mock_connect.call_args.kwargs
    assert kwargs["authenticator"] == "SNOWFLAKE_JWT"
    assert _is_der_key(kwargs["private_key"])
    assert "password" not in kwargs


@patch("nlqueries.connectors.snowflake.snowflake.connector.connect")
def test_a_key_file_with_its_passphrase_in_a_variable(
    mock_connect: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-keychain shape: a path and the name of a variable."""
    key_file = tmp_path / "svc.p8"
    key_file.write_text(_pem(_PASSPHRASE), encoding="utf-8")
    monkeypatch.setenv("SF_KEY_PASS", _PASSPHRASE)

    SnowflakeConnector().connect(
        {**_SITE, "private_key_file": str(key_file), "private_key_passphrase_env": "SF_KEY_PASS"}
    )

    assert _is_der_key(mock_connect.call_args.kwargs["private_key"])


@patch("nlqueries.connectors.snowflake.snowflake.connector.connect")
def test_without_a_key_it_still_signs_in_with_the_password(mock_connect: Any) -> None:
    SnowflakeConnector().connect({**_SITE, "password": "s3cret"})

    kwargs = mock_connect.call_args.kwargs
    assert kwargs["password"] == "s3cret"
    assert "private_key" not in kwargs and "authenticator" not in kwargs


# -- the loader ---------------------------------------------------------------


def test_the_loader_reads_the_key_and_passphrase_from_the_keychain() -> None:
    stored = {
        private_key_entry("sf:prod"): "PEM",
        passphrase_entry("sf:prod"): _PASSPHRASE,
    }
    cfg = {
        "db_type": "snowflake",
        "url": "snowflake://svc_nlq:@acme-prod/PROD",
        "account": "acme-prod",
        "private_key_storage": "keychain",
    }
    with patch("keyring.get_password", lambda _service, entry: stored.get(entry)):
        credentials = loader.credentials_for("sf:prod", cfg)

    assert credentials["private_key"] == "PEM"
    assert credentials["private_key_passphrase"] == _PASSPHRASE
    # Where the key is kept describes the entry, not the connection.
    assert "private_key_storage" not in credentials


# -- nlqueries connect --------------------------------------------------------


class _Reachable:
    seen: dict[str, Any] = {}

    def connect(self, credentials: dict[str, Any]) -> None:
        _Reachable.seen = credentials

    def connection_problem(self) -> None:
        return None


def _connect(
    tmp_path: Path, pem: str, *extra: str, keychain: bool, env: dict[str, str] | None = None
) -> tuple[Any, dict[str, Any], dict[str, str]]:
    key_file = tmp_path / "svc.p8"
    key_file.write_text(pem, encoding="utf-8")
    saved: dict[str, Any] = {}
    secrets: dict[str, str] = {}

    def _set(_service: str, entry: str, value: str) -> None:
        if not keychain:
            raise RuntimeError("no keyring backend")
        secrets[entry] = value

    with (
        patch("nlqueries.cli.main._save_connector", lambda cid, cfg: saved.update(cfg)),
        patch("nlqueries.cli.main.connector_class_for", return_value=_Reachable),
        patch("keyring.set_password", _set),
        patch("keyring.delete_password", lambda *a: None),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "connect",
                "snowflake",
                "--account",
                "acme-prod",
                "--database",
                "PROD",
                "--user",
                "svc_nlq",
                "--warehouse",
                "WH",
                "--private-key-file",
                str(key_file),
                *extra,
            ],
            env=env,
            input=f"{_PASSPHRASE}\n",
        )
    return result, saved, secrets


def test_connect_keeps_the_key_in_the_keychain_and_not_in_the_file(tmp_path: Path) -> None:
    pem = _pem()
    result, saved, secrets = _connect(tmp_path, pem, keychain=True)

    assert result.exit_code == 0, result.output
    assert saved["private_key_storage"] == "keychain"
    assert pem not in str(saved) and "private_key_file" not in saved
    assert next(iter(secrets.values())) == pem
    # The connection was tested with the key, not a password.
    assert _Reachable.seen["private_key"] == pem and not _Reachable.seen["password"]


def test_an_encrypted_keys_passphrase_is_asked_for_and_kept_with_it(tmp_path: Path) -> None:
    result, saved, secrets = _connect(tmp_path, _pem(_PASSPHRASE), keychain=True)

    assert result.exit_code == 0, result.output
    assert _PASSPHRASE in secrets.values()
    assert _PASSPHRASE not in str(saved)


def test_without_a_keychain_the_key_files_path_is_recorded(tmp_path: Path) -> None:
    result, saved, _ = _connect(
        tmp_path,
        _pem(_PASSPHRASE),
        "--private-key-passphrase-env",
        "SF_KEY_PASS",
        keychain=False,
        env={"SF_KEY_PASS": _PASSPHRASE},
    )

    assert result.exit_code == 0, result.output
    assert saved["private_key_file"] == str((tmp_path / "svc.p8").resolve())
    assert saved["private_key_passphrase_env"] == "SF_KEY_PASS"
    assert "private_key_storage" not in saved
    assert "keyring unavailable" in " ".join(result.output.split())


def test_without_a_keychain_a_typed_passphrase_is_refused_rather_than_lost(
    tmp_path: Path,
) -> None:
    """It could be kept nowhere, and the connector would fail at its next use."""
    result, saved, _ = _connect(tmp_path, _pem(_PASSPHRASE), keychain=False)

    assert result.exit_code != 0
    assert "--private-key-passphrase-env" in result.output
    assert saved == {}


@pytest.mark.parametrize(
    "extra",
    [["--password", "x"], ["--password-env", "PW"]],
    ids=["password", "password-env"],
)
def test_a_key_and_a_password_together_are_refused(tmp_path: Path, extra: list[str]) -> None:
    result, saved, _ = _connect(tmp_path, _pem(), *extra, keychain=True)

    assert result.exit_code != 0
    assert saved == {}


def test_a_key_is_refused_for_another_database(tmp_path: Path) -> None:
    key_file = tmp_path / "k.p8"
    key_file.write_text(_pem(), encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "connect",
            "postgres",
            "--database",
            "d",
            "--user",
            "u",
            "--private-key-file",
            str(key_file),
        ],
    )

    assert result.exit_code != 0
    assert "only valid with snowflake" in result.output
