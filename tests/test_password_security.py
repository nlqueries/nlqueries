"""Tests for keychain-backed password helpers in nlqueries.cli.main (issue #1)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from nlqueries.cli.main import _get_full_url, _load_password, _save_password

# ---------------------------------------------------------------------------
# _save_password
# ---------------------------------------------------------------------------


def test_save_password_returns_true_on_success():
    mock_kr = MagicMock()
    with (
        patch("nlqueries.cli.main.keyring", mock_kr, create=True),
        patch.dict("sys.modules", {"keyring": mock_kr}),
    ):
        result = _save_password("my_connector", "s3cr3t")
    assert result is True


def test_save_password_returns_false_when_keyring_raises():
    mock_kr = MagicMock()
    mock_kr.set_password.side_effect = RuntimeError("No keyring backend")
    with patch.dict("sys.modules", {"keyring": mock_kr}):
        result = _save_password("my_connector", "s3cr3t")
    assert result is False


def test_save_password_returns_false_when_keyring_missing():
    with patch.dict("sys.modules", {"keyring": None}):
        result = _save_password("my_connector", "s3cr3t")
    assert result is False


# ---------------------------------------------------------------------------
# _load_password — keychain path
# ---------------------------------------------------------------------------


def test_load_password_from_keychain():
    mock_kr = MagicMock()
    mock_kr.get_password.return_value = "secret123"
    cfg = {"password_storage": "keychain", "url": "postgresql://user@host/db"}
    with patch.dict("sys.modules", {"keyring": mock_kr}):
        result = _load_password("my_connector", cfg)
    assert result == "secret123"
    mock_kr.get_password.assert_called_once_with("nlqueries", "my_connector")


def test_load_password_from_keychain_returns_none_on_error():
    mock_kr = MagicMock()
    mock_kr.get_password.side_effect = RuntimeError("Backend unavailable")
    cfg = {"password_storage": "keychain", "url": "postgresql://user@host/db"}
    with patch.dict("sys.modules", {"keyring": mock_kr}):
        result = _load_password("my_connector", cfg)
    assert result is None


# ---------------------------------------------------------------------------
# _load_password — legacy path (password in URL)
# ---------------------------------------------------------------------------


def test_load_password_legacy_extracts_from_url():
    cfg = {"url": "postgresql+psycopg2://alice:mypass@localhost/mydb"}
    result = _load_password("connector1", cfg)
    assert result == "mypass"


def test_load_password_legacy_returns_none_for_url_without_password():
    cfg = {"url": "postgresql+psycopg2://alice@localhost/mydb"}
    result = _load_password("connector1", cfg)
    assert result is None


def test_load_password_legacy_returns_none_for_bad_url():
    cfg = {"url": "not-a-valid-url"}
    result = _load_password("connector1", cfg)
    assert result is None


# ---------------------------------------------------------------------------
# _get_full_url
# ---------------------------------------------------------------------------


def test_get_full_url_returns_url_unchanged_for_legacy_connector():
    cfg = {"url": "postgresql+psycopg2://alice:mypass@localhost/mydb"}
    result = _get_full_url("connector1", cfg)
    assert result == "postgresql+psycopg2://alice:mypass@localhost/mydb"


def test_get_full_url_injects_keychain_password():
    mock_kr = MagicMock()
    mock_kr.get_password.return_value = "keychain_pass"
    cfg = {
        "password_storage": "keychain",
        "url": "postgresql+psycopg2://alice@localhost/mydb",
    }
    with patch.dict("sys.modules", {"keyring": mock_kr}):
        result = _get_full_url("connector1", cfg)
    assert "keychain_pass" in result
    assert "alice" in result
    assert "localhost" in result


def test_get_full_url_returns_stored_url_when_keychain_has_no_password():
    mock_kr = MagicMock()
    mock_kr.get_password.return_value = None
    cfg = {
        "password_storage": "keychain",
        "url": "postgresql+psycopg2://alice@localhost/mydb",
    }
    with patch.dict("sys.modules", {"keyring": mock_kr}):
        result = _get_full_url("connector1", cfg)
    assert result == "postgresql+psycopg2://alice@localhost/mydb"


def test_get_full_url_no_keychain_flag_returns_stored_url():
    cfg = {"url": "postgresql+psycopg2://user:pass@host/db"}
    result = _get_full_url("connector1", cfg)
    assert result == "postgresql+psycopg2://user:pass@host/db"
