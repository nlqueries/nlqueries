"""Tests for ``scripts/sign_rates.py``, the signing side of the LLM rate file.

Enterprise verifies what this signs. So what matters is that ``build`` leaves out
every entry enterprise would refuse a whole file over, and every entry that looks
like a data error, and says which; that a signature covers the exact bytes; and
that ``keygen`` never overwrites a key.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import pathlib
import re
import threading
import urllib.error
import urllib.request
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "sign_rates.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("sign_rates", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sign_rates = _load()

_SOURCE = {
    "sample_spec": "placeholder",
    "claude-sonnet-5-5": {
        "input_cost_per_token": 2e-06,
        "output_cost_per_token": 1e-05,
        "cache_read_input_token_cost": 1e-07,
        "input_cost_per_token_above_200k_tokens": 4e-06,
        "max_tokens": 64000,
        "litellm_provider": "anthropic",
        "input_cost_per_token_priority": 3e-06,
    },
    "text-embedding-3-small": {
        "input_cost_per_token": 2e-08,
        "output_cost_per_token": 0.0,
        "mode": "embedding",
    },
    "image-model": {"output_cost_per_image": 0.04},
    "wandb/zai-org/GLM-4.5": {"input_cost_per_token": 0.055, "output_cost_per_token": 0.2},
    "negative": {"input_cost_per_token": -1e-06, "output_cost_per_token": 1e-06},
    "stringly": {"input_cost_per_token": "1e-06", "output_cost_per_token": 1e-06},
    "two-tiers": {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 1e-06,
        "input_cost_per_token_above_128k_tokens": 2e-06,
        "input_cost_per_token_above_256k_tokens": 3e-06,
    },
    "zero-tier": {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 1e-06,
        "input_cost_per_token_above_0k_tokens": 2e-06,
    },
    "null-input": {"input_cost_per_token": None, "output_cost_per_token": 1e-06},
    "null-cache": {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 2e-06,
        "cache_read_input_token_cost": None,
    },
    "Case-Twin": {"input_cost_per_token": 1e-06, "output_cost_per_token": 1e-06},
    "case-twin": {"input_cost_per_token": 2e-06, "output_cost_per_token": 2e-06},
}


def _built(**kwargs: Any) -> tuple[dict[str, Any], list[str]]:
    data, dropped = sign_rates.build(
        _SOURCE, date(2026, 11, 1), kwargs.get("max_per_million", 1000.0)
    )
    return json.loads(data), dropped


def test_build_keeps_what_enterprise_prices_and_only_its_rate_fields() -> None:
    body, _ = _built()
    assert body["format"] == "nlq-rates-v1"
    assert body["effective_from"] == "2026-11-01"
    assert body["models"]["claude-sonnet-5-5"] == {
        "input_cost_per_token": 2e-06,
        "output_cost_per_token": 1e-05,
        "cache_read_input_token_cost": 1e-07,
        "input_cost_per_token_above_200k_tokens": 4e-06,
    }
    # No output rate per token: not something enterprise prices. Left out silently.
    assert "image-model" not in body["models"]
    assert "sample_spec" not in body["models"]


@pytest.mark.parametrize(
    ("model", "reason"),
    [
        ("wandb/zai-org/GLM-4.5", "is $55,000 per million, over $1,000"),
        ("negative", "input_cost_per_token is -1e-06"),
        ("stringly", "input_cost_per_token is not a number"),
        ("two-tiers", "2 thresholds"),
        ("zero-tier", "threshold of 0"),
        ("case-twin", "same id in lower case"),
    ],
)
def test_build_leaves_out_and_names_each_entry_enterprise_would_refuse(
    model: str, reason: str
) -> None:
    body, dropped = _built()
    assert model not in body["models"]
    assert any(line.startswith(f"{model}: ") and reason in line for line in dropped), dropped


def test_the_sanity_bound_can_be_raised() -> None:
    body, _ = _built(max_per_million=1_000_000.0)
    assert "wandb/zai-org/GLM-4.5" in body["models"]


def test_build_from_litellms_bundled_map() -> None:
    """The real source: thousands of entries, and none over the bound kept.

    Which models litellm prices absurdly changes from release to release, so the
    bound is checked as a property of the output, not by a model's name."""
    data, dropped = sign_rates.build(sign_rates._litellm_map(), date(2026, 11, 1), 1000.0)
    models = json.loads(data)["models"]
    assert len(models) > 1000
    assert all(
        value * 1_000_000 <= 1000.0
        for entry in models.values()
        for value in entry.values()
        if isinstance(value, (int, float))
    )
    assert all(line.split(": ", 1)[0] not in models for line in dropped)


def _key_pair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    key = Ed25519PrivateKey.generate()
    return key, key.public_key()


def test_a_signed_file_verifies() -> None:
    key, public = _key_pair()
    data, _ = sign_rates.build(_SOURCE, date(2026, 11, 1), 1000.0)
    sign_rates.verify(data, sign_rates.sign(data, key, "rates-2026"), public, "rates-2026")


def test_the_signature_covers_every_byte() -> None:
    key, public = _key_pair()
    data, _ = sign_rates.build(_SOURCE, date(2026, 11, 1), 1000.0)
    sig = sign_rates.sign(data, key, "rates-2026")
    with pytest.raises(SystemExit, match="does not verify"):
        sign_rates.verify(data.replace(b"2e-06", b"1e-06", 1), sig, public, "rates-2026")
    with pytest.raises(SystemExit, match="does not verify"):
        sign_rates.verify(data + b" ", sig, public, "rates-2026")


def test_another_key_id_is_refused() -> None:
    key, public = _key_pair()
    data, _ = sign_rates.build(_SOURCE, date(2026, 11, 1), 1000.0)
    with pytest.raises(SystemExit, match="not 'rates-2027'"):
        sign_rates.verify(data, sign_rates.sign(data, key, "rates-2026"), public, "rates-2027")


def test_only_a_rate_file_is_signed() -> None:
    key, _ = _key_pair()
    with pytest.raises(SystemExit, match="Not a nlq-rates-v1 rate file"):
        sign_rates.sign(b'{"format": "something-else"}', key, "rates-2026")


def test_the_command_line_round_trip(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "map.json"
    source.write_text(json.dumps(_SOURCE), encoding="utf-8")
    key_path, rates = tmp_path / "rates.pem", tmp_path / "rates.json"

    assert sign_rates.main(["keygen", str(key_path)]) == 0
    public = capsys.readouterr().out.rsplit(": ", 1)[1].strip()
    Ed25519PublicKey.from_public_bytes(base64.urlsafe_b64decode(public + "=" * (-len(public) % 4)))

    assert (
        sign_rates.main(
            ["build", str(rates), "--effective-from", "2026-11-01", "--source", str(source)]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "left out wandb/zai-org/GLM-4.5" in out
    assert sign_rates.main(["sign", str(rates), "--key", str(key_path), "--kid", "rates-2026"]) == 0
    assert (tmp_path / "rates.json.sig").exists()
    assert (
        sign_rates.main(["verify", str(rates), "--public-key", public, "--kid", "rates-2026"]) == 0
    )


def test_keygen_never_overwrites_a_key(tmp_path: pathlib.Path) -> None:
    existing = tmp_path / "rates.pem"
    existing.write_bytes(b"a key someone needs")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        sign_rates.main(["keygen", str(existing)])
    assert existing.read_bytes() == b"a key someone needs"


def test_a_null_rate_is_never_written() -> None:
    """A null input or output rate is no price, so the entry is not priced, as
    enterprise would leave it; a null cache rate is dropped from an entry kept."""
    body, dropped = _built()
    assert "null-input" not in body["models"]
    assert not any(line.startswith("null-input") for line in dropped)
    assert body["models"]["null-cache"] == {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 2e-06,
    }
    assert all(v is not None for entry in body["models"].values() for v in entry.values())


def test_keygen_refuses_a_path_inside_this_repository() -> None:
    inside = _SCRIPT.parents[1] / "rates-signing-key.pem"
    with pytest.raises(SystemExit, match="inside this repository"):
        sign_rates.main(["keygen", str(inside)])
    assert not inside.exists()


def test_a_pem_file_is_ignored_by_git() -> None:
    import subprocess

    probe = "assets/rates-signing-key.pem"
    result = subprocess.run(
        ["git", "check-ignore", "-q", probe], cwd=_SCRIPT.parents[1], capture_output=True
    )
    assert result.returncode == 0, f"{probe} is not ignored"


# --- update: what the daily job runs ---------------------------------------------------

_DAY = date(2026, 11, 1)
_KID = "rates-2026"


def _map(n: int = 20, changes: dict[str, Any] | None = None) -> dict[str, Any]:
    """*n* priced models; a change of ``None`` takes that model out."""
    source: dict[str, Any] = {
        f"model-{i:02d}": {"input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}
        for i in range(n)
    }
    for name, entry in (changes or {}).items():
        if entry is None:
            source.pop(name)
        else:
            source[name] = entry
    return source


def _update(
    directory: pathlib.Path,
    source: dict[str, Any],
    key: Ed25519PrivateKey,
    day: date = _DAY,
    **kwargs: Any,
) -> tuple[bool, list[str]]:
    return sign_rates.update(directory, source, key, _KID, today=day, **kwargs)


def _files(directory: pathlib.Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in directory.iterdir()} if directory.exists() else {}


def _public(key: Ed25519PrivateKey) -> str:
    return sign_rates._b64url(key.public_key().public_bytes_raw())


def test_the_first_update_publishes_a_signed_file(tmp_path: pathlib.Path) -> None:
    key, public = _key_pair()
    directory = tmp_path / "assets"
    wrote, report = _update(directory, _map(), key)
    assert wrote
    data = (directory / "rates.json").read_bytes()
    sign_rates.verify(data, (directory / "rates.json.sig").read_bytes(), public, _KID)
    body = json.loads(data)
    assert body["effective_from"] == "2026-11-01"
    assert len(body["models"]) == 20
    assert report[0] == "Rates for 20 models, effective 2026-11-01: 20 added, 0 removed, 0 changed"


def test_an_update_with_no_rate_changed_writes_nothing(tmp_path: pathlib.Path) -> None:
    """Not even a new effective date: the file in force stays the one published."""
    key, _ = _key_pair()
    _update(tmp_path, _map(), key)
    before = _files(tmp_path)
    wrote, report = _update(tmp_path, _map(), key, day=_DAY + timedelta(days=1))
    assert not wrote
    assert report[0] == "Rates unchanged: 20 models."
    assert _files(tmp_path) == before


def test_a_changed_rate_is_published_effective_that_day(tmp_path: pathlib.Path) -> None:
    key, public = _key_pair()
    _update(tmp_path, _map(), key)
    later = _DAY + timedelta(days=3)
    source = _map(
        changes={
            "model-03": {"input_cost_per_token": 3e-06, "output_cost_per_token": 2e-06},
            "model-new": {"input_cost_per_token": 5e-06, "output_cost_per_token": 6e-06},
        }
    )
    wrote, report = _update(tmp_path, source, key, day=later)
    assert wrote
    data = (tmp_path / "rates.json").read_bytes()
    sign_rates.verify(data, (tmp_path / "rates.json.sig").read_bytes(), public, _KID)
    body = json.loads(data)
    assert body["effective_from"] == later.isoformat()
    assert body["models"]["model-03"]["input_cost_per_token"] == 3e-06
    assert report[0].endswith(": 1 added, 0 removed, 1 changed")
    assert "  changed: model-03" in report
    assert "  added: model-new" in report


def test_an_update_that_would_drop_too_many_models_writes_nothing(tmp_path: pathlib.Path) -> None:
    """A tenth of the published models may go; more is refused unless allowed."""
    key, _ = _key_pair()
    _update(tmp_path, _map(), key)
    before = _files(tmp_path)
    three_gone = _map(changes={"model-00": None, "model-01": None, "model-02": None})
    with pytest.raises(SystemExit, match="3 of the 20 published models would go"):
        _update(tmp_path, three_gone, key, day=_DAY + timedelta(days=1))
    assert _files(tmp_path) == before

    assert _update(tmp_path, three_gone, key, max_removed=0.2)[0]


def test_a_tenth_of_the_models_may_go(tmp_path: pathlib.Path) -> None:
    key, _ = _key_pair()
    _update(tmp_path, _map(), key)
    wrote, report = _update(tmp_path, _map(changes={"model-00": None, "model-01": None}), key)
    assert wrote
    assert "  removed: model-00, model-01" in report


def test_a_key_other_than_the_one_named_writes_nothing(tmp_path: pathlib.Path) -> None:
    key, _ = _key_pair()
    other, _ = _key_pair()
    with pytest.raises(SystemExit, match="not the one --public-key names"):
        _update(tmp_path, _map(), key, public_key=_public(other))
    assert _files(tmp_path) == {}
    assert _update(tmp_path, _map(), key, public_key=_public(key))[0]


@pytest.mark.parametrize(
    ("kid", "public_key", "message"),
    [("", None, "key id is empty"), (_KID, "", "--public-key is empty")],
)
def test_an_empty_setting_is_refused(
    tmp_path: pathlib.Path, kid: str, public_key: str | None, message: str
) -> None:
    """What an unset repository variable passes."""
    key, _ = _key_pair()
    with pytest.raises(SystemExit, match=message):
        sign_rates.update(tmp_path, _map(), key, kid, today=_DAY, public_key=public_key)
    assert _files(tmp_path) == {}


def test_a_missing_signature_is_written_again(tmp_path: pathlib.Path) -> None:
    key, public = _key_pair()
    _update(tmp_path, _map(), key)
    (tmp_path / "rates.json.sig").unlink()
    assert _update(tmp_path, _map(), key)[0]
    sign_rates.verify(
        (tmp_path / "rates.json").read_bytes(),
        (tmp_path / "rates.json.sig").read_bytes(),
        public,
        _KID,
    )


def test_the_command_line_update_takes_the_key_from_the_environment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from cryptography.hazmat.primitives import serialization

    key, public = _key_pair()
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    source = tmp_path / "map.json"
    source.write_text(json.dumps(_map()), encoding="utf-8")
    args = ["update", str(tmp_path / "assets"), "--kid", _KID, "--key-env", "RATES_KEY_TEST"]
    args += ["--source", str(source), "--public-key", _public(key)]

    monkeypatch.setenv("RATES_KEY_TEST", "")
    with pytest.raises(SystemExit, match="RATES_KEY_TEST is empty or unset"):
        sign_rates.main(args)

    monkeypatch.setenv("RATES_KEY_TEST", pem)
    assert sign_rates.main(args) == 0
    assert capsys.readouterr().out.startswith("Rates for 20 models, effective ")
    sign_rates.verify(
        (tmp_path / "assets" / "rates.json").read_bytes(),
        (tmp_path / "assets" / "rates.json.sig").read_bytes(),
        public,
        _KID,
    )


# --- fetching LiteLLM's map ---------------------------------------------------------------


class _Response:
    def __init__(self, status: int, body: bytes) -> None:
        self.status, self._body = status, body

    def read(self, n: int) -> bytes:
        return self._body[:n]

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _Opener:
    def __init__(
        self, status: int = 200, body: bytes = b"{}", error: Exception | None = None
    ) -> None:
        self.status, self.body, self.error = status, body, error
        self.urls: list[str] = []

    def open(self, url: str, timeout: float) -> _Response:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return _Response(self.status, self.body)


def test_a_fetched_map_is_returned() -> None:
    opener = _Opener(body=json.dumps(_map(2)).encode())
    assert sign_rates.fetch_map("https://example.test/map.json", opener=opener) == _map(2)
    assert opener.urls == ["https://example.test/map.json"]


def test_the_map_is_fetched_over_https_only() -> None:
    opener = _Opener()
    with pytest.raises(SystemExit, match="must be fetched over https"):
        sign_rates.fetch_map("http://example.test/map.json", opener=opener)
    assert opener.urls == []


@pytest.mark.parametrize(
    ("opener", "message"),
    [
        (_Opener(status=204), "HTTP 204"),
        (
            _Opener(error=urllib.error.HTTPError("u", 404, "Not Found", {}, None)),
            "Could not fetch",
        ),
        (_Opener(error=urllib.error.URLError("refused")), "Could not fetch"),
        (_Opener(body=b"<html>"), "is not JSON"),
        (_Opener(body=b"[]"), "not a model-cost map"),
        (_Opener(body=b'{"m": "' + b"x" * 64 + b'"}'), "larger than 32 bytes"),
    ],
)
def test_a_bad_fetch_says_why(opener: _Opener, message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        sign_rates.fetch_map("https://example.test/map.json", opener=opener, max_bytes=32)


def test_the_fetch_follows_no_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fetch's own opener is built with the handler, and the handler, given a
    real redirect, follows nothing."""
    handlers: list[Any] = []

    def _build_opener(*given: Any) -> _Opener:
        handlers.extend(given)
        return _Opener()

    monkeypatch.setattr(sign_rates.urllib.request, "build_opener", _build_opener)
    sign_rates.fetch_map("https://example.test/map.json")
    assert handlers == [sign_rates._NoRedirect]
    monkeypatch.undo()

    class _Redirect(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), _Redirect)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        opener = urllib.request.build_opener(sign_rates._NoRedirect)
        with pytest.raises(urllib.error.HTTPError) as raised:
            opener.open(f"http://127.0.0.1:{server.server_port}/map.json", timeout=10)
        assert raised.value.code == 302
    finally:
        thread.join(timeout=10)
        server.server_close()


# --- the job's dependencies -----------------------------------------------------------------


def _lock_blocks(path: pathlib.Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return {
        m.group(1): m.group(0)
        for m in re.finditer(r"^([a-z0-9-]+)==[^\n]*\n(?:[ \t][^\n]*\n)*", text, re.M)
    }


def test_the_rates_lock_is_core_locks_own_pins() -> None:
    """The signing job installs these; lock-audit scans core.lock, so each block
    has to be core.lock's, unchanged."""
    requirements = _SCRIPT.parents[1] / "requirements"
    rates = _lock_blocks(requirements / "rates.lock")
    core = _lock_blocks(requirements / "core.lock")
    assert set(rates) == {"cffi", "cryptography", "pycparser"}
    for name, block in rates.items():
        assert block == core[name], f"{name} in rates.lock differs from core.lock"
