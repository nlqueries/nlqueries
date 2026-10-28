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
from datetime import date
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
