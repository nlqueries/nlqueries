#!/usr/bin/env python3
"""Build and sign the LLM rate file that nlqueries-enterprise prices from.

The rate file lets a deployment take new LLM prices without an upgrade.
``assets/rates.json`` is LiteLLM's model-cost format with an effective date, and
``assets/rates.json.sig`` is a detached Ed25519 signature over its exact bytes.
Enterprise ships the public key and refuses any file it does not verify. This
script is the signing side. It never sees a private key until you pass one.

    keygen  OUT.pem                 a new signing key; prints the public half
    build   OUT.json --effective-from YYYY-MM-DD [--source MAP.json]
    sign    rates.json --key KEY.pem --kid KID
    verify  rates.json --public-key BASE64URL --kid KID

``build`` takes LiteLLM's bundled map (or ``--source``) and keeps only what
enterprise prices:
- the entries with an input and an output rate per token;
- in each, only the rate fields, including a long-prompt tier
  (``*_above_<N>k_tokens``).

It applies the checks enterprise applies, and a **sanity bound** enterprise
deliberately does not: a rate above ``--max-per-million`` USD per million tokens
(default 1000) is taken as a data error. Each entry it drops is listed, with the
reason, so the file is what you meant to publish.
- litellm 1.102.1 has four such models, for example ``wandb/zai-org/GLM-4.5``,
  priced at $200,000 per million output tokens.
- Refusing at signing time stops the signer. Refusing at verification would stop
  every customer.

The file this writes is exactly what is signed, so do not reformat it after
``sign``. Re-run ``sign`` if you change it.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import pathlib
import re
import sys
from datetime import UTC, date, datetime
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

FORMAT = "nlq-rates-v1"

#: The per-token rate fields enterprise reads, and their long-prompt tier forms.
BASE_FIELDS = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
)
TIER_FIELD = re.compile(
    r"^(input_cost_per_token|output_cost_per_token|cache_read_input_token_cost"
    r"|cache_creation_input_token_cost)_above_(\d+)k_tokens$"
)


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


# --- build ---------------------------------------------------------------------


def _litellm_map() -> dict[str, Any]:
    """LiteLLM's bundled map, never the remote one."""
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    import litellm

    return dict(litellm.model_cost)


def _rate_fields(entry: dict[str, Any]) -> dict[str, Any]:
    """The rate fields enterprise reads. A null is dropped: it is no rate, and
    enterprise treats a missing field the same way."""
    return {
        k: v
        for k, v in entry.items()
        if (k in BASE_FIELDS or TIER_FIELD.match(k)) and v is not None
    }


def _problem(entry: dict[str, Any], max_per_million: float) -> str | None:
    """Why enterprise would refuse *entry*, or why it looks like a data error."""
    thresholds = {int(m.group(2)) for k in entry if (m := TIER_FIELD.match(k))}
    if len(thresholds) > 1:
        return f"long-prompt rates at {len(thresholds)} thresholds; enterprise takes one"
    if 0 in thresholds:
        return "a long-prompt threshold of 0"
    for field, value in entry.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{field} is not a number"
        try:
            finite = math.isfinite(value)
        except OverflowError:
            return f"{field} is too large to be a rate"
        if not finite or value < 0:
            return f"{field} is {value}"
        if value * 1_000_000 > max_per_million:
            return f"{field} is ${value * 1_000_000:,.0f} per million, over ${max_per_million:,.0f}"
    return None


def build(
    source: dict[str, Any], effective_from: date, max_per_million: float
) -> tuple[bytes, list[str]]:
    """The rate file's bytes, and a line for each entry left out."""
    models: dict[str, dict[str, Any]] = {}
    dropped: list[str] = []
    seen: set[str] = set()
    for name in sorted(source, key=str.lower):
        entry = source[name]
        if not isinstance(entry, dict):
            continue
        fields = _rate_fields(entry)
        if "input_cost_per_token" not in fields or "output_cost_per_token" not in fields:
            continue  # embeddings, images, per-second and per-call pricing, null rates
        problem = _problem(fields, max_per_million)
        if problem is None and name.strip().lower() in seen:
            problem = "another entry has the same id in lower case"
        if problem is not None:
            dropped.append(f"{name}: {problem}")
            continue
        seen.add(name.strip().lower())
        models[name] = fields
    if not models:
        raise SystemExit("No entry has a rate enterprise can price; nothing to build.")
    body = {
        "format": FORMAT,
        "effective_from": effective_from.isoformat(),
        "published": datetime.now(UTC).date().isoformat(),
        "models": models,
    }
    return (json.dumps(body, indent=1, sort_keys=False) + "\n").encode("utf-8"), dropped


# --- keys and signatures -------------------------------------------------------


#: This repository. A key written inside it is one ``git add`` away from public.
_REPOSITORY = pathlib.Path(__file__).resolve().parents[1]


def keygen(out: pathlib.Path) -> str:
    """Write a new private key to *out* and return the public half.

    Refuses a path inside this repository, which is public: the key signs rates
    every enterprise deployment trusts. ``0o600`` restricts the file on POSIX;
    on Windows it takes the directory's permissions, so choose that directory.
    """
    if out.resolve().is_relative_to(_REPOSITORY):
        raise SystemExit(f"{out} is inside this repository; write the key somewhere else.")
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite a key.")
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(pem)
    return _b64url(key.public_key().public_bytes_raw())


def sign(data: bytes, key: Ed25519PrivateKey, kid: str) -> bytes:
    """The ``.sig`` file for *data*."""
    _check_file(data)
    return (json.dumps({"kid": kid, "sig": _b64url(key.sign(data))}) + "\n").encode("utf-8")


def verify(data: bytes, signature: bytes, public_key: Ed25519PublicKey, kid: str) -> None:
    """Raise ``SystemExit`` unless *signature* is *kid*'s over *data*."""
    sig = json.loads(signature)
    if sig.get("kid") != kid:
        raise SystemExit(f"Signed with key {sig.get('kid')!r}, not {kid!r}.")
    try:
        public_key.verify(_b64url_decode(sig["sig"]), data)
    except InvalidSignature:
        raise SystemExit("The signature does not verify.") from None
    _check_file(data)


def _check_file(data: bytes) -> None:
    body = json.loads(data)
    if body.get("format") != FORMAT:
        raise SystemExit(f"Not a {FORMAT} rate file.")
    date.fromisoformat(body["effective_from"])
    if not body.get("models"):
        raise SystemExit("The rate file lists no models.")


def _load_private_key(path: pathlib.Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SystemExit(f"{path} is not an Ed25519 private key.")
    return key


# --- command line --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("keygen", help="write a new signing key and print its public half")
    p.add_argument("out", type=pathlib.Path)

    p = commands.add_parser("build", help="build rates.json from LiteLLM's map")
    p.add_argument("out", type=pathlib.Path)
    p.add_argument("--effective-from", required=True, type=date.fromisoformat)
    p.add_argument(
        "--source",
        type=pathlib.Path,
        help="a model-cost JSON file instead of litellm's bundled map",
    )
    p.add_argument("--max-per-million", type=float, default=1000.0)

    p = commands.add_parser("sign", help="write rates.json.sig beside the file")
    p.add_argument("file", type=pathlib.Path)
    p.add_argument("--key", required=True, type=pathlib.Path)
    p.add_argument("--kid", required=True)

    p = commands.add_parser("verify", help="check a file against its .sig")
    p.add_argument("file", type=pathlib.Path)
    p.add_argument("--public-key", required=True, help="base64url raw Ed25519 public key")
    p.add_argument("--kid", required=True)

    args = parser.parse_args(argv)

    if args.command == "keygen":
        public = keygen(args.out)
        print(f"Private key written to {args.out}. Keep it off this repository.")
        print(f"Public key (base64url), for TRUSTED_RATE_KEYS: {public}")
    elif args.command == "build":
        source = (
            json.loads(args.source.read_text(encoding="utf-8")) if args.source else _litellm_map()
        )
        data, dropped = build(source, args.effective_from, args.max_per_million)
        args.out.write_bytes(data)
        count = len(json.loads(data)["models"])
        print(f"Wrote {args.out}: {count} models, effective {args.effective_from.isoformat()}.")
        for line in dropped:
            print(f"  left out {line}")
    elif args.command == "sign":
        sig = sign(args.file.read_bytes(), _load_private_key(args.key), args.kid)
        out = args.file.with_name(args.file.name + ".sig")
        out.write_bytes(sig)
        print(f"Wrote {out}.")
    elif args.command == "verify":
        public = Ed25519PublicKey.from_public_bytes(_b64url_decode(args.public_key))
        sig_path = args.file.with_name(args.file.name + ".sig")
        verify(args.file.read_bytes(), sig_path.read_bytes(), public, args.kid)
        print(f"{args.file} verifies with {args.kid}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
