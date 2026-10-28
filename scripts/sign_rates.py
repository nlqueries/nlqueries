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
    update  DIR --kid KID (--key KEY.pem | --key-env VAR) [--public-key BASE64URL]

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

``update`` is what ``.github/workflows/rates.yml`` runs every day. It does
``build`` from LiteLLM's live map, effective today, and then:
- if no model's rates changed, it writes nothing, so the published file keeps
  its effective date;
- if they did, it signs and writes ``DIR/rates.json`` and its ``.sig``.
The published file it compares against counts only if it verifies with the
signing key; one that does not is replaced.
It writes nothing either if more than ``--max-removed`` of the published models
(default a tenth) would go, which is a broken map more often than a real change;
or if the key is not the one ``--public-key`` names, which would publish a file
every deployment refuses.
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
import urllib.error
import urllib.parse
import urllib.request
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
    return _private_key(path.read_bytes(), str(path))


def _private_key_from_env(variable: str) -> Ed25519PrivateKey:
    """The key whose PEM text is in *variable*, so CI never writes it to disk."""
    pem = os.environ.get(variable, "").strip()
    if not pem:
        raise SystemExit(f"{variable} is empty or unset; it should hold the PEM signing key.")
    return _private_key(pem.encode("ascii"), variable)


def _private_key(pem: bytes, where: str) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SystemExit(f"{where} is not an Ed25519 private key.")
    return key


# --- update --------------------------------------------------------------------

#: LiteLLM's price map on its main branch, where new prices appear first.
LITELLM_MAP_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
#: Far above the map's size (3 MB in October 2026); a body past this is not the map.
MAX_MAP_BYTES = 64 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Turn a redirect into an error: the map is fetched from the URL given."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def fetch_map(url: str, opener: Any = None, max_bytes: int = MAX_MAP_BYTES) -> dict[str, Any]:
    """The model-cost map at *url*: https only, no redirect, at most *max_bytes*."""
    if urllib.parse.urlsplit(url).scheme != "https":
        raise SystemExit(f"The price map must be fetched over https, not {url!r}.")
    opener = opener or urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(url, timeout=60) as response:
            status = response.status
            body = response.read(max_bytes + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise SystemExit(f"Could not fetch {url}: {exc}") from None
    if status != 200:
        raise SystemExit(f"Could not fetch {url}: HTTP {status}.")
    if len(body) > max_bytes:
        raise SystemExit(f"{url} is larger than {max_bytes} bytes; refusing it.")
    try:
        source = json.loads(body)
    except ValueError:
        raise SystemExit(f"{url} is not JSON.") from None
    if not isinstance(source, dict):
        raise SystemExit(f"{url} is not a model-cost map.")
    return source


def _named(label: str, names: list[str], shown: int = 20) -> str:
    more = f", and {len(names) - shown} more" if len(names) > shown else ""
    return f"  {label}: {', '.join(names[:shown])}{more}"


def _published_models(
    target: pathlib.Path, sig_path: pathlib.Path, public_key: Ed25519PublicKey, kid: str
) -> tuple[dict[str, Any], str | None]:
    """The models of the published file, if it verifies with this key; otherwise
    none, and why.

    The published file is the baseline for "unchanged" and for ``--max-removed``,
    and the branch it sits on is not protected. So a file this key did not sign
    counts as no file: the update replaces it with one that verifies, and
    invented entries in it cannot hold a real update back (from #355's review).
    """
    if not target.exists():
        return {}, None
    if not sig_path.exists():
        return {}, "it has no signature"
    try:
        data = target.read_bytes()
        verify(data, sig_path.read_bytes(), public_key, kid)
        return dict(json.loads(data)["models"]), None
    except SystemExit as exc:
        return {}, str(exc)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return {}, f"it could not be read ({exc.__class__.__name__})"


def update(
    directory: pathlib.Path,
    source: dict[str, Any],
    key: Ed25519PrivateKey,
    kid: str,
    *,
    today: date,
    public_key: str | None = None,
    max_per_million: float = 1000.0,
    max_removed: float = 0.1,
) -> tuple[bool, list[str]]:
    """Rebuild ``directory/rates.json`` from *source*, signed, if any model's rates
    changed. Returns whether it wrote, and a report whose first line suits a
    commit subject."""
    if not kid:
        raise SystemExit("The key id is empty.")
    if public_key == "":
        raise SystemExit("--public-key is empty.")
    if public_key is not None and _b64url(key.public_key().public_bytes_raw()) != public_key:
        raise SystemExit("The signing key is not the one --public-key names; nothing was written.")
    data, dropped = build(source, today, max_per_million)
    new = json.loads(data)["models"]
    target = directory / "rates.json"
    sig_path = directory / "rates.json.sig"
    old, unverified = _published_models(target, sig_path, key.public_key(), kid)

    added = sorted(set(new) - set(old), key=str.lower)
    removed = sorted(set(old) - set(new), key=str.lower)
    changed = sorted((n for n in set(new) & set(old) if new[n] != old[n]), key=str.lower)
    left_out = [f"  left out {line}" for line in dropped]
    if unverified is not None:
        left_out.insert(0, f"  replaced the published file, which did not verify: {unverified}")
    if old and not (added or removed or changed):
        return False, [f"Rates unchanged: {len(new)} models.", *left_out]
    if old and len(removed) > max_removed * len(old):
        raise SystemExit(
            f"{len(removed)} of the {len(old)} published models would go, more than "
            f"--max-removed {max_removed:g} allows; nothing was written.\n"
            + _named("would go", removed)
        )

    signature = sign(data, key, kid)
    verify(data, signature, key.public_key(), kid)
    directory.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    sig_path.write_bytes(signature)
    report = [
        f"Rates for {len(new)} models, effective {today.isoformat()}: "
        f"{len(added)} added, {len(removed)} removed, {len(changed)} changed",
    ]
    for label, names in (("added", added), ("removed", removed), ("changed", changed)):
        if names:
            report.append(_named(label, names))
    return True, [*report, *left_out]


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

    p = commands.add_parser(
        "update", help="rebuild DIR/rates.json from LiteLLM's live map if any rate changed"
    )
    p.add_argument("directory", type=pathlib.Path)
    p.add_argument("--kid", required=True)
    key = p.add_mutually_exclusive_group(required=True)
    key.add_argument("--key", type=pathlib.Path)
    key.add_argument("--key-env", metavar="VAR", help="an environment variable holding the PEM key")
    p.add_argument("--public-key", help="refuse unless the key's public half is this (base64url)")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--source-url", default=LITELLM_MAP_URL)
    source.add_argument(
        "--source", type=pathlib.Path, help="a model-cost JSON file instead of fetching"
    )
    p.add_argument("--max-per-million", type=float, default=1000.0)
    p.add_argument(
        "--max-removed",
        type=float,
        default=0.1,
        help="refuse if more than this fraction of the published models would go",
    )

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
    elif args.command == "update":
        signing_key = (
            _load_private_key(args.key) if args.key else _private_key_from_env(args.key_env)
        )
        source_map = (
            json.loads(args.source.read_text(encoding="utf-8"))
            if args.source
            else fetch_map(args.source_url)
        )
        _, report = update(
            args.directory,
            source_map,
            signing_key,
            args.kid,
            today=datetime.now(UTC).date(),
            public_key=args.public_key,
            max_per_million=args.max_per_million,
            max_removed=args.max_removed,
        )
        print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
