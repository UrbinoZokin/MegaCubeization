"""Identify how an unknown build file is encoded, without assuming its schema.

Used for the SCIM megaprint (``.cbp``), whose format isn't publicly documented. The sniffer only
reports facts it can check (magic bytes, compression layers that decode cleanly, JSON that
parses, text vs binary), so we can choose a parsing strategy from evidence, not guesses.
"""
from __future__ import annotations

import base64
import binascii
import gzip
import json
import math
import re
import struct
import zlib
from collections import Counter
from pathlib import Path

UE_PACKAGE_TAG = b"\xc1\x83\x2a\x9e"  # 0x9E2A83C1 little-endian, starts every compressed save chunk
_B64 = re.compile(rb"^[A-Za-z0-9+/=\s]+$")
_B64URL = re.compile(rb"^[A-Za-z0-9_\-=\s]+$")


def entropy(data: bytes) -> float:
    """Shannon entropy in bits/byte (close to 8 suggests compressed or encrypted data)."""
    if not data:
        return 0.0
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in Counter(data).values())


def _printable_ratio(data: bytes) -> float:
    sample = data[:65536]
    if not sample:
        return 0.0
    ok = sum(1 for b in sample if 32 <= b < 127 or b in (9, 10, 13))
    return ok / len(sample)


def _json_summary(obj, depth: int = 0) -> dict:
    if isinstance(obj, dict):
        out = {"type": "object", "keys": list(obj.keys())[:40], "n_keys": len(obj)}
        if depth < 1:
            out["children"] = {k: _json_summary(v, depth + 1) for k, v in list(obj.items())[:12]}
        return out
    if isinstance(obj, list):
        out = {"type": "array", "length": len(obj)}
        if obj and depth < 2:
            out["first"] = _json_summary(obj[0], depth + 1)
        return out
    return {"type": type(obj).__name__}


def _ue_chunks(data: bytes, start: int) -> bytes | None:
    """Decode Unreal compressed chunks (the .sav body layout) starting at ``start``."""
    out, off = [], start
    try:
        while off < len(data):
            if data[off:off + 4] != UE_PACKAGE_TAG:
                return None
            # tag, 0x22222222, 1 byte, max chunk size, algorithm, then (compressed, uncompressed) x2 as u64
            hdr_len = 4 + 4 + 1 + 4 + 4 + 8 * 4
            comp, _uncomp = struct.unpack_from("<QQ", data, off + 17)
            out.append(zlib.decompress(data[off + hdr_len: off + hdr_len + comp]))
            off += hdr_len + comp
    except (struct.error, zlib.error):
        return None
    return b"".join(out)


def _recognisable(data: bytes) -> bool:
    return (data[:2] == b"\x1f\x8b" or data[:4] == UE_PACKAGE_TAG
            or (len(data) > 2 and data[0] & 0x0F == 8 and (data[0] << 8 | data[1]) % 31 == 0)
            or data.strip()[:1] in (b"{", b"[") or _printable_ratio(data) > 0.95)


def _try_layers(data: bytes) -> list[tuple[str, bytes]]:
    """Return every decoding that applies cleanly to ``data``: (layer name, decoded bytes)."""
    found = []
    if data[:2] == b"\x1f\x8b":
        try:
            found.append(("gzip", gzip.decompress(data)))
        except OSError:
            pass
    if len(data) > 2 and data[0] & 0x0F == 8 and (data[0] << 8 | data[1]) % 31 == 0:
        try:
            found.append(("zlib", zlib.decompress(data)))
        except zlib.error:
            pass
    if not found:
        try:
            d = zlib.decompressobj(-15)
            raw = d.decompress(data)
            if d.eof and raw:
                found.append(("raw-deflate", raw))
        except zlib.error:
            pass
    if data[:4] == UE_PACKAGE_TAG:
        dec = _ue_chunks(data, 0)
        if dec is not None:
            found.append(("unreal-compressed-chunks", dec))
    stripped = data.strip()
    if len(stripped) >= 8 and (_B64.match(stripped[:4096]) or _B64URL.match(stripped[:4096])):
        for name, fn in (("base64", base64.b64decode), ("base64url", base64.urlsafe_b64decode)):
            try:
                dec = fn(stripped + b"=" * (-len(stripped) % 4))
            except (binascii.Error, ValueError):
                continue
            # Plain words also look like base64; keep the layer only if it decodes to something recognisable.
            if _recognisable(dec):
                found.append((name, dec))
                break
    return found


def _describe(data: bytes) -> dict:
    info = {
        "size": len(data),
        "magic_hex": data[:16].hex(" "),
        "entropy_bits_per_byte": round(entropy(data[:1 << 20]), 3),
        "printable_ratio": round(_printable_ratio(data), 3),
    }
    stripped = data.strip()
    if stripped[:1] in (b"{", b"["):
        try:
            info["json"] = _json_summary(json.loads(stripped.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            info["json_error"] = str(exc)[:200]
    if len(data) >= 8:
        header_type, save_version = struct.unpack_from("<II", data, 0)
        if header_type in (13, 14) and 20 <= save_version <= 100:
            info["looks_like_sav_header"] = {"saveHeaderType": header_type, "saveVersion": save_version}
    pos = data.find(UE_PACKAGE_TAG)
    if pos >= 0:
        info["first_unreal_chunk_tag_offset"] = pos
    return info


def sniff(path: str | Path, max_depth: int = 4) -> dict:
    """Peel off encoding layers recursively and describe each stage."""
    data = Path(path).read_bytes()
    stages = [{"layer": "file", **_describe(data)}]
    for _ in range(max_depth):
        layers = _try_layers(data)
        # A .sav-style file has a plain header before the first chunk tag: decode from there.
        tag = stages[-1].get("first_unreal_chunk_tag_offset")
        if not layers and tag:
            dec = _ue_chunks(data, tag)
            if dec is not None:
                layers = [(f"unreal-compressed-chunks@{tag}", dec)]
        if not layers:
            break
        name, data = layers[0]
        stages.append({"layer": name, **_describe(data)})
        if len(layers) > 1:
            stages[-1]["other_candidate_layers"] = [n for n, _ in layers[1:]]
    return {"path": str(path), "stages": stages}
