import base64
import gzip
import json
import struct
import zlib

from megacube.sources.sniff import UE_PACKAGE_TAG, sniff

PAYLOAD = json.dumps({"objects": [{"className": "Build_Foundation_8x4_01_C"}], "version": 3}).encode()


def layers(path):
    return [s["layer"] for s in sniff(path)["stages"]]


def test_plain_json(tmp_path):
    f = tmp_path / "a.cbp"
    f.write_bytes(PAYLOAD)
    rep = sniff(f)
    assert layers(f) == ["file"]
    assert rep["stages"][0]["json"]["keys"] == ["objects", "version"]


def test_base64_of_zlib_json(tmp_path):
    f = tmp_path / "b.cbp"
    f.write_bytes(base64.b64encode(zlib.compress(PAYLOAD)))
    rep = sniff(f)
    assert layers(f) == ["file", "base64", "zlib"]
    assert rep["stages"][-1]["json"]["type"] == "object"


def test_gzip(tmp_path):
    f = tmp_path / "c.cbp"
    f.write_bytes(gzip.compress(PAYLOAD))
    assert layers(f) == ["file", "gzip"]


def test_unreal_chunks_after_header(tmp_path):
    body = b"\x00" * 64 + PAYLOAD
    comp = zlib.compress(body)
    chunk = UE_PACKAGE_TAG + struct.pack("<I", 0x22222222) + b"\x00" + struct.pack("<II", 131072, 0x03000000)
    chunk += struct.pack("<QQQQ", len(comp), len(body), len(comp), len(body)) + comp
    header = struct.pack("<II", 14, 52) + b"header-bytes"
    f = tmp_path / "d.sav"
    f.write_bytes(header + chunk)
    rep = sniff(f)
    assert rep["stages"][0]["looks_like_sav_header"] == {"saveHeaderType": 14, "saveVersion": 52}
    assert rep["stages"][1]["layer"].startswith("unreal-compressed-chunks")
    assert rep["stages"][1]["size"] == len(body)


def test_plain_word_is_not_base64(tmp_path):
    f = tmp_path / "e.txt"
    f.write_bytes(b"HelloWorld")
    assert layers(f) == ["file"]
