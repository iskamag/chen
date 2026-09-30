#!/usr/bin/env python3
"""Fetch pinned official offline opponents/runner into an isolated local directory."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

PINS = {
    "fastchess": {
        "version": "v1.8.2-alpha",
        "url": "https://github.com/Disservin/fastchess/releases/download/v1.8.2-alpha/fastchess-linux-x86-64.tar",
        "sha256": "1003f920bebe841acdab5e6d7871e93171a4586857b3ec91c21ef5777f26ff96",
        "archive": "fastchess.tar",
        "binary": "fastchess-linux-x86-64/fastchess",
        "license": "MIT; bundled dependency licenses retained in archive",
    },
    "stockfish": {
        "version": "sf_19",
        "url": "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-linux-x86-64-universal.tar.gz",
        "sha256": "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f",
        "archive": "stockfish.tar.gz",
        "binary": "stockfish/stockfish-linux-x86-64-universal",
        "license": "GPL-3.0; official source, networks and licensing retained in archive",
    },
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(archive, destination):
    """Refuse links/special files and escaping paths even for hash-pinned archives."""
    root = Path(destination).resolve()
    with tarfile.open(archive, "r:*") as stream:
        members = stream.getmembers()
        if sum(m.size for m in members) > 1024 * 1024 * 1024:
            raise ValueError("oversized extracted match-tool archive")
        for member in members:
            if not (member.isfile() or member.isdir()):
                raise ValueError("unsupported archive link or special file")
            if not (root / member.name).resolve().is_relative_to(root):
                raise ValueError("archive path escapes destination")
        stream.extractall(root, members=members, filter="data")


def fetch(output, archives=None):
    output = Path(output).resolve()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("these pinned release artifacts require Linux x86_64")
    if output.exists():
        manifest = json.loads((output / "manifest.json").read_text())
        for name, pin in PINS.items():
            record = manifest["tools"][name]
            if record["pin"] != pin or sha(output / pin["binary"]) != record["binary_sha256"]:
                raise ValueError("existing tool installation does not match pins")
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".chen-tools-", dir=output.parent) as temporary:
        stage = Path(temporary) / "installation"
        stage.mkdir()
        records = {}
        for name, pin in PINS.items():
            archive = stage / pin["archive"]
            if archives is not None:
                shutil.copyfile(Path(archives) / pin["archive"], archive)
            else:
                with urllib.request.urlopen(pin["url"], timeout=60) as response, archive.open("wb") as stream:
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > 512 * 1024 * 1024:
                            raise ValueError("oversized match-tool download")
                        stream.write(chunk)
            if sha(archive) != pin["sha256"]:
                raise ValueError(f"{name} official release archive SHA256 mismatch")
            extract(archive, stage)
            binary = stage / pin["binary"]
            if name == "fastchess":
                version = subprocess.check_output([str(binary), "-version"], text=True, timeout=10).strip()
            else:
                version = subprocess.run([str(binary)], input="uci\nquit\n", text=True,
                                         capture_output=True, check=True, timeout=20).stdout
            records[name] = {"pin": pin, "binary_sha256": sha(binary), "reported_version": version}
        manifest = {"kind": "pinned_official_offline_match_tools", "tools": records,
                    "scope": "Offline references and runner, not engine implementation or Lichess calibration"}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/match-tools")
    parser.add_argument("--archives", help="use already downloaded hash-pinned archives offline")
    args = parser.parse_args()
    print(json.dumps(fetch(args.out, args.archives), indent=2))


if __name__ == "__main__":
    main()
