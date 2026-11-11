"""Download only the locked public development bank's 72 EDF files."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    hashes=json.loads((Path(__file__).resolve().parents[1]/"configs/physionet_files_sha256.json").read_text())
    for relative,digest in hashes.items():
        path=(a.output/relative).resolve()
        if not path.is_relative_to(a.output.resolve()): raise ValueError("Unsafe file path")
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise ValueError(f"Hash mismatch: {relative}")
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(".part")
        with urlopen("https://physionet.org/files/eegmmidb/1.0.0/"+relative,timeout=120) as src, temporary.open("wb") as dst:
            while chunk:=src.read(1024*1024): dst.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=digest: raise ValueError(f"Downloaded hash mismatch: {relative}")
        temporary.replace(path)
        print(relative,flush=True)

if __name__=="__main__": main()
