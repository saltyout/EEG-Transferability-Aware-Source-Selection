"""Best-effort tracked-file and reachable-history release scan; no remote writes."""
import json
import re
import subprocess
from pathlib import Path


def git(*args):
    return subprocess.check_output(["git", *args])


def main():
    root=Path(__file__).resolve().parents[1]
    patterns={
        "private_key":r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "github_token":r"gh[pousr]_[A-Za-z0-9]{30,}",
        "github_pat":r"github_pat_[A-Za-z0-9_]{30,}",
        "aws_key":r"AKIA[A-Z0-9]{16}",
        "windows_absolute":r"[A-Za-z]:[\\/](?:Users|documents)[\\/]",
        "server_home":r"/home/[A-Za-z0-9_.-]+/",
        "ipv4":r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])",
    }
    forbidden={".npz",".npy",".mat",".edf",".pth",".pt",".ckpt",".safetensors",".pem",".key"}
    findings=[]
    def check(data,name):
        if len(data)>1_000_000: findings.append(dict(file=name,rule="over_1MB"))
        if Path(name).suffix.lower() in forbidden: findings.append(dict(file=name,rule="excluded_extension"))
        text=data.decode("utf-8",errors="replace")
        for key,pattern in patterns.items():
            if re.search(pattern,text): findings.append(dict(file=name,rule=key))
    paths=git("ls-files","-z").decode().split("\0")[:-1]
    for name in paths: check((root/name).read_bytes(),name)
    objects=git("rev-list","--objects","--all").decode().splitlines()
    count=0
    for line in objects:
        oid,_,name=line.partition(" ")
        if git("cat-file","-t",oid).strip()==b"blob":
            check(git("cat-file","blob",oid),"history/"+name);count+=1
    print(json.dumps(dict(tracked_files=len(paths),history_blobs=count,findings=findings,
        scope="Tracked working files and all reachable history blobs; pattern scan is not a guarantee of absence of sensitive information"),indent=2))
    if findings: raise SystemExit(1)

if __name__=="__main__": main()
