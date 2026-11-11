"""Generate explicitly synthetic features; no scientific evidence."""
import argparse
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
rng = np.random.default_rng(17)
y = np.tile(np.arange(45) % 2, 17)
x = rng.normal(size=(len(y), 8))
x[:, 0] += y
a.output.parent.mkdir(parents=True, exist_ok=True)
np.savez(a.output, X=x, y=y, subjects=np.repeat([f"synthetic{i:02d}" for i in range(17)],45), bank_indices=np.arange(len(y)))
