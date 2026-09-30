from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def enable_vendor():
    for name in ("imfp_scripts", "imeanflow-torch", "diffusion-posterior-sampling-main"):
        path = str(ROOT / "vendor" / name)
        if path not in sys.path:
            sys.path.insert(0, path)
