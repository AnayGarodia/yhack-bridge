"""
Download WLASL pretrained I3D weights and build the class list.

Usage:
    python scripts/setup_wlasl.py            # downloads top-100 model (default)
    python scripts/setup_wlasl.py --n 300    # downloads top-300 model

What this script does:
  1. Installs gdown if missing.
  2. Downloads the WLASL_v0.3.json dataset manifest from GitHub.
  3. Extracts and saves the class list for the requested split.
  4. Downloads the pretrained I3D checkpoint from Google Drive.

All files land in:
    src/recognition/wlasl_weights/
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parent.parent
WEIGHTS_DIR  = REPO_ROOT / "src" / "recognition" / "wlasl_weights"

WLASL_JSON_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)

# Google Drive IDs for pretrained I3D RGB checkpoints (from WLASL repo README).
# These are the nslt_N models trained on the WLASL-N splits.
GDRIVE_IDS = {
    100:  "1sg8jLB9ePDz9_lV2Kx8dXdRmm2YMbWwG",   # nslt_100.pth.tar
    300:  "1qOZ3mTD2YaChIsGSgUmWQMXhh5U1B1KN",   # nslt_300.pth.tar
    1000: "1V7QxSSXqLWCFZVq3LoYN39MUzRbfkNK3",   # nslt_1000.pth.tar
    2000: "1oJC4NqBfu3cKK0TrHJgIQHi4M3_63qFT",   # nslt_2000.pth.tar
}
WEIGHTS_NAMES = {
    100: "nslt_100.pth.tar",
    300: "nslt_300.pth.tar",
    1000: "nslt_1000.pth.tar",
    2000: "nslt_2000.pth.tar",
}

# ---------------------------------------------------------------------------

def install_gdown():
    """Ensure gdown is importable."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("Installing gdown …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])


def download_wlasl_json(dest: Path) -> Path:
    """Download WLASL_v0.3.json and return its path."""
    out = dest / "WLASL_v0.3.json"
    if out.exists():
        print(f"  [skip] {out.name} already present")
        return out
    print(f"Downloading {WLASL_JSON_URL} …")
    urllib.request.urlretrieve(WLASL_JSON_URL, out)
    print(f"  → {out}")
    return out


def build_class_list(wlasl_json_path: Path, n: int, dest: Path) -> Path:
    """
    Extract the top-N most-frequent glosses and save as class_list_N.json.

    The WLASL dataset JSON is a list of objects:
        [{"gloss": "book", "instances": [...]}, ...]
    We sort by number of instances descending, take the top N, then sort
    those N glosses alphabetically (which matches the training label
    assignment used in the WLASL codebase).
    """
    out = dest / f"class_list_{n}.json"
    if out.exists():
        print(f"  [skip] {out.name} already present")
        return out

    print(f"Building class list for top-{n} glosses …")
    with open(wlasl_json_path) as f:
        dataset = json.load(f)  # list of {gloss, instances}

    # Count instances per gloss
    counts = [(entry["gloss"], len(entry["instances"])) for entry in dataset]
    counts.sort(key=lambda x: x[1], reverse=True)
    top_glosses = [g for g, _ in counts[:n]]

    # Alphabetical sort — matches WLASL training label assignment
    top_glosses.sort()

    with open(out, "w") as f:
        json.dump(top_glosses, f, indent=2)

    print(f"  → {out}  ({n} classes)")
    return out


def download_weights(n: int, dest: Path) -> Path:
    """Download pretrained I3D checkpoint for the N-class split."""
    import gdown

    name = WEIGHTS_NAMES[n]
    out  = dest / name

    if out.exists():
        print(f"  [skip] {name} already present")
        return out

    gdrive_id = GDRIVE_IDS.get(n)
    if gdrive_id is None:
        raise ValueError(f"No Google Drive ID registered for n={n}. "
                         f"Available: {list(GDRIVE_IDS.keys())}")

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"Downloading {name} from Google Drive (this may take a minute) …")
    gdown.download(url, str(out), quiet=False)

    if not out.exists() or out.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"Download failed or file is too small: {out}\n"
            "The Google Drive link may have changed. Check:\n"
            "  https://github.com/dxli94/WLASL  (Pretrained Models section)\n"
            "and update GDRIVE_IDS in this script."
        )

    print(f"  → {out}")
    return out


def verify_model(weights_path: Path, class_list_path: Path, n: int):
    """Quick forward-pass sanity check."""
    print("Verifying model loads correctly …")
    sys.path.insert(0, str(REPO_ROOT))

    import torch
    from src.recognition.wlasl_classifier import WLASLClassifier
    import numpy as np

    clf = WLASLClassifier(
        num_classes=n,
        weights_path=str(weights_path),
        class_list_path=str(class_list_path),
    )
    if not clf.ready:
        print("  WARNING: model did not load — check the weights file.")
        return

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    # Fill the buffer
    for _ in range(16):
        clf.process_frame(dummy)
    # Force an inference cycle
    clf._frame_idx = clf.stride - 1
    result = clf.process_frame(dummy)
    print(f"  Test prediction: {result}")
    print("  Model OK ✓")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Set up WLASL pretrained I3D weights")
    parser.add_argument("--n", type=int, default=100,
                        choices=[100, 300, 1000, 2000],
                        help="Number of sign classes (default: 100)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip post-download model verification")
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== WLASL setup (top-{args.n}) ===\n")

    install_gdown()
    wlasl_json = download_wlasl_json(WEIGHTS_DIR)
    class_list = build_class_list(wlasl_json, args.n, WEIGHTS_DIR)
    weights    = download_weights(args.n, WEIGHTS_DIR)

    if not args.skip_verify:
        verify_model(weights, class_list, args.n)

    print(f"\nSetup complete. Files in {WEIGHTS_DIR}:")
    for p in sorted(WEIGHTS_DIR.iterdir()):
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:<35}  {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
