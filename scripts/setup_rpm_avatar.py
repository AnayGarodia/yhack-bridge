#!/usr/bin/env python3
"""
Download a Ready Player Me GLB avatar for the ASL avatar system.

Usage:
    python scripts/setup_rpm_avatar.py
    python scripts/setup_rpm_avatar.py --url "https://models.readyplayer.me/YOUR_ID.glb"
"""

import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_URL = (
    "https://models.readyplayer.me/638df693d72bffc6fa17943c.glb"
    "?morphTargets=ARKit&textureAtlas=1024"
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "avatar.glb")


def download_avatar(url: str, output: str) -> bool:
    os.makedirs(os.path.dirname(output), exist_ok=True)

    if os.path.exists(output):
        size_mb = os.path.getsize(output) / 1e6
        print(f"[rpm] avatar.glb already exists ({size_mb:.1f} MB)")
        print(f"[rpm] Delete it and re-run to re-download.")
        return True

    print(f"[rpm] Downloading avatar...")
    print(f"[rpm] URL: {url[:80]}...")

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / 1e6
                print(f"\r[rpm] {pct}% ({mb:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, output, reporthook=_progress)
        print()

        size_mb = os.path.getsize(output) / 1e6
        print(f"[rpm] Saved to {output} ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"\n[rpm] Download failed: {e}")
        print(f"[rpm] Try manually downloading from:")
        print(f"      {url}")
        print(f"      Save to: {output}")
        return False

    # Verify with trimesh
    try:
        import trimesh
        scene = trimesh.load(output)
        if isinstance(scene, trimesh.Scene):
            meshes = list(scene.geometry.keys())
            print(f"[rpm] GLB loaded: {len(meshes)} meshes")

            # Print bone/joint names if available
            if hasattr(scene, 'graph') and hasattr(scene.graph, 'nodes'):
                nodes = list(scene.graph.nodes)
                print(f"[rpm] Scene nodes: {len(nodes)}")
                for n in sorted(nodes)[:30]:
                    print(f"       {n}")
                if len(nodes) > 30:
                    print(f"       ... and {len(nodes) - 30} more")
        else:
            print(f"[rpm] GLB loaded as single mesh")
        return True

    except Exception as e:
        print(f"[rpm] Warning: trimesh couldn't verify GLB: {e}")
        print(f"[rpm] File may still be usable — continuing.")
        return True


def main():
    parser = argparse.ArgumentParser(description="Download Ready Player Me avatar")
    parser.add_argument("--url", default=DEFAULT_URL, help="RPM avatar URL")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output path")
    args = parser.parse_args()

    ok = download_avatar(args.url, args.output)
    if ok:
        print("\n[rpm] Setup complete!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
