# Ready Player Me Avatar Setup

## Create a Custom Avatar

1. Go to [readyplayer.me](https://readyplayer.me) (free, no account required for basic avatars)
2. Click "Create Avatar"
3. Customize your avatar's appearance
4. When done, copy the avatar URL from the browser (looks like `https://models.readyplayer.me/YOUR_ID.glb`)

## Download the Avatar

```bash
python scripts/setup_rpm_avatar.py
# or with a custom avatar:
python scripts/setup_rpm_avatar.py --url "https://models.readyplayer.me/YOUR_ID.glb?morphTargets=ARKit&textureAtlas=1024"
```

This saves the avatar to `models/avatar.glb`.

## Replace the Default Avatar

Simply replace `models/avatar.glb` with any RPM-exported GLB file. The renderer will discover the bone names automatically on load.

## Platform Setup for 3D Rendering

The 3D renderer uses pyrender with OSMesa for offscreen rendering:

- **macOS**: `brew install mesa` (or use the skeleton fallback — works great)
- **Linux**: `sudo apt-get install libosmesa6-dev`
- **All**: Set env var `PYOPENGL_PLATFORM=osmesa` for headless rendering

If OSMesa isn't available, the system automatically falls back to a high-quality skeleton renderer using OpenCV.

## License

Ready Player Me avatars are free for commercial use under their standard terms.
