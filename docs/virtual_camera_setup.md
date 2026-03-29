# Virtual Camera Setup

Bridge uses a virtual camera to pipe its processed video feed (webcam + ASL transcription overlays) into Google Meet, Zoom, or any video call app as if it were a real webcam.

## Install the Python package

```
pip install pyvirtualcam
```

---

## macOS

Bridge uses the **OBS Virtual Camera** backend on macOS.

1. Install OBS Studio:
   ```
   brew install --cask obs
   ```
2. Open OBS at least once.
3. Go to **Tools → Start Virtual Camera**, then stop it. This registers the virtual camera plugin with the system.
4. You do **not** need OBS running while Bridge is active — `pyvirtualcam` talks to the plugin directly.

## Linux

Bridge uses the **v4l2loopback** kernel module on Linux.

1. Install the module:
   ```
   sudo apt install v4l2loopback-dkms
   ```
2. Load it (run once after each reboot, or add to `/etc/modules`):
   ```
   sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Bridge Virtual Cam" exclusive_caps=1
   ```
3. Verify the device exists:
   ```
   ls /dev/video*
   ```
   You should see `/dev/video10`.

## Windows

1. Install OBS Studio from <https://obsproject.com>.
2. Open OBS at least once.
3. Go to **Tools → Start Virtual Camera**, then stop it.
4. The virtual camera should now appear in other apps.

---

## Selecting the virtual camera in Google Meet

1. Join or start a Google Meet call.
2. Click the **three-dot menu** (⋮) → **Settings** → **Video**.
3. Under **Camera**, select **OBS Virtual Camera** (macOS/Windows) or **Bridge Virtual Cam** (Linux).
4. You should see the Bridge output with transcription overlays.

The same approach works in Zoom, Teams, Discord, and Photo Booth (macOS).

---

## Testing

Run the standalone test to verify the pipeline before wiring in ASL recognition:

```
python scripts/test_virtual_cam.py
python scripts/test_virtual_cam.py --camera 1    # GX10 uses camera index 1
```

This opens your real webcam, overlays red test text, and pipes it to the virtual camera for 30 seconds. Open Google Meet or Photo Booth and select the virtual camera to confirm it works.
