Bridge — Real-time ASL Sign Language Translator
Project Context
YHack 2026 hackathon (Yale, March 28-29). Team of 2 (Dorian + Anay). First hackathon.
Hardware: ASUS Ascent GX10 (NVIDIA GB10 supercomputer). Everything runs locally, no cloud.
Deadline: hacking ends 11 AM Sunday March 29. Video submission by 11:30 AM. Live demos at 12 PM.
Tracks Targeting

Societal Impact (ASUS, primary)
Lava API
Grand Prize
Best First Hack
ElevenLabs
Best UI/UX
Most on Theme/Love
Most Creative

Architecture Overview
Two-layer ASL recognition system:

Fingerspelling (A-Z): Rule-based classifier using MediaPipe hand landmark geometry. FALLBACK layer — always available.
Word-level signs: I3D model pretrained on WLASL2000 dataset. Up to 2000 sign glosses. PRIMARY recognition layer.

Bidirectional communication:

ASL → Text → Speech (ElevenLabs TTS)
Speech → Text (Whisper) → display for deaf user

Tech Stack
Python 3.10, OpenCV, MediaPipe, PyTorch, Flask, ElevenLabs API, Lava API, Whisper
Current File Structure
~/Desktop/bridge/
├── src/
│ └── recognition/
│ ├── hand_tracker.py # MediaPipe hand tracking — WORKING
│ └── asl_classifier.py # Rule-based A-Z classifier — WORKING (663 lines)
├── venv/ # Python 3.10 virtualenv
└── CLAUDE.md # This file
What's Done

MediaPipe hand tracking via HandTracker class (camera index 1 on this machine)
Rule-based ASL fingerspelling classifier (A-Z) in src/recognition/asl_classifier.py

Extracts geometric features from 21 normalized landmarks (wrist-relative pixel coords)
Per-finger extension/curl detection, thumb position, fingertip spread, palm distances
26 rule functions, one per letter
5-frame rolling prediction smoothing
Color-coded confidence overlay (green >60%, yellow >40%, red <40%)
J and Z have low confidence by design (require motion detection)
Tested: imports clean, synthetic data test passes, webcam test ran successfully

What Needs To Be Built (priority order)
Phase 1: Core Recognition (CRITICAL)

I3D word-level model setup

Download pretrained I3D weights from WLASL repo (github.com/dxli94/WLASL)
Start with top 100-300 glosses, not full 2000
Inference on buffered 16-32 frame clips
GX10 GPU should handle ~30fps inference

State machine routing between fingerspelling and word-level

Held hand shape → route to fingerspelling
Motion detected → buffer frames → route to I3D

Tune fingerspelling rules — fist letters (A, E, M, N, S, T) may confuse each other

Phase 2: Speech Integration

ElevenLabs TTS — recognized sign text → speech output via API
Whisper STT — microphone input → text display (use whisper-large-v3, GX10 can handle it)
Lava API integration — contextual sentence completion from partial sign sequences, or second opinion on ambiguous signs

Phase 3: Polish

Flask web UI — camera feed left, recognized text center, conversation log right. Dark theme, clean typography.
Avatar for Google Meet (STRETCH GOAL, cut if behind) — virtual camera via pyvirtualcam rendering a signing avatar

Phase 4: Demo Prep (Sunday 8-11 AM)

Record 60-90 second video submission
Practice 3-minute live demo with canned signs the model handles well

Key Technical Details

Camera index is 1 (not 0) on this machine
MediaPipe returns 21 hand landmarks, normalized to wrist-relative pixel coordinates
The HandTracker class in hand_tracker.py handles all MediaPipe setup and landmark extraction
ASLClassifier takes normalized_landmarks (list of 21 (x,y,z) tuples) and returns (letter, confidence)
Virtualenv at ./venv with flask, mediapipe, opencv already installed

Side Projects (during breaks)

Storybook: 30-minute build. Describe a memory → children's book page. For Most on Theme/Love track.
Snap Lenses: 2-3 lenses for Snap Lens Studio track ($1,500/$1,000/$500 prizes). Ideas: ASL alphabet teacher, caption-what-I'm-signing.
