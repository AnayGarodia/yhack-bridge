# Bridge 🤟🔊

**Turns sign language into speech and speech into sign language, in real time.**

Built at [YHack 2026](https://yhack.org) (Yale, March 28-29).

---

## How It Works

### ASL → English Speech
A deaf person signs into a camera. Bridge recognizes their signs, converts ASL grammar to natural English, and speaks the translation aloud.

```
Camera → MediaPipe hand tracking → Sign recognition (I3D + fingerspelling)
→ ASL gloss buffer → Lava API (ASL grammar → English) → ElevenLabs TTS
→ Hearing person hears natural speech
```

### English Speech → ASL
A hearing person speaks. Bridge transcribes their words, converts English to ASL grammar, and animates the signs through an avatar.

```
Microphone → Whisper STT → English text → Lava API (English → ASL gloss)
→ SVG avatar animates signs → Deaf person sees signs on screen
```

### Google Meet Integration
Bridge can join Google Meet calls as a participant, working toward real-time interpretation in any video call.

---

## Architecture

```
                        ┌─────────────────┐
                        │     Bridge      │
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
     ┌────────▼────────┐ ┌──────▼──────┐ ┌─────────▼────────┐
     │  ASL → English  │ │  English →  │ │  Google Meet      │
     │                 │ │  ASL        │ │  Integration      │
     └────────┬────────┘ └──────┬──────┘ └─────────┬────────┘
              │                 │                   │
    ┌─────────┴──────┐   ┌─────┴──────┐   ┌───────┴───────┐
    │ MediaPipe      │   │ Whisper    │   │ Virtual Camera│
    │ Hand Tracking  │   │ STT       │   │ + Meet Bot    │
    ├────────────────┤   ├────────────┤   └───────────────┘
    │ I3D Word       │   │ Lava API  │
    │ Recognition    │   │ Grammar   │
    ├────────────────┤   ├────────────┤
    │ Fingerspelling │   │ SVG Avatar│
    │ Classifier     │   │ Animation │
    ├────────────────┤   └────────────┘
    │ Lava API       │
    │ Grammar        │
    ├────────────────┤
    │ ElevenLabs TTS │
    └────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Hand tracking | MediaPipe Holistic |
| Word-level recognition | I3D trained on WLASL100 (100 signs) |
| Fingerspelling | Rule-based geometric classifier (A-Z) |
| Speech-to-text | Whisper via faster-whisper |
| Text-to-speech | ElevenLabs + pyttsx3 offline fallback |
| Grammar translation | Lava API → GPT-4o-mini |
| ASL avatar | Custom SVG animation pipeline |
| Web interface | Flask + SocketIO |
| Video call integration | Google Meet bot + virtual camera |

---

## Project Structure

```
bridge/
├── src/
│   ├── app.py                          # Flask web UI
│   ├── recognition/
│   │   ├── asl_classifier.py           # Rule-based fingerspelling (A-Z)
│   │   ├── hand_tracker.py             # MediaPipe hand landmarks
│   │   ├── holistic_tracker.py         # MediaPipe Holistic (543 landmarks)
│   │   ├── sign_router.py             # Routes between word + fingerspell modes
│   │   ├── landmark_classifier.py      # Landmark-based classification
│   │   ├── tflite_classifier.py        # TFLite model integration
│   │   └── train_landmark_model.py     # Training pipeline
│   ├── speech/
│   │   ├── tts.py                      # ElevenLabs TTS + offline fallback
│   │   ├── stt.py                      # Whisper speech-to-text
│   │   └── pipeline.py                 # Token buffer + sentence completion
│   ├── translation/
│   │   ├── text_smoother.py            # ASL gloss → English (Lava API)
│   │   ├── english_to_signs.py         # English → ASL gloss (Lava API)
│   │   └── sign_decoder.py            # Sign sequence decoding
│   ├── avatar/
│   │   ├── avatar_controller.py        # Avatar state management
│   │   ├── avatar_renderer.py          # Render avatar to frames
│   │   ├── sign_animator.py            # Sign animation sequencing
│   │   ├── sign_library.py             # Sign animation data (~100 signs)
│   │   ├── sign_database.py            # Sign lookup and retrieval
│   │   ├── svg_generator.py            # SVG keyframe generation
│   │   ├── hand_renderer.py            # Hand mesh rendering
│   │   ├── animation_engine.py         # Chained SVG animation engine
│   │   └── recorder.py                 # Animation recording utilities
│   └── output/
│       ├── bridge_camera.py            # Camera management
│       ├── virtual_camera.py           # Virtual camera for Meet
│       ├── meet_session.py             # Google Meet bot session
│       └── frame_composer.py           # Compose output frames
├── models/
│   └── archived/
│       ├── asl100/                     # Pretrained I3D (100 signs, 65.89% top-1)
│       ├── asl1000/                    # Pretrained I3D (1000 signs)
│       └── asl2000/                    # Pretrained I3D (2000 signs)
└── templates/
    └── index.html                      # Web UI
```

---

## Setup

```bash
git clone https://github.com/YOUR_REPO/bridge.git
cd bridge
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables
```bash
export ELEVENLABS_API_KEY="your_key"
export LAVA_API_KEY="your_key"
```

### Run
```bash
python src/app.py
```

---

## Acknowledgments

- [WLASL](https://github.com/dxli94/WLASL) for the sign language dataset and pretrained I3D weights
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) for hand tracking
- [ElevenLabs](https://elevenlabs.io) for text-to-speech
- [Lava](https://www.lava.ai) for LLM API gateway
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition

---

## Team

Built by Anay and Pratyush at YHack 2026. First hackathon for both of us.
