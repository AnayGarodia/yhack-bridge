Bridge — Real-time ASL Sign Language Translator

Project Context
YHack 2026 hackathon (Yale, March 28-29). Team of 2 (Dorian + Anay). First hackathon.
Hardware: ASUS Ascent GX10 (NVIDIA GB10 supercomputer). Everything runs locally, no cloud.
Deadline: hacking ends 11 AM Sunday March 29. Video submission by 11:30 AM. Live demos at 12 PM.

How to Run
    python main.py                  # default camera 0
    python main.py --camera 1       # GX10 uses camera index 1
    python main.py --threshold 0.40 # confidence threshold (default 40%)

Controls: q=quit, r=reset sentence, SPACE=add space, f=toggle fingerspell mode

Architecture
Two-layer recognition system:

1. Word-level (PRIMARY): hoyso48's 1st-place Kaggle TFLite model
   - 250 ASL signs (hello, goodbye, please, thankyou, yes, no, help, eat, etc.)
   - 1st place in Google Isolated Sign Language Recognition competition
   - Input: variable-length (T, 543, 3) MediaPipe Holistic landmarks
   - Model internally selects 118 landmarks (lips + hands + nose + eyes)
   - Preprocessing: center on nose (landmark 17), normalize, compute velocity/acceleration
   - Architecture: Conv1D + Transformer blocks → 250-class output
   - TFLite inference (~10-50ms per prediction)
   - Confidence threshold: 40%, requires 2 consecutive agreeing predictions
   - Hand-presence gating: only buffers when hands detected

2. Fingerspelling (FALLBACK): Rule-based classifier using MediaPipe hand landmarks
   - A-Z from 21 hand landmark geometry
   - Press 'f' to toggle fingerspelling mode

Pipeline:
  Webcam frame
    → MediaPipe Holistic (543 landmarks per frame)
    → Buffer landmarks (up to 384 frames)
    → TFLite inference every 15 frames
    → Softmax → confidence filter → consecutive agreement
    → Emit to sentence buffer

File Structure
    main.py                             # Entry point — run this
    requirements.txt                    # Dependencies
    src/
      recognition/
        tflite_classifier.py            # TFLite model wrapper (primary)
        holistic_tracker.py             # MediaPipe holistic tracking
        hand_tracker.py                 # MediaPipe hand tracking (fingerspelling)
        asl_classifier.py              # Rule-based A-Z fingerspelling
    models/
      model.tflite                      # 1st-place Kaggle TFLite model (11 MB)
      sign_to_prediction_index_map.json # 250-sign label mapping
    _archive/                           # Old files (I3D, LSTM, etc.)

Key Technical Details
- Camera index is 1 (not 0) on the GX10
- TFLite model includes all preprocessing internally — just feed raw (T, 543, 3) landmarks
- MediaPipe Holistic extracts 543 landmarks: face (468) + left hand (21) + pose (33) + right hand (21)
- Model was trained on 94,477 sequences from 21 participants
- The model selects 118 landmarks internally: 41 lip + 21+21 hands + 4 nose + 16+16 eyes
- NaN landmarks (undetected body parts) are handled by the model's preprocessing
- No PyTorch dependency — runs on TFLite only

250-sign Vocabulary
TV, after, airplane, all, alligator, animal, another, any, apple, arm,
aunt, awake, backyard, bad, balloon, bath, because, bed, bedroom, bee,
before, beside, better, bird, black, blow, blue, boat, book, boy,
brother, brown, bug, bye, callonphone, can, car, carrot, cat, cereal,
chair, cheek, child, chin, chocolate, clean, close, closet, cloud, clown,
cow, cowboy, cry, cut, cute, dad, dance, dirty, dog, doll, donkey, down,
drawer, drink, drop, dry, dryer, duck, ear, elephant, empty, every, eye,
face, fall, farm, fast, feet, find, fine, finger, finish, fireman, first,
fish, flag, flower, food, for, frenchfries, frog, garbage, gift, giraffe,
girl, give, glasswindow, go, goose, grandma, grandpa, grass, green, gum,
hair, happy, hat, hate, have, haveto, head, hear, helicopter, hello, hen,
hesheit, hide, high, home, horse, hot, hungry, icecream, if, into, jacket,
jeans, jump, kiss, kitty, lamp, later, like, lion, lips, listen, look,
loud, mad, make, man, many, milk, minemy, mitten, mom, moon, morning,
mouse, mouth, nap, napkin, night, no, noisy, nose, not, now, nuts, old,
on, open, orange, outside, owie, owl, pajamas, pen, pencil, penny, person,
pig, pizza, please, police, pool, potty, pretend, pretty, puppy, puzzle,
quiet, radio, rain, read, red, refrigerator, ride, room, sad, same, say,
scissors, see, shhh, shirt, shoe, shower, sick, sleep, sleepy, smile,
snack, snow, stairs, stay, sticky, store, story, stuck, sun, table, talk,
taste, thankyou, that, there, think, thirsty, tiger, time, tomorrow,
tongue, tooth, toothbrush, touch, toy, tree, uncle, underwear, up, vacuum,
wait, wake, water, wet, weus, where, white, who, why, will, wolf, yellow,
yes, yesterday, yourself, yucky, zebra, zipper
