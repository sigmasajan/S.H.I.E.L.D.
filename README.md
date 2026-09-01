# S.H.I.E.L.D.
Speech Hearing Intelligibility Enhancement in Loud Defences

AI/ML-enabled adaptive noise cancellation for defence communications — SIH26052 (DRDO).
Suppresses stationary (engine/rotor), impulsive (blast/gunfire), and mixed battlefield
noise in real time while preserving speech intelligibility, using a hybrid
classical-filter + deep-learning pipeline.

## Quick start

```bash
git clone https://github.com/sigmasajan/S.H.I.E.L.D.
cd S.H.I.E.L.D.
pip install -r requirements.txt
streamlit run app.py
```

Default dashboard password is `shield2026` — it's in `app.py`

## Repo structure

```
S.H.I.E.L.D./
├── .gitignore
├── .streamlit/
│   └── config.toml        # headless=true, so `streamlit run` doesn't prompt on shared machines
├── LICENSE
├── README.md
├── app.py                 # the live dashboard
├── generate_sample_data.py    # sample-data generator — lives at repo root, not scripts/
├── pyproject.toml         # pins Python to >=3.12,<3.13 — prevents the 3.14 build/import crisis
├── requirements.txt
├── data/
│   ├── clean/              # 4 sample speech clips
│   │   ├── speech_00.wav
│   │   ├── speech_01.wav
│   │   ├── speech_02.wav
│   │   └── speech_03.wav
│   ├── noise/               # 3 synthesized noise types
│   │   ├── engine_hum.wav
│   │   ├── gunfire_burst.wav
│   │   └── rotor_wash.wav
│   └── mixed/                # 12 generated noisy-clean pairs (committed, not gitignored)
│       ├── speech_00__engine_hum__3dB.wav
│       ├── speech_00__gunfire_burst__3dB.wav
│       ├── ... (12 total: 4 speech × 3 noise types, all at 3dB SNR)
└── src/
    ├── __init__.py
    ├── enhance.py          # DeepFilterNet2 wrapper
    ├── mix_generator.py    # noisy-clean pair generator
    ├── nlms_filter.py      # classical adaptive filter
    ├── regime_detector.py  # per-frame noise classifier
    └── metrics.py          # PESQ / STOI / SNR scoring
```

## How the pieces fit together

`app.py` classifies incoming audio frame-by-frame (`regime_detector.py`) and routes
steady-hum frames to the fast classical filter (`nlms_filter.py`) and everything else
to the AI model (`enhance.py`). `mix_generator.py` and `metrics.py` are used offline
to build training/test data and to honestly score results (real PESQ/STOI/SNR, not
claimed numbers).

## Known gotchas (full detail in `requirements.txt` comments)

- `numpy` is pinned `<2.0` on purpose — DeepFilterNet requires it; this also pins
  `scipy` and `librosa` to older compatible versions.
- `from df.enhance import ...` needs `torch`/`torchaudio` at runtime even though
  DeepFilterNet doesn't list them as hard pip dependencies.
- `sounddevice` needs a system library on Linux: `sudo apt install libportaudio2`.
- `webrtcvad` needs `setuptools<81` (newer setuptools dropped `pkg_resources`).
- Fine-tuning (`DeepFilterNet[train]`) needs the system HDF5 library installed
  *before* you uncomment those lines in `requirements.txt`, or the install fails
  with a Rust/cargo build error.
