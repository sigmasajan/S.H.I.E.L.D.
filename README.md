# S.H.I.E.L.D.
Speech Hearing Intelligibility Enhancement in Loud Defences

AI/ML-enabled adaptive noise cancellation for defence communications — SIH26052 (DRDO).
Suppresses stationary (engine/rotor), impulsive (blast/gunfire), and mixed battlefield
noise in real time while preserving speech intelligibility, using a hybrid
classical-filter + deep-learning pipeline.

## Quick start

**Requires Python 3.12** (not 3.13+, not 3.14) — `pyproject.toml` pins this because
newer Python versions break the pinned scipy/torchaudio combo below. See "Known
gotchas."

```bash
git clone https://github.com/sigmasajan/S.H.I.E.L.D.
cd S.H.I.E.L.D.
python3.12 -m venv venv && source venv/bin/activate   # use 3.12 explicitly here
pip install -r requirements.txt
streamlit run app.py
```

Sample audio is already committed under `data/`, so this runs immediately — no
dataset download needed. Regenerate or extend it any time with:
```bash
python3 generate_sample_data.py
```
(needs `espeak-ng` for real synthesized speech — `sudo apt install espeak-ng` /
`brew install espeak`; falls back to a synthetic tone automatically if it's missing)

Default dashboard password is `shield2026` — it's hardcoded in `app.py`. **Change
it before this ever goes anywhere public.**

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
steady-hum frames to the fast classical filter (`nlms_filter.py`); everything else —
and everything by default — goes through the AI model (`enhance.py`). `mix_generator.py`
and `metrics.py` are used to build test data and to honestly score results (real
PESQ/STOI/SNR, not claimed numbers). `generate_sample_data.py` exists purely so the
repo is runnable out of the box without any external dataset download.

## Known gotchas (full detail in `requirements.txt` comments)

- **Python must be 3.12.x.** On 3.14, `scipy==1.11.4` has no prebuilt wheel and fails
  to build from source (`Compiler cython cannot compile programs`), and even if you
  get past that, newer `torchaudio` removes `AudioMetaData`, which DeepFilterNet 0.5.6
  imports directly — causing `ImportError: cannot import name 'AudioMetaData'`.
  `pyproject.toml`'s `requires-python` pin exists specifically to prevent this; make
  sure your venv is actually built with 3.12, since the pin alone doesn't enforce
  itself under a plain `pip install`.
- `numpy` is pinned `<2.0` on purpose — DeepFilterNet requires it; this also pins
  `scipy` and `librosa` to older compatible versions.
- `from df.enhance import ...` needs `torch`/`torchaudio` at runtime even though
  DeepFilterNet doesn't list them as hard pip dependencies.
- `sounddevice` needs a system library on Linux: `sudo apt install libportaudio2`.
- `webrtcvad` needs `setuptools<81` (newer setuptools dropped `pkg_resources`).
- Fine-tuning (`DeepFilterNet[train]`) needs the system HDF5 library installed
  *before* you uncomment those lines in `requirements.txt`, or the install fails
  with a Rust/cargo build error.
