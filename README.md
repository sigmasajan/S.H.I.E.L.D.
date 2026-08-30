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
├── app.py                    # the live dashboard — Software Eng.
├── requirements.txt          # single pinned dependency list, verified installable
├── README.md
├── src/
│   ├── enhance.py            # DeepFilterNet2 wrapper (the AI model)      
│   ├── mix_generator.py      # builds noisy-clean pairs at controlled SNR 
│   ├── metrics.py            # PESQ / STOI / SNR scoring                  
│   ├── nlms_filter.py        # classical adaptive filter, steady-hum     
│   └── regime_detector.py    # classifies each frame's noise type        
└── data/
    ├── clean/                # clean speech clips (LibriSpeech/VCTK subset)
    ├── noise/                # noise clips (UrbanSound8K gun_shot/engine_idling, DEMAND)
    └── mixed/                # generated noisy-clean pairs (git-ignore this, it's big)
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
