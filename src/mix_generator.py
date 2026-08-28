"""
S.H.I.E.L.D. — Mix Generator
Synthesises noisy demo/training clips by mixing clean speech with noise at a
target SNR.  Outputs land in ``data/mixed/`` exactly where app.py expects its
bundled presets (see app.py line 81).

Noise sources
─────────────
 • Real recordings  — pass ``--noise-dir`` pointing at a folder of .wav files.
 • Built-in synthetic — when no real recordings are available, the generator
   fabricates plausible noise envelopes for each of the four mission-mode
   scenarios so the Streamlit dashboard has something to play immediately.

Usage examples
──────────────
  # Generate all four presets with synthetic noise at −5 dB SNR:
  python -m src.mix_generator --speech data/clean/speech.wav --snr -5

  # Use a real noise file for one scenario:
  python -m src.mix_generator --speech data/clean/speech.wav \
         --noise data/noise/helo.wav --scenario "Helicopter Rotor" --snr 0

  # Batch: mix every speech file with every noise file at several SNRs:
  python -m src.mix_generator --speech-dir data/clean/ \
         --noise-dir data/noise/ --snr -10 -5 0 5

  # Dry-run to see what would be generated:
  python -m src.mix_generator --speech data/clean/speech.wav --dry-run

Owner: CS Core 1 — see main playbook §5.1.4.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy imports — degrade gracefully so the module can at least be
# imported (e.g. for testing) even if librosa/soundfile aren't installed yet.
# ---------------------------------------------------------------------------
try:
    import librosa
    import soundfile as sf
except ImportError as _exc:  # pragma: no cover
    _IMPORT_ERROR = _exc

    class _Stub:
        """Raises at call-time so the error message is clear."""
        def __getattr__(self, name):
            raise ImportError(
                "mix_generator requires librosa and soundfile.  "
                "Install them:  pip install librosa soundfile"
            ) from _IMPORT_ERROR

    librosa = _Stub()  # type: ignore[assignment]
    sf = _Stub()        # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCENARIOS: list[str] = [
    "Vehicle_Engine",
    "Gunfire_Burst",
    "Helicopter_Rotor",
    "Mixed_-_Unknown",
]

DEFAULT_SR: int = 16_000          # Streamlit app re-samples anyway; 16 kHz is
                                  # the standard for speech enhancement models.
DEFAULT_SNR_DB: float = -5.0      # Harsh default — tests the pipeline on hard
                                  # cases first (main playbook §3.2).
DEFAULT_DURATION_SEC: float = 10.0

OUTPUT_DIR: Path = Path("data/mixed")


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic noise generators — one per scenario
# ═══════════════════════════════════════════════════════════════════════════
def _synth_vehicle_engine(n_samples: int, sr: int) -> np.ndarray:
    """Low-frequency rumble + harmonic hum characteristic of diesel engines.

    Models a ~30 Hz fundamental with 3 harmonics, amplitude-modulated at ~2 Hz
    (simulating RPM flutter), plus broadband low-frequency noise shaped by a
    1st-order IIR lowpass.
    """
    t = np.arange(n_samples) / sr
    fundamental = 30.0  # Hz — typical idle diesel

    # Harmonic stack
    hum = np.zeros(n_samples)
    for k in range(1, 5):
        hum += (0.6 ** k) * np.sin(2 * np.pi * fundamental * k * t)

    # Amplitude modulation — RPM flutter
    mod = 1.0 + 0.3 * np.sin(2 * np.pi * 2.0 * t)
    hum *= mod

    # Broadband rumble — simple 1st-order IIR lowpass on white noise
    wn = np.random.randn(n_samples)
    alpha = 0.98  # heavier smoothing → more bass
    rumble = np.empty(n_samples)
    rumble[0] = wn[0]
    for i in range(1, n_samples):
        rumble[i] = alpha * rumble[i - 1] + (1 - alpha) * wn[i]

    out = 0.7 * hum + 0.3 * rumble
    return out / (np.max(np.abs(out)) + 1e-12)


def _synth_gunfire_burst(n_samples: int, sr: int) -> np.ndarray:
    """Impulsive bursts with exponential decays — simulates a short automatic
    burst (3–5 rounds) with supersonic crack transients.

    Each shot is a Gaussian-windowed broadband click followed by a 60 ms
    exponential decay.  Inter-shot interval ~120 ms (≈500 RPM).
    """
    out = np.zeros(n_samples)
    shot_interval = int(0.12 * sr)    # ~120 ms between rounds
    decay_len = int(0.06 * sr)        # 60 ms decay tail
    n_shots = max(3, min(8, n_samples // shot_interval))

    # Start burst 10 %–30 % into the clip so there's a quiet lead-in
    burst_start = int(n_samples * np.random.uniform(0.10, 0.30))

    for i in range(n_shots):
        pos = burst_start + i * shot_interval
        if pos + decay_len >= n_samples:
            break
        # Transient crack — broadband
        click_len = int(0.002 * sr)  # 2 ms
        click = np.random.randn(click_len) * np.hanning(click_len)
        end_click = min(pos + click_len, n_samples)
        out[pos:end_click] += click[: end_click - pos] * 1.5

        # Exponential decay tail
        tail = np.random.randn(decay_len) * np.exp(-np.linspace(0, 6, decay_len))
        end_tail = min(pos + decay_len, n_samples)
        out[pos:end_tail] += tail[: end_tail - pos]

    # Soft background hiss
    out += 0.02 * np.random.randn(n_samples)
    return out / (np.max(np.abs(out)) + 1e-12)


def _synth_helicopter_rotor(n_samples: int, sr: int) -> np.ndarray:
    """Periodic blade-pass thumps + broadband turbulence.

    Main rotor blade-pass frequency ~17 Hz (4-blade, 260 RPM).  Each pass
    creates a brief pressure pulse; between passes there is broadband
    turbulent flow noise.
    """
    t = np.arange(n_samples) / sr
    bpf = 17.0  # blade-pass frequency

    # Blade-pass tonal
    tonal = np.zeros(n_samples)
    for k in range(1, 7):
        tonal += (0.5 ** k) * np.sin(2 * np.pi * bpf * k * t + np.random.uniform(0, 2 * np.pi))

    # Shape into sharper pulses via half-wave rectification + squaring
    tonal = np.clip(tonal, 0, None) ** 2

    # Turbulent broadband
    turb = np.random.randn(n_samples)
    # Simple moving-average lowpass (≈400 Hz cutoff at 16 kHz SR)
    kernel_len = max(1, sr // 400)
    kernel = np.ones(kernel_len) / kernel_len
    turb = np.convolve(turb, kernel, mode="same")

    out = 0.6 * tonal + 0.4 * turb
    return out / (np.max(np.abs(out)) + 1e-12)


def _synth_mixed_unknown(n_samples: int, sr: int) -> np.ndarray:
    """Blend of engine hum, occasional impulsive pops, and wind noise —
    simulates a 'we-don't-know-what-this-is' scenario that the regime
    detector should handle adaptively.
    """
    engine = _synth_vehicle_engine(n_samples, sr)
    gunfire = _synth_gunfire_burst(n_samples, sr)
    helo = _synth_helicopter_rotor(n_samples, sr)

    # Random blend weights
    w = np.random.dirichlet([1.0, 0.3, 0.5])
    out = w[0] * engine + w[1] * gunfire + w[2] * helo

    # Add pink-ish wind noise
    wn = np.random.randn(n_samples)
    pink = np.cumsum(wn)
    pink -= np.mean(pink)
    pink /= np.max(np.abs(pink)) + 1e-12
    out += 0.15 * pink

    return out / (np.max(np.abs(out)) + 1e-12)


SYNTH_GENERATORS: dict[str, callable] = {
    "Vehicle_Engine": _synth_vehicle_engine,
    "Gunfire_Burst": _synth_gunfire_burst,
    "Helicopter_Rotor": _synth_helicopter_rotor,
    "Mixed_-_Unknown": _synth_mixed_unknown,
}


# ═══════════════════════════════════════════════════════════════════════════
# Core mixing logic
# ═══════════════════════════════════════════════════════════════════════════
def rms(x: np.ndarray) -> float:
    """Root-mean-square level of *x*."""
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Mix *speech* and *noise* at the given SNR (dB).

    If the arrays differ in length the shorter one is tiled / truncated so
    they match.

    Parameters
    ----------
    speech : 1-D float array — clean speech signal.
    noise  : 1-D float array — noise signal.
    snr_db : target signal-to-noise ratio in decibels.

    Returns
    -------
    mix : 1-D float array — noisy mixture, peak-normalised to [−1, 1].
    """
    target_len = max(len(speech), len(noise))

    # Tile / truncate to equal length
    if len(speech) < target_len:
        reps = int(np.ceil(target_len / len(speech)))
        speech = np.tile(speech, reps)[:target_len]
    else:
        speech = speech[:target_len]

    if len(noise) < target_len:
        reps = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, reps)[:target_len]
    else:
        noise = noise[:target_len]

    # Scale noise to achieve target SNR
    speech_rms = rms(speech)
    noise_rms = rms(noise)
    if noise_rms < 1e-12:
        return speech  # silence → nothing to mix

    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_noise_rms / noise_rms)

    mix = speech + noise_scaled

    # Peak-normalise to avoid clipping
    peak = np.max(np.abs(mix))
    if peak > 1e-12:
        mix /= peak
    return mix


def generate_pair(clean_path: str | Path, noise_path: str | Path, snr_db: float, sr: int = 16000):
    """Load clean speech and noise from disk and mix them at target SNR."""
    clean, _ = librosa.load(str(clean_path), sr=sr, mono=True)
    noise, _ = librosa.load(str(noise_path), sr=sr, mono=True)
    return clean, mix_at_snr(clean, noise, snr_db)


def generate_silence_speech(duration_sec: float, sr: int) -> np.ndarray:
    """Generate a simple synthetic 'speech' signal for testing when no real
    clean-speech file is provided.  A 200 Hz tone pulsed on/off every 0.5 s
    (vaguely simulates voiced/unvoiced alternation).
    """
    n = int(duration_sec * sr)
    t = np.arange(n) / sr
    tone = np.sin(2 * np.pi * 200 * t)

    # On/off gating — 0.5 s windows
    gate_len = int(0.5 * sr)
    gate = np.zeros(n)
    for start in range(0, n, 2 * gate_len):
        end = min(start + gate_len, n)
        gate[start:end] = 1.0
    # Smooth edges to avoid clicks
    ramp = int(0.01 * sr)
    for start in range(0, n, 2 * gate_len):
        for j in range(ramp):
            if start + j < n:
                gate[start + j] *= j / ramp
        end = min(start + gate_len, n)
        for j in range(ramp):
            idx = end - 1 - j
            if 0 <= idx < n:
                gate[idx] *= j / ramp

    return tone * gate


# ═══════════════════════════════════════════════════════════════════════════
# High-level API
# ═══════════════════════════════════════════════════════════════════════════
def generate_scenario_preset(
    scenario: str,
    speech: np.ndarray,
    sr: int,
    snr_db: float = DEFAULT_SNR_DB,
    noise: Optional[np.ndarray] = None,
    output_dir: Path = OUTPUT_DIR,
    dry_run: bool = False,
) -> Path:
    """Create a single scenario preset WAV and write it to *output_dir*.

    Parameters
    ----------
    scenario  : one of the four SCENARIOS names.
    speech    : 1-D clean speech array.
    sr        : sample rate.
    snr_db    : target SNR in dB.
    noise     : optional externally-supplied noise array.  If ``None``, the
                built-in synthetic generator for *scenario* is used.
    output_dir: directory to write into (created if missing).
    dry_run   : if ``True``, skip writing and just return the target path.

    Returns
    -------
    Path to the written (or would-be-written) WAV file.
    """
    # Sanitise filesystem-unsafe characters (the '/' in "Mixed / Unknown")
    safe_name = scenario.replace("/", "-")
    out_path = output_dir / f"{safe_name}.wav"

    if dry_run:
        print(f"[dry-run] would write → {out_path}  (SNR={snr_db:+.1f} dB)")
        return out_path

    if noise is None:
        gen = SYNTH_GENERATORS.get(scenario)
        if gen is None:
            raise ValueError(
                f"Unknown scenario {scenario!r}.  "
                f"Must be one of: {', '.join(SCENARIOS)}"
            )
        noise = gen(len(speech), sr)

    mix = mix_at_snr(speech, noise, snr_db)

    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), mix, sr)
    print(f"✓ {out_path}  ({len(mix)/sr:.1f} s, SNR={snr_db:+.1f} dB)")
    return out_path


def generate_all_presets(
    speech: np.ndarray,
    sr: int,
    snr_db: float = DEFAULT_SNR_DB,
    output_dir: Path = OUTPUT_DIR,
    dry_run: bool = False,
) -> list[Path]:
    """Generate preset mixes for **all four** scenarios."""
    return [
        generate_scenario_preset(
            scenario=s,
            speech=speech,
            sr=sr,
            snr_db=snr_db,
            output_dir=output_dir,
            dry_run=dry_run,
        )
        for s in SCENARIOS
    ]


def batch_mix(
    speech_paths: Sequence[Path],
    noise_paths: Sequence[Path],
    snr_dbs: Sequence[float],
    output_dir: Path = OUTPUT_DIR,
    sr: int = DEFAULT_SR,
    dry_run: bool = False,
) -> list[Path]:
    """Cross-product batch: every speech × every noise × every SNR.

    Output filenames encode all three axes so nothing collides:
        ``{speech_stem}_x_{noise_stem}_{snr}dB.wav``
    """
    written: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for sp in speech_paths:
        speech, _ = librosa.load(str(sp), sr=sr, mono=True)
        for np_ in noise_paths:
            noise, _ = librosa.load(str(np_), sr=sr, mono=True)
            for snr_db in snr_dbs:
                fname = f"{sp.stem}_x_{np_.stem}_{snr_db:+.0f}dB.wav"
                out_path = output_dir / fname
                if dry_run:
                    print(f"[dry-run] would write → {out_path}")
                else:
                    mix = mix_at_snr(speech, noise, snr_db)
                    sf.write(str(out_path), mix, sr)
                    dur = len(mix) / sr
                    print(f"✓ {out_path}  ({dur:.1f} s, SNR={snr_db:+.1f} dB)")
                written.append(out_path)
    return written


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mix_generator",
        description=(
            "S.H.I.E.L.D. mix generator — synthesise noisy demo/training "
            "clips by combining clean speech with noise at a target SNR."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.mix_generator --speech data/clean/speech.wav --snr -5\n"
            "  python -m src.mix_generator --speech data/clean/speech.wav "
            '--noise data/noise/helo.wav --scenario "Helicopter_Rotor" --snr 0\n'
            "  python -m src.mix_generator --speech-dir data/clean/ "
            "--noise-dir data/noise/ --snr -10 -5 0 5\n"
        ),
    )

    # --- Input sources ---
    inp = p.add_argument_group("input sources")
    inp.add_argument(
        "--speech",
        type=Path,
        default=None,
        help="Path to a single clean-speech .wav file.",
    )
    inp.add_argument(
        "--speech-dir",
        type=Path,
        default=None,
        help="Directory of clean-speech .wav files (for batch mode).",
    )
    inp.add_argument(
        "--noise",
        type=Path,
        default=None,
        help="Path to a single noise .wav file (overrides built-in synth).",
    )
    inp.add_argument(
        "--noise-dir",
        type=Path,
        default=None,
        help="Directory of noise .wav files (for batch mode).",
    )
    inp.add_argument(
        "--synth-speech",
        action="store_true",
        help=(
            "Generate a synthetic pulsed tone as a stand-in for clean speech. "
            "Useful for smoke-testing the pipeline without any real audio."
        ),
    )

    # --- Mix parameters ---
    mix = p.add_argument_group("mix parameters")
    mix.add_argument(
        "--snr",
        type=float,
        nargs="+",
        default=[DEFAULT_SNR_DB],
        metavar="DB",
        help=f"Target SNR(s) in dB.  (default: {DEFAULT_SNR_DB})",
    )
    mix.add_argument(
        "--scenario",
        type=str,
        choices=SCENARIOS,
        default=None,
        help=(
            "Generate a preset for one specific scenario only "
            "(default: generate all four)."
        ),
    )
    mix.add_argument(
        "--sr",
        type=int,
        default=DEFAULT_SR,
        help=f"Output sample rate in Hz.  (default: {DEFAULT_SR})",
    )
    mix.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SEC,
        metavar="SEC",
        help=(
            f"Duration for synthetic speech/noise when no real files are "
            f"provided.  (default: {DEFAULT_DURATION_SEC} s)"
        ),
    )

    # --- Output ---
    out = p.add_argument_group("output")
    out.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Where to write the mixed WAVs.  (default: {OUTPUT_DIR})",
    )
    out.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing any files.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry-point.  Returns 0 on success, 1 on user error."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── Resolve speech input ──────────────────────────────────────────────
    # Batch mode: --speech-dir + --noise-dir
    if args.speech_dir and args.noise_dir:
        speech_paths = sorted(args.speech_dir.glob("*.wav"))
        noise_paths = sorted(args.noise_dir.glob("*.wav"))
        if not speech_paths:
            print(f"ERROR: no .wav files found in {args.speech_dir}", file=sys.stderr)
            return 1
        if not noise_paths:
            print(f"ERROR: no .wav files found in {args.noise_dir}", file=sys.stderr)
            return 1
        print(
            f"Batch mode: {len(speech_paths)} speech × "
            f"{len(noise_paths)} noise × {len(args.snr)} SNR(s)"
        )
        batch_mix(
            speech_paths,
            noise_paths,
            args.snr,
            output_dir=args.output_dir,
            sr=args.sr,
            dry_run=args.dry_run,
        )
        return 0

    # Single-file / synthetic mode
    if args.speech:
        if not args.speech.exists():
            print(f"ERROR: speech file not found: {args.speech}", file=sys.stderr)
            return 1
        speech, _ = librosa.load(str(args.speech), sr=args.sr, mono=True)
        print(f"Loaded speech: {args.speech}  ({len(speech)/args.sr:.1f} s)")
    elif args.synth_speech:
        speech = generate_silence_speech(args.duration, args.sr)
        print(f"Using synthetic speech  ({args.duration:.1f} s)")
    else:
        # Auto-generate synthetic speech as a fallback
        speech = generate_silence_speech(args.duration, args.sr)
        print(
            f"No --speech or --synth-speech provided; auto-generating "
            f"synthetic speech  ({args.duration:.1f} s)"
        )

    # ── Resolve noise input ───────────────────────────────────────────────
    noise: Optional[np.ndarray] = None
    if args.noise:
        if not args.noise.exists():
            print(f"ERROR: noise file not found: {args.noise}", file=sys.stderr)
            return 1
        noise, _ = librosa.load(str(args.noise), sr=args.sr, mono=True)
        print(f"Loaded noise: {args.noise}  ({len(noise)/args.sr:.1f} s)")

    # ── Generate ──────────────────────────────────────────────────────────
    snr_db = args.snr[0]  # for preset mode, use the first SNR
    if len(args.snr) > 1:
        print(
            "NOTE: multiple SNRs supplied but not in batch mode — "
            f"using the first one ({snr_db:+.1f} dB)."
        )

    if args.scenario:
        generate_scenario_preset(
            scenario=args.scenario,
            speech=speech,
            sr=args.sr,
            snr_db=snr_db,
            noise=noise,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    else:
        generate_all_presets(
            speech=speech,
            sr=args.sr,
            snr_db=snr_db,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
