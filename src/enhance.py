from pathlib import Path
import torch
from df.enhance import enhance, init_df, load_audio, save_audio


class AudioEnhancer:
    """Production-grade wrapper for DeepFilterNet audio enhancement."""

    def __init__(self, atten_lim_db: float | None = None, log_level: str = "error"):
        """Initialize and cache the model in memory.
        
        Args:
            atten_lim_db: Max noise attenuation in dB (e.g., 20.0 prevents 
                          unnatural/robotic artifacts; None for max reduction).
            log_level: DeepFilterNet log verbosity ('info', 'warning', 'error').
        """
        self.model, self.df_state, _ = init_df(log_level=log_level)
        self.atten_lim_db = atten_lim_db
        self.sample_rate = self.df_state.sr()

    def process_file(self, input_path: str | Path, output_path: str | Path) -> torch.Tensor:
        """Cleans a single audio file and writes the output.
        
        Args:
            input_path: Path to source audio file (.wav, .flac, .mp3).
            output_path: Target path for the enhanced audio file.
        """
        in_path, out_path = Path(input_path), Path(output_path)
        
        if not in_path.is_file():
            raise FileNotFoundError(f"Source file standard path not found: {in_path}")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Resamples automatically to model target sample rate
        audio, _ = load_audio(in_path, sr=self.sample_rate)

        # Run noise removal model
        enhanced = enhance(
            self.model, 
            self.df_state, 
            audio, 
            atten_lim_db=self.atten_lim_db
        )

        save_audio(out_path, enhanced, self.sample_rate)
        return enhanced

    def process_directory(
        self, 
        input_dir: str | Path, 
        output_dir: str | Path, 
        extensions: tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a")
    ) -> list[Path]:
        """Recursively cleans all audio files in a folder, preserving directory tree structure."""
        in_dir, out_dir = Path(input_dir), Path(output_dir)
        processed_files: list[Path] = []

        for file_path in in_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                relative_path = file_path.relative_to(in_dir)
                target_path = (out_dir / relative_path).with_suffix(".wav")

                try:
                    self.process_file(file_path, target_path)
                    processed_files.append(target_path)
                except Exception as err:
                    print(f"Skipping {file_path.name}: {err}")

        return processed_files


# Example Usage:
if __name__ == "__main__":
    # Preserve natural speech timbre by setting maximum noise attenuation to 20 dB
    cleaner = AudioEnhancer(atten_lim_db=20.0)

    # 1. Clean single file
    cleaner.process_file("input_noisy.wav", "output_clean.wav")

    # 2. Batch process folder
    cleaner.process_directory("./raw_audio", "./cleaned_audio")
