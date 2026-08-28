from pathlib import Path
from typing import Union, List, Optional
import logging
import torch
from df.enhance import enhance, init_df, load_audio, save_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AudioEnhancer:
    """Production-ready wrapper for DeepFilterNet audio enhancement."""
    
    def __init__(self, post_filter: bool = True, log_level: str = "error"):
        """
        Initializes the model once into memory.
        
        :param post_filter: Enables DeepFilterNet post-filter for additional noise suppression.
        :param log_level: Controls DeepFilterNet internal logging verbosity.
        """
        logging.info("Initializing DeepFilterNet model...")
        self.model, self.df_state, _ = init_df(post_filter=post_filter, log_level=log_level)
        self.sr = self.df_state.sr()
        logging.info(f"Model loaded successfully. Sample rate: {self.sr} Hz")

    def clean_file(
        self, 
        input_path: Union[str, Path], 
        output_path: Union[str, Path], 
        atten_lim_db: Optional[float] = None
    ) -> torch.Tensor:
        """
        Cleans a single audio file and saves the output.
        
        :param input_path: Path to noisy audio file.
        :param output_path: Path where enhanced audio will be saved.
        :param atten_lim_db: Maximum attenuation limit in dB (e.g., 10 to 20 to preserve natural ambient sound).
        :return: Enhanced audio tensor.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Ensure destination directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Processing: {input_path.name}")
        
        # Load audio resampled to model's expected sample rate
        audio, _ = load_audio(input_path, sr=self.sr)

        # Enhance audio
        kwargs = {}
        if atten_lim_db is not None:
            kwargs["atten_lim_db"] = atten_lim_db

        enhanced = enhance(self.model, self.df_state, audio, **kwargs)

        # Save cleaned audio
        save_audio(output_path, enhanced, self.sr)
        logging.info(f"Saved enhanced file to: {output_path}")
        
        return enhanced

    def clean_batch(
        self, 
        input_dir: Union[str, Path], 
        output_dir: Union[str, Path], 
        extensions: List[str] = [".wav", ".flac", ".mp3"],
        atten_lim_db: Optional[float] = None
    ) -> List[Path]:
        """
        Processes all matching audio files within a directory.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        processed_files = []

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        audio_files = [f for f in input_dir.rglob("*") if f.suffix.lower() in extensions]
        logging.info(f"Found {len(audio_files)} audio file(s) in {input_dir}")

        for file in audio_files:
            relative_path = file.relative_to(input_dir)
            out_file = output_dir / relative_path
            
            try:
                self.clean_file(file, out_file, atten_lim_db=atten_lim_db)
                processed_files.append(out_file)
            except Exception as e:
                logging.error(f"Failed to process {file}: {e}")

        return processed_files

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean audio files using DeepFilterNet")
    parser.add_argument("-i", "--input", required=True, help="Input file or folder path")
    parser.add_argument("-o", "--output", required=True, help="Output file or folder path")
    parser.add_argument("--atten-lim", type=float, default=None, help="Max noise attenuation limit in dB (e.g. 15)")
    
    args = parser.parse_args()
    
    enhancer = AudioEnhancer()
    
    inp = Path(args.input)
    if inp.is_dir():
        enhancer.clean_batch(inp, args.output, atten_lim_db=args.atten_lim)
    else:
        enhancer.clean_file(inp, args.output, atten_lim_db=args.atten_lim)
