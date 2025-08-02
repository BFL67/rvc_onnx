#!/usr/bin/env python3
"""
Command-line interface for RVC ONNX voice conversion.
"""

import argparse
import sys
from rvc_onnx.infer import run_convert_script


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RVC ONNX Voice Conversion")
    
    # Required arguments
    parser.add_argument("--input", "-i", required=True, help="Input audio file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output audio file")
    parser.add_argument("--model", "-m", required=True, help="Path to RVC model (.pth or .onnx)")
    
    # Optional arguments
    parser.add_argument("--pitch", "-p", type=int, default=0, help="Pitch shift in semitones (default: 0)")
    parser.add_argument("--filter-radius", type=int, default=3, help="Filter radius (default: 3)")
    parser.add_argument("--index-rate", type=float, default=0.5, help="Index rate (default: 0.5)")
    parser.add_argument("--volume-envelope", type=float, default=0.25, help="Volume envelope (default: 0.25)")
    parser.add_argument("--protect", type=float, default=0.33, help="Protect voiceless consonants (default: 0.33)")
    parser.add_argument("--hop-length", type=int, default=128, help="Hop length (default: 128)")
    parser.add_argument("--f0-method", default="rmvpe", help="F0 extraction method (default: rmvpe)")
    parser.add_argument("--index-path", default="", help="Path to index file")
    parser.add_argument("--f0-autotune", action="store_true", help="Enable F0 autotuning")
    parser.add_argument("--f0-autotune-strength", type=float, default=0.8, help="F0 autotune strength (default: 0.8)")
    parser.add_argument("--clean-audio", action="store_true", help="Clean audio with high-pass filter")
    parser.add_argument("--clean-strength", type=float, default=0.7, help="Clean strength (default: 0.7)")
    parser.add_argument("--export-format", default="wav", help="Export format (default: wav)")
    parser.add_argument("--embedder-model", default="hubert_base", help="Embedder model (default: hubert_base)")
    parser.add_argument("--resample-sr", type=int, default=0, help="Resample sample rate (default: 0 - no resampling)")
    parser.add_argument("--split-audio", action="store_true", help="Split audio into chunks for processing")
    
    args = parser.parse_args()
    
    try:
        run_convert_script(
            pitch=args.pitch,
            filter_radius=args.filter_radius,
            index_rate=args.index_rate,
            volume_envelope=args.volume_envelope,
            protect=args.protect,
            hop_length=args.hop_length,
            f0_method=args.f0_method,
            input_path=args.input,
            output_path=args.output,
            pth_path=args.model,
            index_path=args.index_path,
            f0_autotune=args.f0_autotune,
            f0_autotune_strength=args.f0_autotune_strength,
            clean_audio=args.clean_audio,
            clean_strength=args.clean_strength,
            export_format=args.export_format,
            embedder_model=args.embedder_model,
            resample_sr=args.resample_sr,
            split_audio=args.split_audio
        )
        print("Voice conversion completed successfully!")
    except Exception as e:
        print(f"Error during voice conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

