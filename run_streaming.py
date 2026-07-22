"""
Standalone LSL Streaming Script.

Run this to simulate an EEG hardware stream:
    python run_streaming.py

Then in another terminal, your BCI framework can listen to the stream.
"""
import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig
from synthetic import generate_synthetic_eeg
from data_pipeline import mock_lsl_stream


def main():
    parser = argparse.ArgumentParser(description="Mock EEG LSL Streamer")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--sfreq", type=int, default=250)
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Duration in seconds")
    parser.add_argument("--chunk", type=int, default=250,
                        help="Samples per chunk")
    parser.add_argument("--noise-snr", type=float, default=-5.0,
                        help="Add noise at this SNR (dB)")

    args = parser.parse_args()

    n_samples = int(args.duration * args.sfreq)
    print(f"[Streamer] Generating {args.duration}s of {args.channels}ch EEG @ {args.sfreq}Hz")

    data = generate_synthetic_eeg(
        n_channels=args.channels, n_samples=n_samples, sfreq=args.sfreq
    )

    if args.noise_snr is not None and args.noise_snr < 30:
        from synthetic import inject_noise
        data = inject_noise(data, noise_type="all", snr_db=args.noise_snr, sfreq=args.sfreq)
        print(f"[Streamer] Injected noise at SNR = {args.noise_snr} dB")

    mock_lsl_stream(data, args.sfreq, "SimulatedEEG", args.chunk)


if __name__ == "__main__":
    main()
