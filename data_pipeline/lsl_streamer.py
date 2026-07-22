import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet


def mock_lsl_stream(data, sfreq, stream_name="SimulatedEEG",
                    chunk_size=250, sleep_per_chunk=None):
    """Stream EEG data over LSL to simulate hardware.

    Args:
        data: ndarray (n_channels, n_samples)
        sfreq: sampling frequency
        stream_name: name of the LSL stream
        chunk_size: samples per push
        sleep_per_chunk: seconds to sleep between chunks (defaults to 1/sfreq)
    """
    if sleep_per_chunk is None:
        sleep_per_chunk = 1.0 / sfreq

    n_channels = data.shape[0]
    info = StreamInfo(stream_name, "EEG", n_channels, sfreq, "float32", "uid_eeg_sim")
    outlet = StreamOutlet(info)

    print(f"[LSL] Streaming '{stream_name}' | {n_channels}ch | {sfreq}Hz | chunk={chunk_size}")

    for start in range(0, data.shape[1], chunk_size):
        end = min(start + chunk_size, data.shape[1])
        chunk = data[:, start:end].astype(np.float32)

        for s in range(chunk.shape[1]):
            outlet.push_sample(chunk[:, s].tolist())
            time.sleep(sleep_per_chunk)

        pct = min(100.0, end / data.shape[1] * 100)
        print(f"\r[LSL] Progress: {pct:.1f}%", end="", flush=True)

    print("\n[LSL] Streaming complete.")
