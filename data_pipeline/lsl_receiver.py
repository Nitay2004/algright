import numpy as np
from collections import deque
from pylsl import StreamInlet, resolve_byprop


class LSLReceiver:
    """Receives EEG data from an LSL stream and buffers into epochs."""

    def __init__(self, stream_name="SimulatedEEG", epoch_samples=250, buffer_epochs=10):
        self.stream_name = stream_name
        self.epoch_samples = epoch_samples
        self.buffer = deque(maxlen=epoch_samples * buffer_epochs)
        self.inlet = None

    def connect(self, timeout=10.0):
        """Resolve and connect to the LSL stream."""
        print(f"[LSL-Recv] Searching for stream '{self.stream_name}'...")
        streams = resolve_byprop('type', 'EEG', timeout=timeout)
        if not streams:
            raise ConnectionError(f"No LSL stream found with name '{self.stream_name}'")

        self.inlet = StreamInlet(streams[0])
        print(f"[LSL-Recv] Connected to '{self.stream_name}'")

    def receive_chunk(self):
        """Receive a single sample or chunk and buffer it."""
        if self.inlet is None:
            raise RuntimeError("Not connected. Call connect() first.")

        chunk, _ = self.inlet.pull_sample(timeout=0.0)
        if chunk is not None:
            self.buffer.append(np.array(chunk))
        return chunk is not None

    def get_epoch(self):
        """Return a complete epoch (n_channels, epoch_samples) if available."""
        if len(self.buffer) >= self.epoch_samples:
            samples = []
            for _ in range(self.epoch_samples):
                samples.append(self.buffer.popleft())
            return np.array(samples).T  # (n_channels, epoch_samples)
        return None

    def receive_epoch(self, timeout=5.0):
        """Block until a full epoch is buffered."""
        import time
        start = time.time()
        while time.time() - start < timeout:
            self.receive_chunk()
            epoch = self.get_epoch()
            if epoch is not None:
                return epoch
        return None
