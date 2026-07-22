import numpy as np
import mne
from moabb.datasets import PhysionetMI


def fetch_eeg_dataset(subjects=None, tmin=0.0, tmax=4.0,
                      n_classes=2):
    """Fetch PhysioNet Motor Imagery dataset via MOABB.

    Returns:
        X: ndarray (n_trials, n_channels, n_samples)
        y: ndarray (n_trials,) - labels
        meta: dict with sfreq
    """
    if subjects is None:
        subjects = [1]

    dataset = PhysionetMI()
    raw_data = dataset.get_data(subjects=subjects)

    all_epochs = []
    all_labels = []

    for subj_id in raw_data:
        for sess_key in raw_data[subj_id]:
            for run_key in raw_data[subj_id][sess_key]:
                raw = raw_data[subj_id][sess_key][run_key]
                sfreq = raw.info["sfreq"]

                raw.pick(["eeg", "stim"])
                raw.filter(l_freq=0.5, h_freq=50.0, fir_design="firwin")

                event_times, event_id_map = mne.events_from_annotations(raw)

                if len(event_id_map) < 2:
                    continue

                name_to_int = {}
                for name, integer_id in event_id_map.items():
                    if "rest" not in name:
                        name_to_int[name] = integer_id

                if len(name_to_int) < 2:
                    name_to_int = {}
                    for name, integer_id in event_id_map.items():
                        name_to_int[name] = integer_id

                label_names = list(name_to_int.keys())[:n_classes]
                label_ints = {name_to_int[n]: i for i, n in enumerate(label_names)}

                for ev_row in event_times:
                    ev_integer = ev_row[2]
                    if ev_integer not in label_ints:
                        continue
                    sample_idx = ev_row[0]
                    start = int(sample_idx + tmin * sfreq)
                    end = int(sample_idx + tmax * sfreq)
                    if start < 0 or end > raw.get_data().shape[1]:
                        continue
                    segment = raw.get_data()[:, start:end]
                    if segment.shape[1] > 0:
                        all_epochs.append(segment)
                        all_labels.append(label_ints[ev_integer])

    X = np.array(all_epochs)
    y = np.array(all_labels)

    print(f"  Dataset: {X.shape[0]} trials | {X.shape[1]} channels | {X.shape[2]} samples | sfreq={sfreq}")

    return X, y, {"sfreq": sfreq, "n_channels": X.shape[1]}
