from pathlib import Path
import gc

import numpy as np
import mne

import moabb.datasets.download as _moabb_download
from moabb.datasets import (PhysionetMI, BNCI2014_001, Lee2019_MI, Weibo2014,
                            BNCI2014_002, BNCI2014_004, Ofner2017, Zhou2016,
                            Cho2017, Shin2017A, AlexMI, GrosseWentrup2009,
                            Sosulski2019)


def _sanitize_path_win(path):
    # MOABB 1.5.x _sanitize_path replaces every ':' with '-', which mangles
    # Windows drive letters (C:\ -> C-\), turning absolute paths relative and
    # making downloads land in "<cwd>\C-\Users\...". Keep the drive colon so
    # datasets are stored in the real MNE_DATA folder.
    s = str(path)
    if len(s) >= 2 and s[1] == ":":
        rest = s[2:].translate({ord(c): "-" for c in ':*?"<>|'})
        return Path(s[:2] + rest)
    return Path(s.translate({ord(c): "-" for c in ':*?"<>|'}))


_moabb_download._sanitize_path = _sanitize_path_win

# Registry of supported real-world motor-imagery datasets.
# Each maps to a MOABB loader; get_data(subjects=[...]) handles downloading
# and caching automatically.
DATASETS = {
    # --- Original four ---
    "physionet": PhysionetMI,   # PhysioNet EEG Motor Imagery (109 subjects, 64ch)
    "bci2a": BNCI2014_001,      # BCI Competition IV 2a (9 subjects, 22ch, 250 Hz)
    "openbmi": Lee2019_MI,      # OpenBMI Lee2019 MI (54 subjects, binary MI)
    "weibo2014": Weibo2014,     # Weibo2014 MI (10 subjects)
    # --- Extra sources (auto-download on first use) ---
    "bci2b": BNCI2014_002,      # BCI Competition IV 2b (9 subjects, 3ch, 250 Hz)
    "bci4": BNCI2014_004,       # BCI Competition IV 4 (9 subjects, motor imagery)
    "ofner2017": Ofner2017,     # Ofner et al. 2017 (15 subjects, 61ch, 3-class MI)
    "zhou2016": Zhou2016,       # Zhou et al. 2016 (4 subjects, 30ch)
    "cho2017": Cho2017,         # Cho et al. 2017 (52 subjects, 64ch, 512 Hz)
    "shin2017": Shin2017A,      # Shin et al. 2017 A (29 subjects, 62ch)
    "alexmi": AlexMI,           # AlexMotorImagery (40 subjects, 64ch, 512 Hz)
    "grossewentrup2009": GrosseWentrup2009,  # Grosse-Wentrup 2009 (10 subjects)
    "sosulski2019": Sosulski2019,            # Sosulski et al. 2019 (5 subjects, 256ch)
}


def fetch_eeg_dataset(dataset_name="physionet", subjects=None, tmin=0.0, tmax=4.0,
                      n_classes=2):
    """Fetch a real motor-imagery EEG dataset via MOABB.

    Args:
        dataset_name: key into DATASETS registry
        subjects: list of subject IDs (None -> first subject)
        tmin/tmax: epoch window in seconds relative to each cue
        n_classes: number of classes to keep (prefers left/right hand)

    Returns:
        X: ndarray (n_trials, n_channels, n_samples)
        y: ndarray (n_trials,) - labels in 0..n_classes-1
        meta: dict with sfreq, n_channels, dataset name and per-trial subject
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. "
                         f"Choose from: {list(DATASETS)}")

    if subjects is None:
        subjects = [1]

    ds = DATASETS[dataset_name]()
    print(f"  Dataset: {dataset_name} ({ds.code}) | subjects: {subjects}")
    raw_data = ds.get_data(subjects=subjects)

    all_epochs = []
    all_labels = []
    all_subjects = []
    sfreq = None

    # Iterate and drop each Raw as soon as it is processed so large datasets
    # (e.g. OpenBMI Lee2019: multi-GB, long recordings) never hold all runs in
    # RAM at once. raw.get_data() is also called ONCE per run instead of once
    # per trial to avoid repeatedly allocating full-recording copies.
    for subj_id in list(raw_data.keys()):
        for sess_key in list(raw_data[subj_id].keys()):
            for run_key in list(raw_data[subj_id][sess_key].keys()):
                raw = raw_data[subj_id][sess_key][run_key]
                try:
                    sfreq = raw.info["sfreq"]

                    # Keep only EEG channels for the signal; use stim for events.
                    # Including STIM/EOG/ECG channels in X would leak trigger
                    # values / artifacts into denoising and classification.
                    eeg_idx = mne.pick_types(raw.info, meg=False, eeg=True, stim=False)
                    if len(eeg_idx) == 0:
                        continue
                    raw.pick(eeg_idx)
                    raw.filter(l_freq=0.5, h_freq=50.0, fir_design="firwin")

                    event_times, event_id_map = mne.events_from_annotations(raw)
                    if len(event_id_map) < 2:
                        continue

                    # Drop rest/relax classes, prefer left/right hand for a stable
                    # binary motor-imagery mapping.
                    name_to_int = {str(name): i for name, i in event_id_map.items()
                                   if not isinstance(name, (int, np.integer))
                                   and "rest" not in str(name).lower()}
                    if n_classes == 2 and {"left_hand", "right_hand"}.issubset(name_to_int):
                        label_names = ["left_hand", "right_hand"]
                    else:
                        label_names = sorted(name_to_int.keys())[:n_classes]
                    label_ints = {name_to_int[n]: i for i, n in enumerate(label_names)}

                    data = raw.get_data()
                    n_samples = data.shape[1]
                    for ev_row in event_times:
                        ev_integer = ev_row[2]
                        if ev_integer not in label_ints:
                            continue
                        sample_idx = ev_row[0]
                        start = int(sample_idx + tmin * sfreq)
                        end = int(sample_idx + tmax * sfreq)
                        if start < 0 or end > n_samples:
                            continue
                        segment = data[:, start:end]
                        if segment.shape[1] > 0:
                            all_epochs.append(segment)
                            all_labels.append(label_ints[ev_integer])
                            all_subjects.append(subj_id)
                finally:
                    # Free this Raw (and the local `data` copy) immediately so
                    # we never hold every recording in memory simultaneously.
                    raw_data[subj_id][sess_key][run_key] = None
                    del raw
                    if "data" in locals():
                        del data
                    gc.collect()

    if len(all_epochs) == 0:
        raise RuntimeError(f"No trials could be extracted for dataset "
                           f"'{dataset_name}' (subjects {subjects}).")

    X = np.array(all_epochs)
    y = np.array(all_labels)

    print(f"  Loaded: {X.shape[0]} trials | {X.shape[1]} channels | "
          f"{X.shape[2]} samples | sfreq={sfreq} | classes={np.unique(y)}")

    return X, y, {
        "sfreq": sfreq,
        "n_channels": X.shape[1],
        "dataset_name": dataset_name,
        "subjects": np.array(all_subjects),
    }
