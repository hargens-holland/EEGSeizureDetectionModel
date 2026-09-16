"""CHB-MIT EEG loading, windowing and labeling.

Turns multi-hour EDF recordings into fixed-length labeled windows:

    EDF file -> bandpass 0.5-40 Hz -> resample 256 Hz -> 4 s windows (50% overlap)
    -> per-channel normalization -> (23, 1024) tensor + seizure/non-seizure label

Every window also remembers which recording it came from, so splits can be
made at the recording level instead of the window level (see train_utils).
"""

import numpy as np
import mne
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy import signal


class CHBMITDataset(Dataset):
    """Windowed CHB-MIT recordings.

    Expects the standard CHB-MIT layout::

        data_dir/
            chb01/
                chb01_01.edf
                chb01-summary.txt
            chb02/
                ...

    The ``max_*`` arguments cap how much is loaded, which keeps development
    runs tractable — the full dataset is tens of gigabytes.
    """

    def __init__(self, data_dir, window_size=4, overlap=0.5, target_fs=256,
                 max_patients=None, max_files_per_patient=None, max_windows=None):
        """
        Args:
            data_dir: Path to the directory of patient folders.
            window_size: Window length in seconds.
            overlap: Overlap fraction between consecutive windows (0 to 1).
            target_fs: Target sampling frequency in Hz.
            max_patients: Optional cap on number of patient folders to load.
            max_files_per_patient: Optional cap on EDF files per patient.
            max_windows: Optional hard cap on total windows across all patients.
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.overlap = overlap
        self.target_fs = target_fs

        # 4 s * 256 Hz = 1024 samples per window; 50% overlap -> stride of 512.
        self.samples_per_window = int(window_size * target_fs)
        self.stride = int(self.samples_per_window * (1 - overlap))

        self.max_patients = max_patients
        self.max_files_per_patient = max_files_per_patient
        self.max_windows = max_windows

        # ``recording_ids[i]`` is the index into ``recordings`` of the EDF file
        # that window ``i`` was cut from; ``recording_has_seizure`` is per file.
        self.recordings = []
        self.recording_has_seizure = []
        self.windows, self.labels, self.recording_ids = self._load_all_data()

    def _load_edf(self, edf_file, seizure_intervals):
        """Read one EDF recording and cut it into labeled windows.

        Args:
            edf_file: Path to the recording.
            seizure_intervals: List of (onset, end) times in seconds.
        """
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

        # Drop DC drift and high-frequency noise outside the band of interest.
        raw.filter(0.5, 40., fir_design='firwin', verbose=False)

        # Some sessions were recorded at 512 Hz; standardize them all.
        raw.resample(self.target_fs, verbose=False)

        data = raw.get_data()  # (n_channels, n_samples)
        _, n_samples = data.shape

        windows = []
        labels = []

        for start in range(0, n_samples - self.samples_per_window, self.stride):
            end = start + self.samples_per_window
            window = data[:, start:end].astype(np.float32, copy=False)

            # Per-channel z-normalization within the window.
            window -= window.mean(axis=1, keepdims=True, dtype=np.float32)
            window /= (window.std(axis=1, keepdims=True, dtype=np.float32) + 1e-6)

            # A window counts as a seizure if any part of it overlaps any
            # annotated seizure interval.
            window_start = start / self.target_fs
            window_end = window_start + self.window_size
            is_seizure = any(
                window_start < seizure_end and window_end > onset
                for onset, seizure_end in seizure_intervals
            )

            windows.append(window)
            labels.append(1 if is_seizure else 0)

        return windows, labels

    def _load_all_data(self):
        """Walk the patient directories and accumulate every window."""
        windows = []
        labels = []
        recording_ids = []

        patient_dirs = sorted(d for d in self.data_dir.iterdir() if d.is_dir())
        if self.max_patients is not None:
            patient_dirs = patient_dirs[:self.max_patients]

        for patient_dir in patient_dirs:
            print(f"Loading {patient_dir.name}...")

            seizure_info = self._parse_summary(patient_dir)

            edf_files = sorted(patient_dir.glob("*.edf"))
            if self.max_files_per_patient is not None:
                edf_files = edf_files[:self.max_files_per_patient]

            for edf_file in edf_files:
                intervals = seizure_info.get(edf_file.name, [])
                file_windows, file_labels = self._load_edf(edf_file, intervals)

                rec_id = len(self.recordings)
                self.recordings.append(f"{patient_dir.name}/{edf_file.name}")
                self.recording_has_seizure.append(bool(intervals))

                windows.extend(file_windows)
                labels.extend(file_labels)
                recording_ids.extend([rec_id] * len(file_windows))

                if self.max_windows is not None and len(windows) >= self.max_windows:
                    print(f"Reached max_windows={self.max_windows}, stopping early.")
                    break

            if self.max_windows is not None and len(windows) >= self.max_windows:
                break

        print(
            f"Total windows: {len(windows)} from {len(self.recordings)} recordings "
            f"({sum(self.recording_has_seizure)} with seizures), "
            f"Seizure: {int(np.sum(labels))}, "
            f"Non-seizure: {len(labels) - int(np.sum(labels))}"
        )

        if len(windows) == 0:
            return np.array([]), np.array([]), np.array([])

        # Channel counts vary slightly between recordings; keep only the
        # dominant shape so the windows can be stacked into one array.
        expected_shape = windows[0].shape
        good_windows, good_labels, good_recs = [], [], []
        for w, y, r in zip(windows, labels, recording_ids):
            if w.shape == expected_shape:
                good_windows.append(w)
                good_labels.append(y)
                good_recs.append(r)

        print(f"Keeping {len(good_windows)} windows after filtering shape mismatches")

        return (
            np.array(good_windows, dtype=np.float32),
            np.array(good_labels, dtype=np.int64),
            np.array(good_recs, dtype=np.int64),
        )

    def _parse_summary(self, patient_dir):
        """Extract seizure intervals from a patient's summary file.

        Returns a dict mapping EDF filename -> list of (onset, end) times in
        seconds. Files with no seizures map to an empty list.
        """
        summary_file = patient_dir / f"{patient_dir.name}-summary.txt"
        seizure_info = {}

        if not summary_file.exists():
            return seizure_info

        with open(summary_file, 'r') as f:
            lines = f.readlines()

        current_file = None
        pending_onset = None
        for line in lines:
            if 'File Name:' in line:
                current_file = line.split(':')[1].strip()
                seizure_info[current_file] = []
                pending_onset = None
            elif 'Seizure' in line and 'Start Time:' in line and current_file:
                pending_onset = int(line.split(':')[1].strip().split()[0])
            elif 'Seizure' in line and 'End Time:' in line and current_file:
                end = int(line.split(':')[1].strip().split()[0])
                if pending_onset is not None:
                    seizure_info[current_file].append((pending_onset, end))
                    pending_onset = None

        return seizure_info

    @property
    def stride_seconds(self):
        """Seconds of recording each successive window advances by."""
        return self.stride / self.target_fs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.windows[idx])
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def make_spectrogram(window, nperseg=64, noverlap=32, target_fs=256):
    """Convert a multi-channel EEG window into a single-channel spectrogram.

    Channels are averaged into one signal before the STFT, so the 2D CNN sees
    time-frequency structure but not per-channel detail.

    Args:
        window: Array of shape (n_channels, n_samples), e.g. (23, 1024).
        nperseg: STFT segment length.
        noverlap: STFT overlap between segments.
        target_fs: Sampling frequency in Hz.

    Returns:
        Array of shape (1, freq_bins, time_bins), min-max scaled to [0, 1].
    """
    averaged_signal = np.mean(window, axis=0)

    f, t, Sxx = signal.spectrogram(
        averaged_signal,
        fs=target_fs,
        nperseg=nperseg,
        noverlap=noverlap,
        mode='magnitude',
    )

    # Log scaling compresses the large dynamic range of EEG power.
    Sxx = np.log1p(Sxx)

    Sxx_min, Sxx_max = Sxx.min(), Sxx.max()
    if Sxx_max > Sxx_min:
        Sxx = (Sxx - Sxx_min) / (Sxx_max - Sxx_min)

    return Sxx[np.newaxis, :, :].astype(np.float32)


class SpectrogramDataset(Dataset):
    """Wraps a CHBMITDataset and yields spectrograms instead of raw windows."""

    def __init__(self, base_dataset, nperseg=64, noverlap=32):
        self.base_dataset = base_dataset
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.target_fs = base_dataset.target_fs

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        window, label = self.base_dataset[idx]

        if isinstance(window, torch.Tensor):
            window = window.numpy()

        spectrogram = make_spectrogram(
            window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            target_fs=self.target_fs,
        )

        return torch.FloatTensor(spectrogram), torch.tensor(int(label), dtype=torch.long)
