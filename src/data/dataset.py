import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import os
import ctypes
from src.features.extractor import FeatureExtractor
from config import MAX_AUDIO_LENGTH, SAMPLE_RATE, DATA_DIR, LANGUAGES


def is_file_offline(file_path):
    """Check if file is offline in Windows OneDrive without recalling it."""
    if os.name != 'nt':
        return False
    try:
        attrs = ctypes.windll.kernel32.GetFileAttributesW(file_path)
        if attrs == -1:
            return False
        # 0x1000 = FILE_ATTRIBUTE_OFFLINE, 0x400000 = FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
        return bool(attrs & 0x1000) or bool(attrs & 0x400000)
    except Exception:
        return False


class IndianAccentDataset(Dataset):
    """
    Dataset class for Indian Accent Database (raw audio -> on-the-fly feature extraction).
    """
    def __init__(self, data_dir=None, feature_type='mfcc', augment=False):
        self.data_dir = data_dir or os.path.join(DATA_DIR, 'IndicAccentDB')
        self.feature_type = feature_type
        self.augment = augment
        self.feature_extractor = FeatureExtractor(feature_type=feature_type)
        
        # Load dataset
        self.audio_paths, raw_labels = self._load_dataset()
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(raw_labels)))}
        self.labels = [self.label_to_idx[label] for label in raw_labels]

    def _load_dataset(self):
        audio_paths = []
        labels = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_paths.append(os.path.join(root, file))
                    labels.append(os.path.basename(root))
        return audio_paths, labels

    def _augment_audio(self, waveform, sample_rate):
        noise = np.random.normal(0, 0.005, waveform.shape)
        waveform = waveform + noise
        n_steps = np.random.uniform(-2, 2)
        waveform = librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=n_steps)
        rate = np.random.uniform(0.9, 1.1)
        waveform = librosa.effects.time_stretch(waveform, rate=rate)
        return waveform

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        try:
            audio_path = self.audio_paths[idx]
            waveform, sample_rate = librosa.load(audio_path, sr=None)
            
            if self.augment and np.random.rand() < 0.5:
                waveform = self._augment_audio(waveform, sample_rate)
            
            # Truncate/pad to consistent length
            target_length = int(MAX_AUDIO_LENGTH * sample_rate)
            if len(waveform) > target_length:
                waveform = waveform[:target_length]
            elif len(waveform) < target_length:
                waveform = np.pad(waveform, (0, target_length - len(waveform)), 'constant')
            
            features = self.feature_extractor.extract_features(torch.FloatTensor(waveform), sample_rate)
            
            # Pad/trim to 1000 time frames for MFCC
            if self.feature_type == 'mfcc':
                if features.shape[1] < 1000:
                    features = torch.nn.functional.pad(features, (0, 1000 - features.shape[1]))
                elif features.shape[1] > 1000:
                    features = features[:, :1000]
            
            return features, self.labels[idx]
        except Exception as e:
            print(f"Error processing item {idx} ({self.audio_paths[idx]}): {e}")
            if self.feature_type == 'mfcc':
                return torch.zeros(39, 1000), 0
            else:
                return torch.zeros(1000, 768), 0


def collate_fn(batch):
    """
    Collate function for DataLoader.
    Handles both MFCC (channels, time) and HuBERT (time, hidden) shapes.
    """
    features, labels = zip(*batch)
    
    first = features[0]
    # MFCC: first dim is small (channels: 13 or 39); HuBERT: first dim is time (large)
    is_mfcc = (first.ndim == 2 and first.shape[0] < 200)
    
    if is_mfcc:
        # All MFCC tensors are already padded to (channels, 1000) — stack directly
        features = torch.stack(features)
    else:
        # HuBERT: variable-length sequences, pad to max
        max_len = max(f.shape[0] for f in features)
        padded = []
        for f in features:
            if f.shape[0] < max_len:
                f = torch.nn.functional.pad(f, (0, 0, 0, max_len - f.shape[0]))
            elif f.shape[0] > max_len:
                f = f[:max_len, :]
            padded.append(f.unsqueeze(0))
        features = torch.cat(padded, dim=0)
    
    labels = torch.tensor(labels, dtype=torch.long)
    return features, labels


def create_data_loader(dataset, batch_size=16, shuffle=True):
    """Create a DataLoader for the given dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # 0 avoids issues on Windows
        pin_memory=torch.cuda.is_available(),
    )


class FeatureDataset(Dataset):
    """
    Dataset class to load pre-extracted features (MFCC or HuBERT).

    Supports two directory layouts:
    - **Flat layout**: all .npy files live in ``features_dir`` itself, named
      ``<lang>_<speaker>_<number>.npy``.  This is the main layout used for MFCC.
    - **Subdir layout**: .npy files live in ``features_dir/<lang>/``.  This is
      the layout used for HuBERT layer features.

    When flat files exist they take priority over subdir files.  Files whose
    channel-count does not match the dominant channel-count (determined by
    sampling the first loadable flat file) are silently discarded so that
    ``torch.stack`` never fails on shape mismatches.
    """

    def __init__(self, features_dir: str, augment=False):
        self.features_dir = features_dir
        self.is_mfcc = "mfcc" in features_dir.lower()
        self.augment = augment
        self.label_to_idx = {lang: idx for idx, lang in enumerate(LANGUAGES)}

        # ── Step 1: Collect candidate (path, label_idx) pairs ────────────────

        # Flat layout: files in the root dir named with a language prefix
        flat_candidates: list[tuple[str, int]] = []
        for filename in os.listdir(features_dir):
            if not filename.endswith('.npy'):
                continue
            filepath = os.path.join(features_dir, filename)
            if is_file_offline(filepath):
                continue
            for lang in LANGUAGES:
                if filename.startswith(lang + '_'):
                    flat_candidates.append((filepath, self.label_to_idx[lang]))
                    break

        # Subdir layout: files nested under a language-named subdirectory
        subdir_candidates: list[tuple[str, int]] = []
        for root, _dirs, files in os.walk(features_dir):
            if root == features_dir:
                continue
            label_str = os.path.basename(root)
            if label_str not in self.label_to_idx:
                continue
            for filename in files:
                if not filename.endswith('.npy'):
                    continue
                filepath = os.path.join(root, filename)
                if is_file_offline(filepath):
                    continue
                subdir_candidates.append((filepath, self.label_to_idx[label_str]))

        # Flat files have priority; fall back to subdir layout if no flat files
        all_candidates = flat_candidates if flat_candidates else subdir_candidates

        # Set default expected channel count (no file loading during init!)
        # Files with wrong shapes will produce errors in __getitem__ but we handle those gracefully.
        self.expected_ch = 39 if self.is_mfcc else 768

        self.file_list: list[str] = [fp for fp, _ in all_candidates]
        self.label_list: list[int] = [li for _, li in all_candidates]

        n_flat = len(flat_candidates)
        n_sub = len(subdir_candidates)
        print(f"FeatureDataset: {len(self.file_list)} files indexed "
              f"(flat={n_flat}, subdir={n_sub}, expected_ch={self.expected_ch})")


    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        n_ch = self.expected_ch or (39 if self.is_mfcc else 768)
        try:
            arr = np.load(self.file_list[idx])
            features = torch.FloatTensor(arr)
        except Exception:
            # Fallback zero tensor (e.g. offline OneDrive placeholder)
            if self.is_mfcc:
                return torch.zeros(n_ch, 1000), self.label_list[idx]
            else:
                return torch.zeros(1000, n_ch), self.label_list[idx]

        target = 1000
        if self.is_mfcc:
            # Shape: (channels, time) — pad/trim the time axis
            t = features.shape[1]
            if t < target:
                features = torch.nn.functional.pad(features, (0, target - t))
            elif t > target:
                features = features[:, :target]
            
            # Apply SpecAugment if enabled (random time/frequency masking)
            if self.augment and np.random.rand() < 0.5:
                features = self._spec_augment(features)
        else:
            # Shape: (time, hidden) — pad/trim the time axis
            t = features.shape[0]
            if t < target:
                features = torch.nn.functional.pad(features, (0, 0, 0, target - t))
            elif t > target:
                features = features[:target, :]

        return features, self.label_list[idx]
    
    def _spec_augment(self, features):
        """
        Apply SpecAugment-style augmentation to MFCC features
        - Random time masking: mask out a contiguous section on the time axis
        - Random frequency masking: mask out a contiguous section on the frequency axis
        """
        features = features.clone()
        
        # Time masking (mask T consecutive frames)
        time_mask_param = 50  # Max number of frames to mask
        t_mask = np.random.randint(0, time_mask_param)
        if t_mask > 0:
            t_start = np.random.randint(0, max(1, features.shape[1] - t_mask))
            features[:, t_start:t_start+t_mask] = 0.0
        
        # Frequency masking (mask F consecutive channels)
        freq_mask_param = 8  # Max number of channels to mask
        f_mask = np.random.randint(0, freq_mask_param)
        if f_mask > 0:
            f_start = np.random.randint(0, max(1, features.shape[0] - f_mask))
            features[f_start:f_start+f_mask, :] = 0.0
        
        return features