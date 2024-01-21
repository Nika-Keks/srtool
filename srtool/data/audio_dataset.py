import soundfile as sf
import numpy as np

import torch
from torch.utils.data import Dataset
from whisper import log_mel_spectrogram
from whisper.audio import pad_or_trim
from srtool import utils as srutils


def _read_crop(path: str, crop_size: int, duration: int):
    start = 0 if duration <= crop_size else \
        np.random.randint(0, duration - crop_size)
    stop = min(start + crop_size, duration)

    wav, sr = sf.read(path, start=start, stop=stop, dtype="float32")
    wav = wav.take(np.arange(crop_size) % wav.shape[0])

    return wav, sr


class AudioDataset(Dataset):

    def __init__(self, utt2spk: str, 
                 crop_size: int,
                 sample_rate: int = 16_000) -> None:
        super().__init__()
        self.u2s, self.durations = srutils.parse_utt2spk(utt2spk)
        
        self.spk_lst = sorted(list(set(self.u2s.values())))
        self.wav_lst = sorted(list(self.u2s.keys()))
    
        self.crop_size = crop_size
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.wav_lst)

    def __getitem__(self, index):
        wav_path = self.wav_lst[index]
        duration = self.durations[wav_path]
        wav, sr = _read_crop(wav_path, self.crop_size * self.sample_rate, duration)

        if sr != self.sample_rate:
            raise ValueError(f"file {wav_path} "
                             f"has {sr} sample rate "
                             f"but expected {self.sample_rate}")

        spk_id = self.spk_lst.index(self.u2s[wav_path])
        mel = log_mel_spectrogram(wav)

        return mel, spk_id, index
    
    def collate(self, samples: list):
        mels_min_len = min(sample[0].shape[-1] for sample in samples)
        mels = torch.concat([pad_or_trim(sample[0][None, ...], mels_min_len)
                             for sample in samples])
        int_to_tensor = lambda i: torch.LongTensor([s[i] for s in samples])

        return mels, int_to_tensor(1), int_to_tensor(2)

    @property
    def n_classes(self):
        return len(self.spk_lst)