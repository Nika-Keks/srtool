import numpy as np

from pathlib import Path
from typing import Union


def parse_utt2spk(path: Union[Path, str]):
    u2s = {}
    durations = {}
    with open(path) as u2sfile:
        for line in u2sfile:
            utt, spk, dur = line.strip().split()
            u2s[utt] = spk
            durations[utt] = int(dur)

    return u2s, durations
    
def parse_protocol(path: Union[str, Path], sep: str = "\t"):
    with open(path) as ifile:
        protocol = set(
            tuple(line.strip().split(sep))
            for line in ifile
            if line.count(sep) == 1
        )
    return protocol

def prepare_protocol(utt2spk: dict, protocol: list):
        utts = sorted(list(utt2spk.keys()))
        utts = [Path(utt).stem for utt in utts]

        utt_ids = [[utts.index(item[0]), utts.index(item[1])]
                   for item in protocol
                   if item[0] in utts and item[1] in utts]
        
        return np.array(utt_ids).astype(np.int32)

def compute_scors(embeddings: np.array, 
                  utt_ids: np.array, 
                  protocol: np.array):
        utt_ids = np.argsort(utt_ids)
        a_vec, b_vec = [embeddings[utt_ids[protocol[:, i]]][..., np.newaxis, :] for i in [0, 1]]

        vec_dot = lambda x, y: x @ np.transpose(y, axes=(0, 2, 1))
        norm = lambda x: np.sqrt(vec_dot(x, x)) + np.finfo(x.dtype).eps
        scores = vec_dot(a_vec, b_vec) / norm(a_vec) / norm(b_vec)

        return np.squeeze(scores)

