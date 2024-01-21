import numpy as np

from pathlib import Path

from srtool import utils as srutils

def cdf(scores, x):
    yp = np.arange(1, 1 + scores.shape[0]) / scores.shape[0]
    xp = np.sort(scores, axis=0)
    
    return np.interp(x, xp, yp)

def compute_eer(tar_scores: np.array, imp_scores: np.array):
    x = np.sort(np.concatenate([tar_scores, imp_scores], axis=0))

    far = 1 - cdf(imp_scores, x)
    frr = cdf(tar_scores, x)

    i = np.argmin(np.abs(far - frr))

    return frr[i], x[i]

class EERMetric:

    def __init__(self, data_path: str,
                 utt2spk_path: str = "utt2spk",
                 imposters_path: str = "imposters",
                 targets_path: str = "targets"):
        self.data_path = Path(data_path)

        # parse utt2spk
        self.utt2spk, _ = srutils.parse_utt2spk(self.data_path / utt2spk_path)

        # parse protocols
        targets = srutils.parse_protocol(self.data_path / targets_path)
        self.targets =  srutils.prepare_protocol(self.utt2spk, targets)

        imposters = srutils.parse_protocol(self.data_path / imposters_path)
        self.imposters = srutils.prepare_protocol(self.utt2spk, imposters)

    def __call__(self, predicts):
        embeddings, _, utt_ids = [output.numpy() for output in predicts]
        embeddings = embeddings.astype(np.float32)
        utt_ids = utt_ids.astype(np.int32)

        tar_scores = srutils.compute_scors(embeddings, utt_ids, self.targets)
        imp_scores = srutils.compute_scors(embeddings, utt_ids, self.imposters)

        eer, thresh = compute_eer(tar_scores, imp_scores)

        return {"EER": eer * 100, "EER thresh": thresh}