import os
import torch

from dataclasses import dataclass
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Iterable
from whisper.model import (
    ModelDimensions, 
    ResidualAttentionBlock, 
    sinusoids, 
    AudioEncoder
)
from pytorch_metric_learning.losses import CosFaceLoss, LargeMarginSoftmaxLoss
from omegaconf import DictConfig


__all__ = [
    "SRModelPreTrained",
    "SRTrainableModel"
]

@dataclass
class ModelConfig:
    name_or_path: str
    frame_out_dim: int = 1024


# NOTE: 
#       this class is copy-paste from whisper.model.AudioEncoder
#       the applying of positional enoding is changed
class MyAudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        # x = (x + self.positional_embedding).to(x.dtype)
        x = (x + self.positional_embedding[: x.shape[1], ...]).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class SRFrameLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1)
        )
        
    def forward(self, x: Tensor):
        return self.tdnn(x)


class SRPolling(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        mu = torch.mean(x, dim=-1)
        std = torch.std(x, dim=-1)

        return torch.cat([mu, std], dim=-1)


class SRSegmentLayer(nn.Module):
    def __init__(self, in_features: int, embedding_dim: int):
        super().__init__()
        self.proj1 = nn.Linear(in_features, embedding_dim)
        self.proj2 = nn.Linear(in_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim, affine=False)

    def forward(self, x: Tensor):
        return self.bn(torch.max(self.proj1(x), self.proj2(x)))


class SRModelPreTrained(nn.Module):
    def __init__(self, path: str, frame_out_dim: int = 1024):
        super().__init__()

        encoder_state = torch.load(os.path.expanduser(path))
        self.dims = ModelDimensions(**encoder_state["dims"])
        self.encoder = MyAudioEncoder(n_mels=self.dims.n_mels,
                                    n_ctx=self.dims.n_audio_ctx,
                                    n_state=self.dims.n_audio_state,
                                    n_head=self.dims.n_audio_head,
                                    n_layer=self.dims.n_audio_layer)
        self.encoder.load_state_dict(encoder_state["model_state"], strict=False)

        self.frame_layer = SRFrameLayer(self.dims.n_audio_state, frame_out_dim)
        self.pooling = SRPolling()
        self.segment_layer = SRSegmentLayer(2 * frame_out_dim, frame_out_dim)  

    def forward(self, input_features: Tensor):
        out: Tensor = self.encoder(input_features)
        out = self.frame_layer(out.permute(0, 2, 1))
        out = self.pooling(out)
        out = self.segment_layer(out)
        
        return out
    

class SRTrainableModel(SRModelPreTrained):

    def __init__(self, config: ModelConfig, 
                 n_classes: int, 
                 criterion_cfg: DictConfig, 
                 *inputs, **kwargs):
        super().__init__(config.name_or_path, config.frame_out_dim, *inputs, **kwargs)

        self.config = config
        if criterion_cfg is None:
            criterion_cfg = {}
        self.criterion = CosFaceLoss(n_classes, 
                                        self.config.frame_out_dim, 
                                        **criterion_cfg)

    def forward(self, input_features: Tensor, labels: Tensor = None, utt_id: Tensor = None):
        embedding = super().forward(input_features)

        loss = None
        if not labels is None and self.training:
            loss = self.criterion(embedding, labels)
        else:
            return tuple(output.cpu() for output in [embedding, labels, utt_id])
     
        return loss,