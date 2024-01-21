import os
import sys
import random
import torch
import logging

from contextlib import nullcontext
from dataclasses import dataclass
from torch import nn
from torch import Tensor
from torch.utils.data import (
    DataLoader, 
    Dataset, 
    DistributedSampler
)
from torch import distributed as dist
from torch.optim import (
    Optimizer,
    SGD,
    AdamW,
    lr_scheduler
)
from torch import multiprocessing as mp
from torch.multiprocessing.spawn import start_processes
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Union, Callable, Tuple, Any
from omegaconf import DictConfig


logger = logging.getLogger(__file__)

@dataclass
class TrainerConfig:
    devices: int
    max_epoch: int
    max_iterations: int
    optimiser: DictConfig
    scheduler: DictConfig
    dataloader_args: DictConfig
    tensorboard_logging: DictConfig
    ckpt_path: str
    grad_acumulation: int = 1

@dataclass
class _TrainerHyperparameters:
    iterations: int = 0
    epoch: int = 0
    updates: int = 0


class Trainer:

    _OPTIMISERS = {
        "sgd": SGD,
        "adamw": AdamW
    }

    def __init__(self,
                 config: TrainerConfig
                 ) -> None:
        logger.info("init trainer")
        self.config = config
        self.params = _TrainerHyperparameters()
        self.logger = None
        self.clilog = None

    def _setup_ddp(self, rank: int):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345" #f"{3_000 + random.randint(0, 1_000)}"
        
        dist.init_process_group(backend="nccl", 
                                world_size=self.config.devices,
                                rank=rank)

    def run(self, model: nn.Module, 
            train_dataset: Dataset, 
            eval_dataset: Dataset,
            compute_metric: Callable):
        mp.spawn(self._train_impl,
                 (model, train_dataset, eval_dataset, compute_metric),
                 nprocs=self.config.devices)
        
    def _train_impl(self, rank: int, 
                    model: nn.Module, 
                    train_dataset: Dataset, 
                    eval_dataset: Dataset,
                    compute_metric: Callable):
        self._setup_logger()
        self._setup_ddp(rank)
        ddp_model = self._setup_model(model)
        train_dataloader = self._setup_train_dataloader(train_dataset)
        train_set_len = len(train_dataloader)
        eval_dataloader = self._setup_eval_dataloader(eval_dataset)
        optimiser, scheduler = self._setup_optimiser(ddp_model, train_set_len)
        
        for sample in self._iterate_dataset(train_dataloader):
            opt_step_now = ((self.params.iterations + 1) % self.config.grad_acumulation) == 0
            last_in_epoch = (self.params.iterations + 1) % train_set_len == 0
            with (ddp_model.no_sync() if opt_step_now else nullcontext()):
                loss, *_ = self._train_step(ddp_model, sample)
                (loss / self.config.grad_acumulation).backward()
            if opt_step_now or last_in_epoch:
                self._optimiser_step(optimiser, scheduler)

            self._log(f"Loss/loss", loss)

            if last_in_epoch:
                metric = self._validate(ddp_model, eval_dataloader, compute_metric)
                for name, val in metric.items():
                    self._log(f"Metric/{name}", val)
                self._save_chpt(ddp_model, optimiser)
            
        self._save_chpt(ddp_model, optimiser, "ckpt-last.pt")
    
    def _iterate_dataset(self, dataloader: DataLoader):
        while self.params.epoch < self.config.max_epoch and \
              self.params.iterations < self.config.max_iterations:
            
            iterable = tqdm(dataloader, 
                            desc=f"Epoch: {self.params.epoch}/{self.config.max_epoch}",
                            file=sys.stdout) if dist.get_rank() == 0 else dataloader
            
            for sample in iterable:
                yield sample
                self.params.iterations += 1
                self.params.updates = self.params.iterations // self.config.grad_acumulation
            self.params.epoch += 1

    def _setup_model(self, model: nn.Module):
        ddp_model = DistributedDataParallel(model.to(dist.get_rank()), device_ids=[dist.get_rank()])

        return ddp_model

    def _train_step(self, model: nn.Module, samples: tuple):
        loss, *model_out = model(*samples)

        return loss, *model_out
    
    def _setup_optimiser(self, model: nn.Module, dataset_len: int):
        if self.config.optimiser.name not in Trainer._OPTIMISERS:
            raise ValueError(f"optimiser should be in {list(Trainer._OPTIMISERS)}")
        
        optimiser_class = Trainer._OPTIMISERS[self.config.optimiser.name]
        
        optimiser = optimiser_class(model.parameters(), **self.config.optimiser.args)

        scheduler = lr_scheduler.OneCycleLR(optimiser, 
                                            **self.config.scheduler.args,
                                            epochs=self.config.max_epoch,
                                            steps_per_epoch=dataset_len // self.config.grad_acumulation)

        return optimiser, scheduler
        
    def _optimiser_step(self, optimiser: Optimizer, scheduler: Any = None):
        optimiser.step()
        if scheduler is not None:
            scheduler.step()
        optimiser.zero_grad()

    def _setup_train_dataloader(self, train_dataset: Dataset):
        dataloader = DataLoader(train_dataset,
                                sampler=DistributedSampler(train_dataset,
                                                           shuffle=False),
                                collate_fn=train_dataset.collate,
                                **self.config.dataloader_args.train)
        
        return dataloader
    
    def _setup_eval_dataloader(self, eval_dataset: Dataset):
        return DataLoader(eval_dataset, 
                          shuffle=False, 
                          collate_fn=eval_dataset.collate,
                          **self.config.dataloader_args.eval)

    def _save_chpt(self, model: nn.Module, optimiser: Optimizer, name: str = None):
        if dist.get_rank() != 0:
            return
        ckpt = {}
        ckpt["model_state"] = model.state_dict()
        ckpt["optimiser_state"] = optimiser.state_dict()
        ckpt["train_params"] = vars(self.params)

        if name is None:
            name = f"e{self.params.epoch}-s{self.params.iterations}.pt"

        os.makedirs(self.config.ckpt_path, exist_ok=True)
        torch.save(ckpt, os.path.join(self.config.ckpt_path, name))

    def _setup_logger(self):
        self.logger = SummaryWriter(**self.config.tensorboard_logging)

    def _log(self, name: str, value: Union[float, Tensor, dict], step: int = None):
        if value is None or name is None or dist.get_rank():
            return
        if step is None:
            step = self.params.iterations
        
        if isinstance(value, dict):
            self.logger.add_scalars(name, value, global_step=step)
        else:
            if isinstance(value, Tensor):
                value = value.item()
            self.logger.add_scalar(name, value, global_step=step)

    def _validate(self, model: DistributedDataParallel, eval_dataloader: DataLoader, compute_metric: Callable):
        if dist.get_rank() == 0:
            is_train = model.training
            model.eval()
            with torch.no_grad():
                batched_predicts = [model(*sample) for sample in tqdm(eval_dataloader, desc="Validation: ")]
            predicts = [torch.concat([model_out[i] for model_out in batched_predicts]) 
                        for i in range(len(batched_predicts[0]))]
            model.train(is_train)
            metric = compute_metric(predicts)
        else:
            metric = {}

        return metric

