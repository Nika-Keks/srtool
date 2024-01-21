import hydra
import logging

from omegaconf import DictConfig, OmegaConf

from srtool.models import SRTrainableModel
from srtool.data import AudioDataset
from srtool.metrics import EERMetric
from srtool.train import Trainer, TrainerConfig


logger = logging.getLogger(__file__)

@hydra.main(version_base=None, config_path="../../configs", config_name="train_task")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logger.info(f"CONFIG:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(cfg.training_args.tensorboard_logging.log_dir)
    # training_args = TrainingArguments(**cfg.training_args)
    logger.info("init train dataset")
    train_dataset = AudioDataset(**cfg.train_dataset)
    logger.info("init eval dataset")
    eval_dataset = AudioDataset(**cfg.eval_dataset)

    logger.info("create model")
    model = SRTrainableModel(cfg.model,
                             n_classes=train_dataset.n_classes,
                             criterion_cfg=cfg.criterion)

    compute_metrics = EERMetric(**cfg.compute_metrics)
    # trainer = Trainer(model=model,
    #                   args=training_args,
    #                   train_dataset=train_dataset,
    #                   eval_dataset=eval_dataset,
    #                   compute_metrics=compute_metrics)
    trainer = Trainer(cfg.training_args)
    
    trainer.run(model, train_dataset, eval_dataset, compute_metrics)

if __name__ == "__main__":
    logger.info("run main")
    main()