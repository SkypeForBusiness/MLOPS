import argparse
from types import SimpleNamespace
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from data_module import GLUEDataModule
from model import GLUETransformer
from utils import ENTITY, PROJECT_NAME

default_config = SimpleNamespace(
    lr = 1e-4,
    batch_size = 32,
    epochs = 3,
    warmup_steps = 10,
    grad_batches = 1,
    weight_decay = 0.0,
    adam_epsilon = 1e-8,
    num_workers = 4
    )

def parse_args():
    docstring = """Overriding default argments"""
    parser = argparse.ArgumentParser(
        description=docstring,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="learning rate",
        required=True
    )

    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        help="init mode of wandb logger"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=default_config.lr,
        help="learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_config.batch_size,
        help="batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=default_config.epochs,
        help="number of training epochs",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=default_config.warmup_steps,
        help="lr warmup",
    )
    parser.add_argument(
        "--grad_batches",
        type=int,
        default=default_config.grad_batches,
        help="after n batches calculate gradients",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=default_config.weight_decay,
        help="optimizers weight_decay",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=default_config.adam_epsilon,
        help="optimizer epsilon",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=default_config.num_workers,
        help="dataloader workers",
    )
    args = parser.parse_args()
    return args

# 1: Define objective/training function
def objective(config) -> float:

    seed_everything(42)

    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
        train_batch_size=config.batch_size,
        eval_batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    dm.setup("fit")

    model = GLUETransformer(
        model_name_or_path="distilbert-base-uncased",
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=config.learning_rate,
        train_batch_size=config.batch_size,
        eval_batch_size=config.batch_size,
        adam_epsilon=config.adam_epsilon,
        warmup_steps=config.warmup_steps
    )

    logger = WandbLogger(project=PROJECT_NAME, dir=config.checkpoint_dir)

    trainer = Trainer(
        max_epochs=config.epochs,
        devices="auto",
        accelerator="auto",
        logger=logger,
        accumulate_grad_batches=config.grad_batches
    )

    hyperparameters = dict(learning_rate=config.learning_rate, batch_size=config.batch_size)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=dm)

    return trainer.callback_metrics["val_loss"].item()


def main():
    args = parse_args()
    config = vars(args)
    wandb.init(entity=ENTITY, project=PROJECT_NAME, config=config, mode=config["wandb_mode"])
    loss = objective(wandb.config)
    wandb.log({"val_loss": loss})
    wandb.finish()

if __name__ == "__main__":
    main()