import torch
from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import functools
import logging
from tqdm import tqdm
from pathlib import Path

import sys

sys.path.append("..")

from sharegpt4v import share4v_val_dataset, share4v_train_dataset
from model import longclip
import argparse
import os
import torch.optim as optim
from accelerate.logging import get_logger
import numpy as np
import datetime
import warnings
import math
from accelerate import Accelerator
from accelerate import FullyShardedDataParallelPlugin
from accelerate.utils import ProjectConfiguration, set_seed


warnings.filterwarnings("ignore")

logger = get_logger(__name__, log_level="INFO")


class CLIP_Clean_Train:
    def __init__(
        self,
        train_batch_size=8,
        num_epoch=6,
        lr=1e-6,
        weight_decay=0.01,
        log_scale=4.6052,
        warmup_length=200,
        base_model="ViT-B/16",
        output_dir="longclip",
    ):

        self.base_model = base_model
        self.model, _ = longclip.load_from_clip(self.base_model, device="cpu")
        self.model = self.model.float()

        self.batch_size = train_batch_size
        self.num_epoch = num_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_length = warmup_length

        self.ckptdir = Path(output_dir, "ckpt/")
        os.makedirs(self.ckptdir, exist_ok=True)

        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * log_scale)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        trainset = share4v_train_dataset(
            batch_size=self.batch_size, num_processes=accelerator.num_processes
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, num_workers=32, pin_memory=True
        )

        self.test_dataloader = None
        self.num_update_steps_per_epoch = (
            math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
            // accelerator.num_processes
        )
        max_train_steps = self.num_epoch * self.num_update_steps_per_epoch
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer,
        #     T_max=max_train_steps,
        #     eta_min=0,
        # )
        warm_up_iter = 200
        T_max = max_train_steps  # 周期
        lr_min = 0  # 最小值

        lambda0 = lambda cur_iter: (
            cur_iter / warm_up_iter
            if cur_iter < warm_up_iter
            else (
                lr_min
                + 0.5
                * (
                    1.0
                    + math.cos(
                        (cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi
                    )
                )
            )
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=[lambda0]
        )
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.test_dataloader,
            self.lr_scheduler,
        ) = accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.test_dataloader,
            self.lr_scheduler,
        )
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
        self.progress_bar = tqdm(
            range(0, max_train_steps),
            initial=0,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

    def train(self, resume_iter=0):

        start_epoch = 0
        resume_iter = 0

        for epoch in range(start_epoch, self.num_epoch):

            train_loss_total = 0.0
            train_loss = 0.0
            train_loss_short = 0.0

            global_step = 0
            for images, texts, texts_short, targets in self.train_dataloader:

                if global_step < resume_iter:
                    continue

                with accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()

                    texts = longclip.tokenize(texts, truncate=True)

                    # loss = self.inference(images, texts)
                    image_features, text_features, logit_scale = self.model(
                        images, texts
                    )
                    image_feat_all = accelerator.gather(image_features)
                    text_feat_all = accelerator.gather(text_features)

                    sim_i2t = torch.matmul(image_features, text_feat_all.T)
                    sim_t2i = torch.matmul(text_features, image_feat_all.T)
                    sim_i2t = logit_scale * sim_i2t
                    sim_t2i = logit_scale * sim_t2i
                    loss = (
                        F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                        + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                    ) / 2
                    loss_short = 0.0
                    loss_total = loss

                    try:
                        images_short = images.clone()
                        texts_short = longclip.tokenize(texts_short, truncate=True)
                        image_features_short, text_features_short, logit_scale_short = (
                            self.model(images_short, texts_short)
                        )
                        image_feat_all_short = accelerator.gather(image_features_short)
                        text_feat_all_short = accelerator.gather(text_features_short)
                        sim_i2t_short = torch.matmul(
                            image_features_short, text_feat_all_short.T
                        )
                        sim_t2i_short = torch.matmul(
                            text_features_short, image_feat_all_short.T
                        )
                        sim_i2t_short = logit_scale_short * sim_i2t_short
                        sim_t2i_short = logit_scale_short * sim_t2i_short

                        loss_short = (
                            0.1
                            * (
                                F.cross_entropy(
                                    sim_i2t_short, targets, label_smoothing=0.1
                                )
                                + F.cross_entropy(
                                    sim_t2i_short, targets, label_smoothing=0.1
                                )
                            )
                            / 2
                        )
                        loss_total += loss_short
                        accelerator.backward(loss_total)
                        # accelerator.backward(loss, retain_graph=True)
                        # accelerator.backward(loss_short, retain_graph=True)

                    except Exception as e:
                        # SVD may encounter infs, very rare occasion.
                        print("SVD may encounter infs, very rare occasion.")
                        logger.error("SVD may encounter infs, very rare occasion.")
                        print(e)
                        logger.error(e)

                        accelerator.backward(loss)
                    avg_loss_total = accelerator.gather(
                        loss_total.repeat(args.train_batch_size)
                    ).mean()
                    train_loss_total += (
                        avg_loss_total.item() / args.gradient_accumulation_steps
                    )

                    avg_loss = accelerator.gather(
                        loss.repeat(args.train_batch_size)
                    ).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    avg_loss_short = accelerator.gather(
                        loss_short.repeat(args.train_batch_size)
                    ).mean()
                    train_loss_short += (
                        avg_loss_short.item() / args.gradient_accumulation_steps
                    )

                    self.optimizer.step()

                if accelerator.sync_gradients:
                    self.progress_bar.update(1)
                    global_step += 1
                    train_loss = 0.0
                    self.lr_scheduler.step()
                    if global_step % 100 == 0:
                        if accelerator.is_main_process:
                            accelerator.log(
                                {"train_loss_total": train_loss_total}, step=global_step
                            )
                            accelerator.log(
                                {"train_loss": train_loss}, step=global_step
                            )
                            accelerator.log(
                                {"train_loss_short": train_loss_short}, step=global_step
                            )
                            accelerator.log(
                                {"lr": self.lr_scheduler.get_last_lr()[0]},
                                step=global_step,
                            )
                            accelerator.log(
                                {"logit_scale": self.model.module.logit_scale.item()},
                                step=global_step,
                            )

                            # self.test()
                logs = {
                    "step_loss": loss.detach().item(),
                    "step_loss_short": loss_short.detach().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                self.progress_bar.set_postfix(**logs)

            if accelerator.is_main_process:
                logger.info(
                    f"loss, loss_short  after training epoch {epoch}: {loss}, {loss_short}"
                )
                print("=====================================")
                print(
                    f"loss, loss_short  after training epoch {epoch}: {loss}, {loss_short}"
                )
                print("=====================================")

                if self.base_model == "ViT-B/16":
                    name = "longclip-B.pt"
                elif self.base_model == "ViT-L/14":
                    name = "longclip-L.pt"
                elif (
                    self.base_model
                    == "/ML-A100/team/mm/gujiasheng/Long-CLIP/ViT-bigG-14-laion2b_s39b_b160k.pt"
                ):
                    name = "longclip-bigG.pt"
                else:
                    name = "longclip-others.pt"
                file_path = Path(self.ckptdir, name + f"_epoch_{epoch}")
                unwrapped_model = accelerator.unwrap_model(self.model)
                torch.save(unwrapped_model.state_dict(), file_path)

    def test(self):
        if self.test_dataloader is None:
            testset = share4v_val_dataset()
            self.test_dataloader = torch.utils.data.DataLoader(
                testset, batch_size=32, num_workers=32, pin_memory=True
            )
        with torch.no_grad():
            self.model.eval()
            for images, texts in tqdm(self.test_dataloader):
                texts = longclip.tokenize(texts, truncate=True)
                logits_per_image, logits_per_text = self.model(images, texts)

                bs = images.shape[0]
                targets = torch.linspace(0, bs - 1, bs, dtype=int).to(
                    logits_per_image.device
                )
                correct = 0
                correct += (logits_per_text.argmax(1) == targets).sum().item()

            acc = correct / bs
            print("=====================================")
            print(f"test mean of share4v retrieval: {acc}")
            print("=====================================")
            self.model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="params")
    parser.add_argument("--lr", default=1e-6, type=float, help="lr.")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="wd.")
    parser.add_argument(
        "--log_scale", default=4.6052, type=float, help="clip temperature log scale."
    )
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument(
        "--base_model",
        default="/ML-A100/team/mm/gujiasheng/Long-CLIP/ViT-bigG-14-laion2b_s39b_b160k.pt",
        help="CLIP Base Model",
    )
    parser.add_argument("--resume_iter", default=0, type=int, help="resume iteration")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--train_batch_size", default=128, type=int)
    parser.add_argument("--num_epoch", default=6, type=int)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="auto",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    args = parser.parse_args()
    if args.output_dir == "auto":
        output_dir = f"lr={args.lr}_wd={args.weight_decay}_wl={args.warmup_length}_log_scale={args.log_scale}_bs={args.train_batch_size}"
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = Path(output_dir, args.logging_dir)
    os.makedirs(logging_dir, exist_ok=True)
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=logging_dir
    )

    os.environ["FSDP_ACTIVATION_CHECKPOINTING"] = "true"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    accelerator.init_trackers(output_dir)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info("output_dir: %s", output_dir)
    logger.info("logging_dir: %s", logging_dir)
    args.lr = args.lr * args.gradient_accumulation_steps

    trainer = CLIP_Clean_Train(
        train_batch_size=args.train_batch_size,
        num_epoch=args.num_epoch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_scale=args.log_scale,
        warmup_length=args.warmup_length,
        base_model=args.base_model,
        output_dir=output_dir,
    )
    trainer.train(resume_iter=args.resume_iter)
    accelerator.end_training()
