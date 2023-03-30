from datasets import build_dataset
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
import argparse
import json
import torch
from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from functools import partial
import util.misc as utils
import math
from collections import namedtuple
import util.dist as dist
from pathlib import Path
import random
import numpy as np
from models import build_model
from copy import deepcopy
from datasets.vidstg_eval import VidSTGEvaluator
from datasets.hcstvg_eval import HCSTVGEvaluator
from engine import evaluate, train_one_epoch
from models.postprocessors import build_postprocessors
import os
import time
from torch.utils.tensorboard import SummaryWriter
import datetime


def get_args_parser():
    parser = argparse.ArgumentParser("Set SAW", add_help=False)

    # Dataset specific
    parser.add_argument("--dataset_config", default=None, required=True)
    parser.add_argument(
        "--combine_datasets",
        nargs="+",
        help="List of datasets to combine for training",
        required=True,
    )
    parser.add_argument(
        "--combine_datasets_val",
        nargs="+",
        help="List of datasets to combine for eval",
        required=True,
    )
    parser.add_argument(
        "--v2",
        default=False,
        type=bool,
        help="whether to use the second version of HC-STVG or not",
    )
    parser.add_argument(
        "--tb_dir", type=str, default="", help="eventual path to tensorboard directory"
    )

    # Training hyper-parameters
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--text_encoder_lr", default=5e-5, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr_drop", default=10, type=int)
    parser.add_argument(
        "--epoch_chunks",
        default=-1,
        type=int,
        help="If greater than 0, will split the training set into chunks and validate/checkpoint after each chunk",
    )
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    parser.add_argument(
        "--eval_skip",
        default=1,
        type=int,
        help='do evaluation every "eval_skip" epochs',
    )

    parser.add_argument(
        "--schedule",
        default="linear_with_warmup",
        type=str,
        choices=("step", "multistep", "linear_with_warmup",
                 "all_linear_with_warmup"),
    )
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument(
        "--fraction_warmup_steps",
        default=0.01,
        type=float,
        help="Fraction of total number of steps",
    )

    # Model parameters
    parser.add_argument(
        "--freeze_text_encoder",
        default=True,
        type=bool,
        help="Whether to freeze the weights of the text encoder",
    )
    parser.add_argument(
        "--freeze_backbone",
        default=True,
        type=bool,
        help="Whether to freeze the weights of the visual encoder",
    )
    parser.add_argument(
        "--text_encoder_type",
        default="roberta-base",
        choices=("roberta-base", "distilroberta-base", "roberta-large"),
    )

    # Backbone
    parser.add_argument(
        "--backbone",
        default="resnet101",
        type=str,
        choices=['resnet101', 'resnet50'],
        help="Name of the convolutional backbone to use such as resnet50 resnet101 timm_tf_efficientnet_b3_ns",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--backbone_multi_scale",
        default=True,
        type=bool
    )

    # Transformer
    parser.add_argument(
        "--layers",
        default=6,
        type=int,
        help="Number of reftrans layers",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=256,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--spatial_decoder_hidden_dim",
        default=64,
        type=int
    )
    parser.add_argument(
        "--temporal_window_width",
        default=[6, 12, 24, 36, 48, 72, 96],
        type=list
    )
    parser.add_argument(
        "--temporal_score_thres",
        default=0.3,
        type=float
    )
    parser.add_argument(
        "--temporal_valid_thres",
        default=0.8,
        type=float
    )
    parser.add_argument(
        "--temporal_decoder_type",
        default='anchor',
        type=str,
        choices=['anchor', 'regression']
    )

    # Loss
    parser.add_argument(
        "--sted",
        default=True,
        help="whether to use start end KL loss",
    )

    # Loss coefficients
    parser.add_argument("--loss_spatial_hm", default=1, type=float)
    parser.add_argument("--loss_spatial_wh", default=0.1, type=float)
    parser.add_argument("--loss_spatial_map", default=0.1, type=float)
    parser.add_argument("--loss_temporal_cls", default=1, type=float)
    parser.add_argument("--loss_temporal_align", default=0.01, type=float)
    parser.add_argument("--loss_temporal_score", default=1, type=float)
    parser.add_argument("--loss_temporal_reg", default=0.5, type=float)
    parser.add_argument("--loss_temporal_iou", default=0.5, type=float)

    # Run specific
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to run evaluation on val or test set",
    )
    parser.add_argument(
        "--output-dir", default="OUTPUT", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--load",
        default="",
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true",
                        help="Only run evaluation")
    parser.add_argument("--num_workers", default=3, type=int)

    # Distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    # Video parameters
    parser.add_argument(
        "--resolution", type=int, default=224, help="spatial resolution of the images"
    )
    parser.add_argument(
        "--video_max_len",
        type=int,
        default=100,
        help="maximum number of frames for a video",
    )
    parser.add_argument(
        "--video_max_len_train",
        type=int,
        default=100,
        help="maximum number of frames used by the model - may it differ from video_max_len, the model ensembles start-end probability predictions at eval time",
    )
    parser.add_argument("--stride", type=int, default=0,
                        help="temporal stride k")
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="number of frames per second extracted from videos",
    )
    parser.add_argument(
        "--tmp_crop",
        action="store_true",
        help="whether to use random temporal cropping during training",
    )
    return parser


def main(args):
    dist.init_distributed_mode(args)
    if args.dataset_config is not None:
        # https://stackoverflow.com/a/16878364
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    # fix the seed for reproducibility
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.set_deterministic(True)
    # torch.use_deterministic_algorithms(True)

    # Build the model
    model, criterion, weight_dict = build_model(args)
    model.to(device)

    model_ema = deepcopy(model) if args.ema else None
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # Set up optimizers
    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "video_encoder" not in n and "text_encoder" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "video_encoder" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "text_encoder" in n and p.requires_grad
            ],
            "lr": args.text_encoder_lr,
        },
    ]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise RuntimeError(f"Unsupported optimizer {args.optimizer}")

    # Train dataset
    if len(args.combine_datasets) == 0 and not args.eval:
        raise RuntimeError("Please provide at least one training dataset")

    dataset_train, sampler_train, data_loader_train = None, None, None
    if not args.eval:
        dataset_train = ConcatDataset(
            [
                build_dataset(name, image_set="train", args=args)
                for name in args.combine_datasets
            ]
        )

        # To handle very big datasets, we chunk it into smaller parts.
        if args.epoch_chunks > 0:
            print(
                "Splitting the training set into {args.epoch_chunks} of size approximately "
                f" {len(dataset_train) // args.epoch_chunks}"
            )
            chunks = torch.chunk(torch.arange(
                len(dataset_train)), args.epoch_chunks)
            datasets = [
                torch.utils.data.Subset(dataset_train, chunk.tolist())
                for chunk in chunks
            ]
            if args.distributed:
                samplers_train = [DistributedSampler(ds) for ds in datasets]
            else:
                samplers_train = [
                    torch.utils.data.RandomSampler(ds) for ds in datasets]

            batch_samplers_train = [
                torch.utils.data.BatchSampler(
                    sampler_train, args.batch_size, drop_last=True
                )
                for sampler_train in samplers_train
            ]
            assert len(batch_samplers_train) == len(datasets)
            data_loaders_train = [
                DataLoader(
                    ds,
                    batch_sampler=batch_sampler_train,
                    collate_fn=partial(utils.video_collate_fn, False, 0),
                    num_workers=args.num_workers,
                )
                for ds, batch_sampler_train in zip(datasets, batch_samplers_train)
            ]
        else:
            if args.distributed:
                sampler_train = DistributedSampler(dataset_train, shuffle=True)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)

            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True
            )
            data_loader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=partial(utils.video_collate_fn, False, 0),
                num_workers=args.num_workers,
            )

    if len(args.combine_datasets_val) == 0:
        raise RuntimeError("Please provide at least one validation dataset")

    Val_all = namedtuple(
        typename="val_data",
        field_names=["dataset_name", "dataloader", "evaluator_list"],
    )
    val_tuples = []
    for dset_name in args.combine_datasets_val:
        dset = build_dataset(dset_name, image_set="val", args=args)
        sampler = (
            DistributedSampler(dset, shuffle=False)
            if args.distributed
            else torch.utils.data.SequentialSampler(dset)
        )
        dataloader = DataLoader(
            dset,
            math.ceil(
                (args.batch_size * args.video_max_len_train) / args.video_max_len
            ),
            sampler=sampler,
            drop_last=False,
            collate_fn=partial(
                utils.video_collate_fn,
                False,
                args.video_max_len_train
                if args.video_max_len_train != args.video_max_len
                else 0,
            ),
            num_workers=args.num_workers,
        )
        val_tuples.append(
            Val_all(dataset_name=dset_name,
                    dataloader=dataloader, evaluator_list=None)
        )

    # Used for resuming training from the checkpoint of a model. Used when training times-out or is pre-empted.
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.eval and "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
        if args.ema:
            if "model_ema" not in checkpoint:
                print(
                    "WARNING: ema model not found in checkpoint, resetting to current model"
                )
                model_ema = deepcopy(model_without_ddp)
            else:
                model_ema.load_state_dict(checkpoint["model_ema"])

    def build_evaluator_list(dataset_name):
        """Helper function to build the list of evaluators for a given dataset"""
        evaluator_list = []
        if "vidstg" in dataset_name:
            evaluator_list.append(
                VidSTGEvaluator(
                    args.vidstg_ann_path,
                    "test",
                    iou_thresholds=[0.3, 0.5],
                    fps=args.fps,
                    video_max_len=args.video_max_len,
                    save_pred=args.test,
                    tmp_loc=args.sted,
                )
            )
        if "hcstvg" in dataset_name:
            evaluator_list.append(
                HCSTVGEvaluator(
                    args.hcstvg_ann_path,
                    "val",  # no val set in v1, no test set in v2
                    iou_thresholds=[0.3, 0.5],
                    fps=args.fps,
                    video_max_len=args.video_max_len,
                    v2=args.v2,
                    save_pred=args.test,
                    tmp_loc=args.sted,
                )
            )
        return evaluator_list

    if args.tb_dir and dist.is_main_process():
        writer = SummaryWriter(args.tb_dir)
    else:
        writer = None

    # Runs only evaluation, by default on the validation set unless --test is passed.
    if args.eval:
        test_stats = {}
        test_model = model_ema if model_ema is not None else model
        for i, item in enumerate(val_tuples):
            evaluator_list = build_evaluator_list(item.dataset_name)
            postprocessors = build_postprocessors(args, item.dataset_name)
            item = item._replace(evaluator_list=evaluator_list)
            print(f"Evaluating {item.dataset_name}")
            curr_test_stats = evaluate(
                model=test_model,
                criterion=criterion,
                postprocessors=postprocessors,
                weight_dict=weight_dict,
                data_loader=item.dataloader,
                evaluator_list=item.evaluator_list,
                device=device,
                args=args,
            )
            test_stats.update(
                {item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()}
            )

        log_stats = {
            **{f"test_{k}": v for k, v in test_stats.items()},
            "n_parameters": n_parameters,
        }
        if args.output_dir and dist.is_main_process():
            json.dump(
                log_stats, open(os.path.join(
                    args.output_dir, "log_stats.json"), "w")
            )
        return

    # Runs training and evaluates after every --eval_skip epochs
    print("Start training")
    postprocessors = build_postprocessors(args)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.epoch_chunks > 0:
            sampler_train = samplers_train[epoch % len(samplers_train)]
            data_loader_train = data_loaders_train[epoch % len(
                data_loaders_train)]
            print(
                f"Starting epoch {epoch // len(data_loaders_train)}, sub_epoch {epoch % len(data_loaders_train)}"
            )
        else:
            print(f"Starting epoch {epoch}")
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            weight_dict=weight_dict,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            max_norm=args.clip_max_norm,
            model_ema=model_ema,
            writer=writer
        )
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 2 epochs
            if (
                (epoch + 1) % args.lr_drop == 0
                or (epoch + 1) % 2 == 0
                or (args.combine_datasets_val[0] == "vidstg")
            ):
                checkpoint_paths.append(
                    output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                dist.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "model_ema": model_ema.state_dict() if args.ema else None,
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        if (epoch + 1) % args.eval_skip == 0:
            test_stats = {}
            test_model = model_ema if model_ema is not None else model
            for i, item in enumerate(val_tuples):
                evaluator_list = build_evaluator_list(item.dataset_name)
                item = item._replace(evaluator_list=evaluator_list)
                postprocessors = build_postprocessors(args, item.dataset_name)
                print(f"Evaluating {item.dataset_name}")
                curr_test_stats = evaluate(
                    model=test_model,
                    criterion=criterion,
                    postprocessors=postprocessors,
                    weight_dict=weight_dict,
                    data_loader=item.dataloader,
                    evaluator_list=item.evaluator_list,
                    device=device,
                    args=args,
                )
                test_stats.update(
                    {item.dataset_name + "_" + k: v for k,
                        v in curr_test_stats.items()}
                )
        else:
            test_stats = {}

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and dist.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "TubeDETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
