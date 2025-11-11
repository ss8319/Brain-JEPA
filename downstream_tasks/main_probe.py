# This source code is licensed under the Apache License, Version 2.0
#
# References:
# deit: https://github.com/facebookresearch/deit/blob/main/main.py
# capi: https://github.com/facebookresearch/capi/blob/main/eval_classification.py

import argparse
import datetime
import json
import math
import os
import sys
import time
from collections import defaultdict
from functools import partial
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence, Optional, Union, List, Dict, Tuple

# Add project root to path for imports when running script directly
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from timm.utils import accuracy

import downstream_tasks.models_probe as models_probe
import downstream_tasks.util.misc as misc
from downstream_tasks.models_vit import VisionTransformer
from src.datasets.hca_sex_datasets import make_hca_sex
from src.datasets.hcya_sex_datasets import make_hcya_sex

DATA_FN_DICT = {
    "hca_sex": make_hca_sex,
    "hcya_sex": make_hcya_sex,
}

DEFAULT_CONFIG = Path(__file__).parent / "config/default_probe.yaml"

MODELS_DICT = {"VisionTransformer": VisionTransformer}


def warmup_cosine_schedule(base_value, final_value, total_iters, warmup_iters):
    """Generate warmup + cosine annealing LR schedule as a list
       Simplified version of the original code in flat_mae.utils.WarmupThenCosine"""
    schedule = []
    for i in range(total_iters):
        if i < warmup_iters:
            lr = base_value * (i / warmup_iters)
        else:
            progress = (i - warmup_iters) / (total_iters - warmup_iters)
            lr = final_value + (base_value - final_value) * 0.5 * (1 + math.cos(math.pi * progress))
        schedule.append(lr)
    return schedule


def _get_arg(args, key, default=None):
    """Helper to get optional args values (works with OmegaConf DictConfig and Config/Namespace objects)"""
    # OmegaConf DictConfig has .get() method
    if hasattr(args, 'get') and callable(getattr(args, 'get', None)):
        return args.get(key, default)
    # Config/Namespace objects use getattr
    return getattr(args, key, default)


def main(args):
    # setup
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    is_master = global_rank == 0
    world_size = misc.get_world_size()
    device = torch.device(_get_arg(args, 'device', 'cuda'))
    seed = _get_arg(args, 'seed', 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.name and not args.output_dir.endswith(args.name):
        args.output_dir = f"{args.output_dir}/{args.name}"
    output_dir = Path(args.output_dir)

    if is_master:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_cfg_path = output_dir / "config.yaml"
        if out_cfg_path.exists():
            prev_cfg = OmegaConf.load(out_cfg_path)
            assert args == prev_cfg, "current config doesn't match previous config"
        else:
            OmegaConf.save(args, out_cfg_path)

        if args.wandb:
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.name,
                notes=args.notes,
                config=OmegaConf.to_container(args),
            )

    misc.setup_for_distributed(is_master)

    print("probe eval Brain-JEPA")
    print(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"cwd: {Path.cwd()}")
    print("config:", OmegaConf.to_yaml(args), sep="\n")

    # data loaders
    train_loader, test_loader, eval_loaders = create_data_loaders(args)

    # backbone embedding model
    if _get_arg(args, "load_path") or _get_arg(args, "pretrain_ckpt"):
        load_path = _get_arg(args, "load_path") or _get_arg(args, "pretrain_ckpt")
        print(f"creating backbone model from checkpoint: {load_path}")
        checkpoint = torch.load(load_path, map_location="cpu")
        # Load Brain-JEPA checkpoint format (target_encoder)
        if "target_encoder" in checkpoint:
            state_dict = checkpoint["target_encoder"]
        else:
            state_dict = checkpoint
        
        # Create model using args
        backbone = VisionTransformer(
            args,
            model_name=_get_arg(args, "model_name", "vit_base"),
            attn_mode=_get_arg(args, "attn_mode", "normal"),
            device=device,
            add_w=_get_arg(args, "add_w", False)
        )
        
        # Remove keys that don't exist in encoder (head, fc_norm)
        state_dict = {k: v for k, v in state_dict.items() 
                     if not k.startswith("head.") and not k.startswith("fc_norm.")}
        backbone.encoder.load_state_dict(state_dict, strict=False)
    else:
        print(f"creating backbone model from scratch")
        backbone = VisionTransformer(
            args,
            model_name=_get_arg(args, "model_name", "vit_base"),
            attn_mode=_get_arg(args, "attn_mode", "normal"),
            device=device,
            add_w=_get_arg(args, "add_w", False)
        )

    if not _get_arg(args, "finetune", False):
        print("freezing backbone model")
        backbone.encoder.requires_grad_(False)
        backbone_param_groups = []
    else:
        print("finetuning backbone model")
        backbone_param_groups = [{"params": [p for p in backbone.encoder.parameters() if p.requires_grad]}]

    backbone.to(device)
    embedding_shapes = get_embedding_shapes(backbone, args.representations, train_loader, device)
    print(f"embedding feature shapes:\n{embedding_shapes}")

    print("initializing sweep of classifier heads")
    classifiers, classifier_param_groups = make_classifiers(args, embedding_shapes)

    model = models_probe.ClassificationWrapper(backbone, classifiers)
    model.to(device)
    print(f"model:\n{model}")
    num_params = sum(p.numel() for p in model.parameters())
    num_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"num params (train): {num_params / 1e6:.1f}M ({num_params_train / 1e6:.1f}M)")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # todo: compile?

    # optimizer
    batch_size = _get_arg(args, "batch_size", 32)
    accum_iter = _get_arg(args, "accum_iter", 1)
    total_batch_size = batch_size * accum_iter * world_size
    print(
        f"total batch size: {total_batch_size} = "
        f"{batch_size} bs per gpu x {accum_iter} accum x {world_size} gpus"
    )

    lr = _get_arg(args, "lr")
    if not lr:
        base_lr = _get_arg(args, "base_lr", 0.001)
        lr = base_lr * total_batch_size / 256
        args.lr = lr  # Set it for later use
        print(f"lr: {lr:.2e} = {base_lr:.2e} x {total_batch_size} / 256")
    else:
        args.lr = lr
        print(f"lr: {lr:.2e}")

    param_groups = backbone_param_groups + classifier_param_groups
    # Set initial LR and weight decay for all param groups
    # Note: lr_multiplier and wd_multiplier are set by make_classifiers() for classifier heads
    weight_decay = _get_arg(args, "weight_decay", 0.05)
    for pg in param_groups:
        lr_multiplier = pg.get("lr_multiplier", 1.0)
        wd_multiplier = pg.get("wd_multiplier", 1.0)
        pg.setdefault("lr", lr * lr_multiplier)
        pg.setdefault("weight_decay", weight_decay * wd_multiplier)
    # cast or else it corrupts the checkpoint
    betas = tuple(args.betas) if args.betas is not None else (0.9, 0.999)
    optimizer = torch.optim.AdamW(param_groups, betas=betas)

    epoch_num_batches = len(train_loader)
    accum_iter = _get_arg(args, "accum_iter", 1)
    steps_per_epoch = math.ceil(epoch_num_batches / accum_iter)
    epochs = _get_arg(args, "epochs", 50)
    warmup_epochs = _get_arg(args, "warmup_epochs", 5)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    min_lr = _get_arg(args, "min_lr", 1e-6)
    lr_schedule = warmup_cosine_schedule(
        base_value=lr,
        final_value=min_lr,
        total_iters=total_steps,
        warmup_iters=warmup_steps,
    )
    print(f"full schedule: epochs = {epochs} (steps = {total_steps})")
    print(f"warmup: epochs = {warmup_epochs} (steps = {warmup_steps})")

    # loss scaling not needed for bfloat16 (according to timm)
    if args.amp and args.amp_dtype != "bfloat16":
        loss_scaler = torch.GradScaler(device.type)
    else:
        loss_scaler = None
    # Debug: count trainable vs frozen params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"debug: trainable_params={trainable_params}, frozen_params={frozen_params}")

    # load checkpoint/resume training
    misc.load_model(args, model_without_ddp, optimizer, loss_scaler)

    # training loss
    if args.task == "classification":
        criterion = nn.CrossEntropyLoss(reduction="none")
    elif args.task == "regression":
        if args.loss_function == "mse":
            criterion = nn.MSELoss(reduction="none")
        elif args.loss_function == "cosine":
            # predictions are shape [batch_size, num_targets, num_classifiers]
            # normalize over the target dimension
            criterion = models_probe.CosineLoss(dim=-2, reduction="none")
        else:
            raise ValueError(f"unknown loss_function {args.loss_function}")
    else:
        raise ValueError(f"unknown task {args.task}")

    # watch middle classifiers on first epoch
    # for later epochs will update to the best
    # a classifier "key" is a tuple of (feature_source, (lr_scale, weight_decay))
    mid_lr_scale = args.lr_scale_grid[len(args.lr_scale_grid) // 2]
    mid_weight_decay = args.weight_decay_grid[len(args.weight_decay_grid) // 2]
    log_classifier_keys = [
        (feature_source, (mid_lr_scale, mid_weight_decay))
        for feature_source in args.representations
    ]

    print(f"start training for {args.epochs} epochs")
    start_time = time.monotonic()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(train_loader, "sampler"):
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            args,
            model,
            criterion,
            train_loader,
            optimizer,
            loss_scaler,
            lr_schedule,
            epoch,
            device,
            log_classifier_keys=log_classifier_keys,
        )

        eval_stats = {}
        for name, loader in eval_loaders.items():
            stats = evaluate(
                args,
                model,
                criterion,
                loader,
                epoch,
                device,
                eval_name=name,
                log_classifier_keys=log_classifier_keys,
            )
            eval_stats.update(stats)

        merged_stats = {"epoch": epoch, **train_stats, **eval_stats}
        if is_master:
            with (output_dir / "log.json").open("a") as f:
                print(json.dumps(merged_stats), file=f)

        best_scores, best_hparams = get_best_classifiers(args, model_without_ddp, eval_stats)

        # log the best classifiers
        log_classifier_keys = [
            (feature_source, hparam) for feature_source, hparam in best_hparams.items()
        ]

        print(f"Epoch: [{epoch}] Best validation scores:\n{json.dumps(best_scores)}")
        print(f"Epoch: [{epoch}] Best validation hparams:\n{json.dumps(best_hparams)}")

        misc.save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler)

    best_classifiers = {
        (feature_source, hparam): classifiers[feature_source, hparam]
        for feature_source, hparam in best_hparams.items()
    }
    best_model = models_probe.ClassificationWrapper(backbone, best_classifiers)

    if test_loader is not None:
        print("Evaluating best models on test set")

        if args.distributed:
            best_model = torch.nn.parallel.DistributedDataParallel(
                best_model, device_ids=[args.gpu]
            )

        test_stats = evaluate(
            args,
            best_model,
            criterion,
            test_loader,
            epoch,
            device,
            _get_arg(args, "test_dataset", "test"),
            log_classifier_keys=log_classifier_keys,
        )
        print(f"Best models test stats:\n{json.dumps(test_stats)}")

        if is_master:
            with (output_dir / "test_log.json").open("a") as f:
                print(json.dumps(test_stats), file=f)

    if is_master:
        best_checkpoint_path = output_dir / "checkpoint-best.pth"
        to_save = {
            "model": best_model.state_dict(),
            "epoch": epoch,
            "hparams": best_hparams,
            "scores": best_scores,
            "test_stats": test_stats,
            "args": OmegaConf.to_container(args),
        }
        torch.save(to_save, best_checkpoint_path)

    if args.distributed:
        torch.distributed.destroy_process_group()

    total_time = time.monotonic() - start_time
    print(f"done! training time: {datetime.timedelta(seconds=int(total_time))}")


def probe_collate_fn(batch):
    """
    Convert Brain-JEPA's (samples, targets) tuple format to probe's dict format.
    Brain-JEPA returns: (samples, targets) where samples is [B, C, H, W]
    Probe expects: {"image": tensor, "target": tensor, "img_mask": optional}
    """
    samples, targets = zip(*batch)
    samples = torch.stack(samples, dim=0)
    targets = torch.stack(targets, dim=0) if isinstance(targets[0], torch.Tensor) else torch.tensor(targets)
    return {
        "image": samples,
        "target": targets,
        # img_mask is not used by Brain-JEPA, set to None
        "img_mask": None
    }


def create_data_loaders(args: DictConfig):
    """
    Create data loaders using Brain-JEPA's dataset functions (make_hca_sex or make_hcya_sex).
    Returns loaders in probe's expected format.
    """
    # Determine which dataset function to use
    data_make_fn = _get_arg(args, "data_make_fn", "hca_sex")
    if data_make_fn not in DATA_FN_DICT:
        raise ValueError(f"Unknown data_make_fn: {data_make_fn}. Must be one of {list(DATA_FN_DICT.keys())}")
    
    data_fn = DATA_FN_DICT[data_make_fn]
    
    # Use Brain-JEPA's data loading function
    data_loader_train, data_loader_val, data_loader_test, train_dataset, valid_dataset, test_dataset = data_fn(
        batch_size=args.batch_size,
        collator=None,  # We'll use probe_collate_fn instead
        pin_mem=True,
        num_workers=args.num_workers,
        drop_last=False,  # Don't drop last for probe evaluation
        processed_dir=_get_arg(args, "data_path", "data"),
        use_normalization=_get_arg(args, "use_normalization", False),
        label_normalization=_get_arg(args, "label_normalization", False),
        downsample=_get_arg(args, "downsample", False),
        make_constant=_get_arg(args, "make_constant", False),
    )
    
    # Wrap loaders to convert (samples, targets) -> {"image": samples, "target": targets}
    # We need to create new DataLoaders with the probe_collate_fn
    def wrap_loader(loader, dataset):
        if loader is None:
            return None
        return DataLoader(
            dataset,
            batch_size=loader.batch_size,
            sampler=loader.sampler,
            shuffle=False,  # Sampler handles shuffling
            num_workers=loader.num_workers,
            collate_fn=probe_collate_fn,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
        )
    
    train_loader = wrap_loader(data_loader_train, train_dataset)
    test_loader = wrap_loader(data_loader_test, test_dataset) if data_loader_test is not None else None
    
    # Create eval_loaders dict
    eval_loaders = {}
    if data_loader_val is not None:
        val_dataset_name = _get_arg(args, "val_dataset", "valid")
        eval_loaders[val_dataset_name] = wrap_loader(data_loader_val, valid_dataset)
    
    return train_loader, test_loader, eval_loaders


@torch.no_grad()
def get_embedding_shapes(
    backbone: nn.Module,
    representations: List[str],
    loader: Iterable,
    device: torch.device,
):
    print("running backbone on example batch to get embedding shapes")
    example_batch = next(iter(loader))
    # Move batch to device
    example_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in example_batch.items()}
    images = example_batch["image"]
    # Use only first sample to avoid OOM during shape inference
    images = images[:1]
    img_mask = example_batch.get("img_mask")
    if img_mask is not None:
        img_mask = img_mask[:1]

    cls_token, object_tokens, patch_tokens = backbone.forward_embedding(images)
    backbone_out = models_probe.pool_representations(
        cls_token, object_tokens, patch_tokens, representations
    )

    embedding_shapes = {k: tuple(v.shape[1:]) for k, v in backbone_out.items()}
    return embedding_shapes


def make_classifiers(args: DictConfig, embedding_shapes: Dict[str, Tuple[int, ...]]):
    # create sweep of classifier heads with varying input features,
    # lr scales, weight decays.
    all_classifiers = {}
    param_groups = {}

    base_weight_decay = args.weight_decay

    for feature_source in args.representations:
        embed_shape = embedding_shapes[feature_source]
        assert len(embed_shape) in {1, 2}

        if len(embed_shape) == 1:
            clf_fn = partial(models_probe.LinearClassifier, embed_shape[-1], args.num_classes)
        else:
            clf_fn = partial(
                models_probe.AttnPoolClassifier,
                embed_shape[-1],
                args.num_classes,
                embed_dim=_get_arg(args, "attn_pool_embed_dim"),
            )

        for lr_scale, weight_decay in product(args.lr_scale_grid, args.weight_decay_grid):
            clf = clf_fn()
            all_classifiers[(feature_source, (lr_scale, weight_decay))] = clf

            for name, param in clf.named_parameters():
                param_weight_decay = 0.0 if "bias" in name else weight_decay

                if (lr_scale, param_weight_decay) not in param_groups:
                    param_groups[lr_scale, param_weight_decay] = {
                        "params": [],
                        "lr_multiplier": lr_scale,
                        "wd_multiplier": param_weight_decay / base_weight_decay,
                    }

                param_groups[lr_scale, param_weight_decay]["params"].append(param)

    param_groups = list(param_groups.values())
    return all_classifiers, param_groups


def train_one_epoch(
    args: DictConfig,
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_scaler: Optional[torch.GradScaler],
    lr_schedule: Sequence[float],
    epoch: int,
    device: torch.device,
    log_classifier_keys: List[Tuple[str, Tuple[float, float]]],
):
    model.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("grad", misc.SmoothedValue())
    header = f"Train: [{epoch}]"
    log_wandb = _get_arg(args, "wandb", False) and misc.is_main_process()

    epoch_num_batches = len(data_loader)
    accum_iter = _get_arg(args, "accum_iter", 1)
    steps_per_epoch = math.ceil(epoch_num_batches / accum_iter)

    use_cuda = device.type == "cuda"
    # Brain-JEPA doesn't use pre_send_to_cuda_wrapper

    print_freq = _get_arg(args, "print_freq", 20) if not _get_arg(args, "debug", False) else 1
    num_batches = epoch_num_batches if not args.debug else 10

    model_without_ddp = model.module if args.distributed else model
    classifier_keys = model_without_ddp.classifier_keys
    clf_key_to_idx = {key: ii for ii, key in enumerate(classifier_keys)}
    num_classifiers = len(classifier_keys)

    amp_dtype = getattr(torch, args.amp_dtype)

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # Debug: confirm per-group LR values on first batch
        if batch_idx == 0:
            lr_values = sorted({pg.get('lr', None) for pg in optimizer.param_groups})
            print(f"debug: step0 param_group_lrs={lr_values}")
        if use_cuda:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        batch_step = batch_idx + 1
        global_step = epoch * steps_per_epoch + batch_idx // accum_iter
        # Clamp global_step to valid range to prevent IndexError
        global_step = min(global_step, len(lr_schedule) - 1)
        lr = lr_schedule[global_step]
        need_update = batch_step % accum_iter == 0

        if need_update:
            for pg in optimizer.param_groups:
                lr_multiplier = pg.get("lr_multiplier", 1.0)
                pg['lr'] = lr * lr_multiplier

        images = batch["image"]
        img_mask = batch.get("img_mask")
        target = batch["target"]

        # handle single target regression
        # predictions are always shape [batch_size, num_targets, num_classifiers], so
        # need to match second dimension.
        if args.task == "regression" and target.ndim == 1:
            target = target.unsqueeze(-1)

        # expand last dimension of target to match prediction
        # note that the num_classifiers dimension has to go at the end bc this is
        # what nn.CrossEntropyLoss expects.
        expand_shape = target.ndim * (-1,) + (num_classifiers,)
        target = target.unsqueeze(-1).expand(*expand_shape)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            pred = model(images, masks=img_mask)
            # [batch, num_classifiers] or [batch, num_targets, num_classifiers]
            all_loss = criterion(pred, target)
            all_loss = all_loss.reshape(-1, num_classifiers).mean(dim=0)
            loss = all_loss.mean()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        # Backward pass with gradient accumulation
        (loss / accum_iter).backward()
        if need_update:
            if loss_scaler is not None:
                loss_scaler.unscale_(optimizer)
            grad_norm = misc.get_grad_norm_(model.parameters())
            clip_grad = _get_arg(args, "clip_grad")
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            if loss_scaler is not None:
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        else:
            grad_norm = None

        all_loss_values = all_loss.detach().cpu().numpy()
        log_loss_dict = {
            f"loss_{key[0]}": float(all_loss_values[clf_key_to_idx[key]]) for key in log_classifier_keys
        }
        metric_logger.update(loss=loss_value)
        metric_logger.update(**log_loss_dict)
        if need_update:
            metric_logger.update(lr=lr)
            grad_norm_value = grad_norm.item()
            metric_logger.update(grad=grad_norm_value)

        if log_wandb:
            log_stats = {
                "train/loss": loss_value,
                **{f"train/{k}": v for k, v in log_loss_dict.items()},
            }
            if need_update:
                log_stats.update({"train/lr": lr, "train/grad": grad_norm_value})
            wandb.log(log_stats, step=int(1000 * (epoch + batch_step / epoch_num_batches)))

        if use_cuda:
            torch.cuda.synchronize()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {f"train/{k}": float(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def evaluate(
    args: DictConfig,
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    epoch: int,
    device: torch.device,
    eval_name: str,
    log_classifier_keys: List[Tuple[str, Tuple[float, float]]],
):
    """
    log_classifier_keys: list of (feature_source, hparams) keys for the classifiers to log.
    """
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f"Eval ({eval_name}): [{epoch}]"
    log_wandb = _get_arg(args, "wandb", False) and misc.is_main_process()

    epoch_num_batches = len(data_loader)

    use_cuda = device.type == "cuda"
    # Brain-JEPA doesn't use pre_send_to_cuda_wrapper

    print_freq = _get_arg(args, "print_freq", 20) if not _get_arg(args, "debug", False) else 1
    num_batches = epoch_num_batches if not args.debug else 10

    model_without_ddp = model.module if args.distributed else model
    classifier_keys = model_without_ddp.classifier_keys
    clf_key_to_idx = {key: ii for ii, key in enumerate(classifier_keys)}
    num_classifiers = len(classifier_keys)

    all_meters = defaultdict(misc.SmoothedValue)

    amp_dtype = getattr(torch, args.amp_dtype)

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if use_cuda:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        images = batch["image"]
        img_mask = batch.get("img_mask")
        target = batch["target"]

        if args.task == "regression" and target.ndim == 1:
            target = target.unsqueeze(-1)
        expand_shape = target.ndim * (-1,) + (num_classifiers,)
        target = target.unsqueeze(-1).expand(*expand_shape)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            pred = model(images, masks=img_mask)
            # [batch, num_classifiers] or [batch, num_targets, num_classifiers]
            all_loss = criterion(pred, target)
            all_loss = all_loss.reshape(-1, num_classifiers).mean(dim=0)
            loss = all_loss.mean()

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        all_loss_values = all_loss.detach().cpu().numpy()
        if args.task == "classification":
            all_acc1_values = [
                accuracy(pred[:, :, ii], target[:, ii])[0].item() for ii in range(num_classifiers)
            ]

        for ii, key in enumerate(classifier_keys):
            fmt_key = format_clf_key(key)
            all_meters[f"loss_{fmt_key}"].update(all_loss_values[ii])
            if args.task == "classification":
                all_meters[f"acc1_{fmt_key}"].update(all_acc1_values[ii])

        log_metric_dict = {}
        for feature_source, hparam in log_classifier_keys:
            idx = clf_key_to_idx[(feature_source, hparam)]
            log_metric_dict[f"loss_{feature_source}"] = float(all_loss_values[idx])
            if args.task == "classification":
                log_metric_dict[f"acc1_{feature_source}"] = float(all_acc1_values[idx])

        metric_logger.update(**log_metric_dict)

        if use_cuda:
            torch.cuda.synchronize()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    for meter in all_meters.values():
        meter.synchronize_between_processes()
    print(f"Averaged stats ({eval_name}):", metric_logger)
    stats = {f"eval/{eval_name}/{k}": float(meter.global_avg) for k, meter in metric_logger.meters.items()}

    if log_wandb:
        wandb.log(stats, step=1000 * (epoch + 1))

    stats.update({f"eval/{eval_name}/{k}": float(meter.global_avg) for k, meter in all_meters.items()})
    return stats


def format_clf_key(key: tuple[str, tuple[float, float]]) -> str:
    feature_source, (lr, weight_decay) = key
    return f"{feature_source}_{lr:.1e}_{weight_decay:.1e}"


def get_best_classifiers(
    args: DictConfig,
    model: models_probe.ClassificationWrapper,
    eval_stats: Dict[str, float],
):
    val_scores = defaultdict(list)
    clf_hparams = defaultdict(list)
    val_dataset_name = _get_arg(args, "val_dataset", "valid")
    prefix = f"eval/{val_dataset_name}"
    for key in model.classifier_keys:
        feature_source, hparam = key
        if args.task == "classification":
            score = eval_stats[f"{prefix}/acc1_{format_clf_key(key)}"]
        else:
            score = 1 - eval_stats[f"{prefix}/loss_{format_clf_key(key)}"]
        val_scores[feature_source].append(score)
        clf_hparams[feature_source].append(hparam)

    best_scores = {}
    best_hparams = {}
    for feature_source in val_scores:
        best_idx = np.argmax(val_scores[feature_source])
        best_score = val_scores[feature_source][best_idx]
        best_hparam = clf_hparams[feature_source][best_idx]
        best_scores[feature_source] = best_score
        best_hparams[feature_source] = best_hparam

    return best_scores, best_hparams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.cfg_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.cfg_path))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
