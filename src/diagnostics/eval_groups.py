"""Experiment 2: worst-group evaluation of any stored synthetic set.

Mirrors distillation/eval.py (same optimizer, schedule, augmentation and
model-selection rule) but additionally breaks the test set down by subgroup.
Model selection is on *average* accuracy, exactly as in eval.py -- selecting on
worst-group accuracy would assume group labels at training time, which the
distillation pipeline never has.

Works on any job_tag, so the same command evaluates the distilled set, the
centroid baseline and the random-real baseline.
"""

import glob
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tap import Tap
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from augmentation import AugBasic
from data.dataloaders import get_dataset
from models import get_fc, get_model


# kept local instead of config/ because this is a throwaway diagnostic
class GroupEvalCfg(Tap):
    dataset: str
    model: str
    eval_model: str

    job_tag: str = "distillation"
    data_root: str = "data/datasets"
    num_workers: int = 16

    real_batch_size: int = 100
    real_res: int = 256
    crop_res: int = 224
    train_crop_mode: str = "random"

    num_eval: int = 5
    eval_epochs: int = 1000
    eval_it: int = -1  # -1 -> only evaluate at the final epoch, as in eval.py
    patience: int = 5

    out_dir: str = "logged_files/diagnostics"


def get_groups(test_dataset) -> Tuple[Tensor, List[str]]:
    """group = label * num_attributes + spurious attribute."""
    # spawrious computes targets lazily, and the same pass fills in `backgrounds`
    if getattr(test_dataset, "targets", None) is None:
        test_dataset.targets = test_dataset.get_targets()

    if not hasattr(test_dataset, "backgrounds"):
        raise NotImplementedError(
            f"{type(test_dataset).__name__} does not expose subgroup labels. "
            "Add a `backgrounds` (or equivalent) attribute first."
        )

    y = test_dataset.targets.long()
    b = test_dataset.backgrounds.long()
    num_attributes = int(b.max().item()) + 1
    groups = y * num_attributes + b

    attr_names = getattr(
        test_dataset, "attribute_names", [f"attr {a}" for a in range(num_attributes)]
    )
    names = [f"{cls} on {attr}" for cls in test_dataset.class_names for attr in attr_names]
    return groups, names


@torch.no_grad()
def predict(model: nn.Module, fc: nn.Module, loader: DataLoader, normalize) -> Tensor:
    """Predictions in dataset order (loader must be shuffle=False)."""
    preds = []
    for x, _ in tqdm(loader, desc="Evaluating", leave=False):
        x = x.cuda(non_blocking=True)
        x = normalize(x)
        with autocast():
            out = fc(model(x))
        preds.append(out.argmax(dim=1).cpu())
    return torch.cat(preds)


def group_accuracies(preds: Tensor, labels: Tensor, groups: Tensor, num_groups: int) -> List[float]:
    """nan for groups with no test samples -- spawrious leaves many combinations empty."""
    correct = (preds == labels).float()
    accs = []
    for g in range(num_groups):
        mask = groups == g
        accs.append(correct[mask].mean().item() if mask.any() else float("nan"))
    return accs


def run_once(cfg: GroupEvalCfg, syn_loader, test_loader, model, num_feats,
             train_dataset, labels, groups, group_names, seed: int) -> Tuple[float, List[float]]:

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device_count = max(1, torch.cuda.device_count())

    fc = get_fc(
        num_feats=num_feats,
        num_classes=train_dataset.num_classes,
        distributed=torch.cuda.device_count() > 1,
    )
    optimizer = torch.optim.Adam(
        list(fc.parameters()),
        0.001 * (syn_loader.batch_size / device_count) / 256.0,  # linear scaling rule
        weight_decay=0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.eval_epochs, eta_min=0
    )
    scaler = GradScaler()
    augmentor = AugBasic(crop_res=cfg.crop_res).cuda()

    best_avg = -1.0
    best_groups = None
    patience_counter = 0

    for epoch in tqdm(range(cfg.eval_epochs), desc="Training Linear Head"):

        for x, y in syn_loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with autocast():
                with torch.no_grad():
                    x = augmentor(x)
                    x = train_dataset.normalize(x)
                    z = model(x)
                loss = nn.functional.cross_entropy(fc(z), y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        is_eval_epoch = (
            cfg.eval_it != -1 and epoch % cfg.eval_it == 0
        ) or epoch == cfg.eval_epochs - 1
        if not is_eval_epoch:
            continue

        preds = predict(model, fc, test_loader, train_dataset.normalize)
        avg = (preds == labels).float().mean().item()

        if avg <= best_avg:
            patience_counter += 1
            if patience_counter == cfg.patience:
                print("Out of patience! Stopping training.")
                break
        else:
            patience_counter = 0
            best_avg = avg
            best_groups = group_accuracies(preds, labels, groups, len(group_names))

    return best_avg, best_groups


def main(cfg: GroupEvalCfg):

    model_dir = os.path.join("logged_files", cfg.job_tag, cfg.dataset, cfg.model)
    syn_files = sorted(glob.glob(os.path.join(model_dir, "**", "data.pth"), recursive=True))
    if len(syn_files) == 0:
        raise FileNotFoundError(f"No data.pth found under {model_dir}")
    if len(syn_files) > 1:
        print("Warning: multiple syn sets found. Using the first one.")
    print(f"Loaded synthetic set from {syn_files[0]}")

    train_dataset, test_dataset = get_dataset(
        name=cfg.dataset,
        res=cfg.real_res,
        crop_res=cfg.crop_res,
        train_crop_mode=cfg.train_crop_mode,
        data_root=cfg.data_root,
    )

    groups, group_names = get_groups(test_dataset)
    labels = test_dataset.targets.long()

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,  # required: predictions are aligned to dataset order
        num_workers=cfg.num_workers,
        batch_size=cfg.real_batch_size,
    )

    syn_set = torch.load(syn_files[0], weights_only=False)
    syn_images = syn_set["images"].cuda().detach().clone()
    syn_labels = syn_set["labels"].cuda().detach().clone()
    print(f"Synthetic set: {len(syn_images)} images, {len(syn_labels.unique())} classes")

    syn_loader = DataLoader(
        TensorDataset(syn_images, syn_labels),
        batch_size=min(100, len(syn_images)),
        shuffle=True,
    )

    eval_model, num_feats = get_model(
        cfg.eval_model, distributed=torch.cuda.device_count() > 1
    )

    avgs, per_group, worsts = [], [], []
    for i in range(cfg.num_eval):
        print(f"\n--- run {i + 1}/{cfg.num_eval} ---")
        avg, g_accs = run_once(
            cfg, syn_loader, test_loader, eval_model, num_feats,
            train_dataset, labels, groups, group_names, seed=3407 + i,
        )
        avgs.append(avg)
        per_group.append(g_accs)
        worsts.append(np.nanmin(g_accs))

    per_group = np.array(per_group)

    print("\n" + "=" * 70)
    print(f"job_tag={cfg.job_tag} dataset={cfg.dataset} "
          f"distill={cfg.model} eval={cfg.eval_model}")
    print("=" * 70)
    for i, name in enumerate(group_names):
        n = int((groups == i).sum().item())
        if n == 0:
            print(f"  {name:<28} n=0      (empty -- excluded)")
            continue
        print(f"  {name:<28} n={n:<6} "
              f"{np.nanmean(per_group[:, i]) * 100:6.2f} +- {np.nanstd(per_group[:, i]) * 100:.2f}")
    print("-" * 70)
    print(f"  {'AVERAGE':<28} {'':<8}"
          f"{np.mean(avgs) * 100:6.2f} +- {np.std(avgs) * 100:.2f}")
    print(f"  {'WORST GROUP':<28} {'':<8}"
          f"{np.mean(worsts) * 100:6.2f} +- {np.std(worsts) * 100:.2f}")
    print("=" * 70)

    save_dict = {
        "job_tag": cfg.job_tag,
        "avg_mean": float(np.mean(avgs)),
        "avg_std": float(np.std(avgs)),
        "worst_mean": float(np.mean(worsts)),
        "worst_std": float(np.std(worsts)),
        "group_names": group_names,
        "group_mean": np.nanmean(per_group, axis=0).tolist(),
        "group_std": np.nanstd(per_group, axis=0).tolist(),
    }
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_file = os.path.join(
        cfg.out_dir,
        f"groups_{cfg.job_tag}_{cfg.dataset}_{cfg.model}_{cfg.eval_model}.pth",
    )
    torch.save(save_dict, out_file)
    print(f"Saved to {out_file}")


if __name__ == "__main__":

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(3407)
    random.seed(3407)
    np.random.seed(3407)

    main(GroupEvalCfg(explicit_bool=True).parse_args())
