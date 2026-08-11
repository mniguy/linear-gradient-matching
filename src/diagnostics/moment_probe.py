"""Experiment 1: does the linear-probe CE gradient collapse to a first-moment statistic?

We compare the real gradient d_L/d_[W,b] against three surrogates:
  1. mean-only : every feature z_i is replaced by its class mean mu_{y_i}.
                 Assumption-free. If cos ~= 1, the gradient literally cannot
                 see anything beyond class means.
  2. analytic  : the closed form obtained by assuming a uniform softmax,
                 dL/dW_c = (1/C) * z_bar - (n_c/N) * mu_c.
                 Validates the *reason* the collapse happens.
  3. shuffled  : mean-only with the class means permuted across classes.
                 Negative control -- must be low, otherwise the high-dimensional
                 cosine is uninformative.

Everything is measured at the exact point the distillation objective uses it
(distillation/linear_gm.py:153), for several head init scales.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tap import Tap
from torch import Tensor
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentation import get_augmentor
from data.dataloaders import get_dataset
from models import get_model


# kept local instead of config/ because this is a throwaway diagnostic
class MomentProbeCfg(Tap):
    dataset: str
    model: str

    data_root: str = "data/datasets"
    num_workers: int = 16

    batch_size: int = 0  # 0 -> augs_per_batch * num_classes, matching distillation
    augs_per_batch: int = 10
    num_batches: int = 20
    heads_per_batch: int = 8

    # 0.01 is the default in models/linear_classifier.py
    head_stds: List[float] = [0.01, 0.1, 1.0]

    aug_mode: str = "standard"
    real_res: int = 256
    crop_res: int = 224
    train_crop_mode: str = "random"

    out_dir: str = "logged_files/diagnostics"


def ce_grad(z: Tensor, y: Tensor, W: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Exact gradient of mean cross-entropy w.r.t. the linear head.

    dL/dW = (1/N) (P - Y)^T Z,  dL/db = (1/N) sum_i (p_i - y_i)
    """
    logits = z @ W.t() + b
    p = torch.softmax(logits, dim=1)
    residual = p - F.one_hot(y, num_classes=W.shape[0]).to(p.dtype)
    residual = residual / z.shape[0]
    return residual.t() @ z, residual.sum(dim=0)


def class_means(z: Tensor, y: Tensor, num_classes: int) -> Tuple[Tensor, Tensor]:
    """Returns (mu, counts). Absent classes get mu = 0 and count = 0."""
    d = z.shape[1]
    sums = torch.zeros(num_classes, d, device=z.device, dtype=z.dtype)
    sums.index_add_(0, y, z)
    counts = torch.zeros(num_classes, device=z.device, dtype=z.dtype)
    counts.index_add_(0, y, torch.ones_like(y, dtype=z.dtype))
    mu = sums / counts.clamp(min=1).unsqueeze(1)
    return mu, counts


def flat(gw: Tensor, gb: Tensor) -> Tensor:
    """Matches the concatenation used by the distillation objective."""
    return torch.cat([gw.flatten(), gb.flatten()], dim=0)


def cos(a: Tensor, b: Tensor) -> float:
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def compare(
    ref: Tuple[Tensor, Tensor], other: Tuple[Tensor, Tensor]
) -> Dict[str, float]:
    gw_r, gb_r = ref
    gw_o, gb_o = other

    # the shared (1/C) * z_bar term is identical in every row of dL/dW, so it can
    # inflate the cosine on its own; subtracting the row mean isolates the
    # class-discriminative part, which is what actually drives the linear probe
    centered_r = gw_r - gw_r.mean(dim=0, keepdim=True)
    centered_o = gw_o - gw_o.mean(dim=0, keepdim=True)

    full_r = flat(gw_r, gb_r)
    full_o = flat(gw_o, gb_o)

    return {
        "cos_full": cos(full_r, full_o),
        "cos_class_centered": cos(centered_r, centered_o),
        "residual_norm_ratio": (
            torch.norm(full_r - full_o) / torch.norm(full_r)
        ).item(),
    }


@torch.no_grad()
def main(cfg: MomentProbeCfg):

    train_dataset, _ = get_dataset(
        name=cfg.dataset,
        res=cfg.real_res,
        crop_res=cfg.crop_res,
        train_crop_mode=cfg.train_crop_mode,
        data_root=cfg.data_root,
    )
    num_classes = train_dataset.num_classes

    batch_size = cfg.batch_size or min(1000, cfg.augs_per_batch * num_classes)

    loader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=cfg.num_workers,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
    )

    model, num_feats = get_model(name=cfg.model, distributed=False)
    augmentor = get_augmentor(aug_mode=cfg.aug_mode, crop_res=cfg.crop_res)

    records = []
    loader_iter = iter(loader)

    for _ in tqdm(range(cfg.num_batches), desc="Probing batches"):

        batch = next(loader_iter, None)
        if batch is None:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        x, y = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # same forward path as get_real_grad (linear_gm.py:199-208)
        with autocast(enabled=True):
            x = augmentor(x)
            x = train_dataset.normalize(x)
            z = model(x)

        # gradient math in fp32 so the cosines are not fp16 noise
        z = z.float()

        mu, counts = class_means(z, y, num_classes)
        z_bar = z.mean(dim=0)
        n_over_N = counts / z.shape[0]

        z_meanonly = mu[y]

        # a random non-zero cyclic shift: guaranteed to have no fixed point, unlike
        # randperm, which is the identity half the time when num_classes == 2
        shift = int(torch.randint(1, num_classes, (1,)).item())
        perm = (torch.arange(num_classes, device=z.device) + shift) % num_classes
        z_shuffled = mu[perm][y]

        # closed form under a uniform softmax
        gw_analytic = z_bar.unsqueeze(0) / num_classes - n_over_N.unsqueeze(1) * mu
        gb_analytic = 1.0 / num_classes - n_over_N

        for std in cfg.head_stds:
            W = torch.empty(num_classes, num_feats, device=z.device).normal_(0.0, std)
            b = torch.zeros(num_classes, device=z.device)

            g_real = ce_grad(z, y, W, b)
            g_meanonly = ce_grad(z_meanonly, y, W, b)
            g_shuffled = ce_grad(z_shuffled, y, W, b)

            p = torch.softmax(z @ W.t() + b, dim=1)
            entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=1).mean()

            record = {
                "head_std": std,
                "meanonly": compare(g_real, g_meanonly),
                "analytic": compare(g_real, (gw_analytic, gb_analytic)),
                "shuffled": compare(g_real, g_shuffled),
                "softmax_entropy_ratio": (entropy / np.log(num_classes)).item(),
                "softmax_max_prob": p.max(dim=1).values.mean().item(),
                "logit_std": (z @ W.t() + b).std().item(),
                "feat_norm_mean": z.norm(dim=1).mean().item(),
            }
            records.append(record)

    summary = summarize(records, cfg)
    report(summary, cfg, num_classes)
    return summary


def summarize(records: List[dict], cfg: MomentProbeCfg) -> dict:
    summary = {"config": {"dataset": cfg.dataset, "model": cfg.model}, "by_std": {}}

    for std in cfg.head_stds:
        rows = [r for r in records if r["head_std"] == std]
        entry = {}
        for surrogate in ["meanonly", "analytic", "shuffled"]:
            for key in ["cos_full", "cos_class_centered", "residual_norm_ratio"]:
                vals = [r[surrogate][key] for r in rows]
                entry[f"{surrogate}/{key}"] = [float(np.mean(vals)), float(np.std(vals))]
        for key in [
            "softmax_entropy_ratio",
            "softmax_max_prob",
            "logit_std",
            "feat_norm_mean",
        ]:
            vals = [r[key] for r in rows]
            entry[key] = [float(np.mean(vals)), float(np.std(vals))]
        summary["by_std"][str(std)] = entry

    return summary


def report(summary: dict, cfg: MomentProbeCfg, num_classes: int):

    print("\n" + "=" * 78)
    print(f"B1 probe | dataset={cfg.dataset} model={cfg.model} C={num_classes}")
    print("=" * 78)

    header = (
        f"{'head_std':>9} {'meanonly':>16} {'meanonly_ctr':>16} "
        f"{'analytic':>16} {'shuffled_ctr':>16} {'softmax_H':>10}"
    )
    print(header)

    for std, e in summary["by_std"].items():
        def fmt(key):
            m, s = e[key]
            return f"{m:>9.4f}+-{s:.3f}"

        print(
            f"{std:>9} {fmt('meanonly/cos_full')} "
            f"{fmt('meanonly/cos_class_centered')} "
            f"{fmt('analytic/cos_full')} "
            f"{fmt('shuffled/cos_class_centered')} "
            f"{e['softmax_entropy_ratio'][0]:>10.4f}"
        )

    print("-" * 78)

    default = summary["by_std"][str(0.01)] if "0.01" in summary["by_std"] else None
    if default is None:
        print("VERDICT: head_std=0.01 (the distillation default) was not probed.")
        return

    mo_full = default["meanonly/cos_full"][0]
    mo_ctr = default["meanonly/cos_class_centered"][0]
    # signed, not absolute: with num_classes == 2 the shift is a swap, so a strongly
    # negative value is the expected healthy outcome, not a degenerate one
    sh_ctr = default["shuffled/cos_class_centered"][0]

    if num_classes == 2:
        print(
            "NOTE: num_classes == 2, so the shuffled control is a label swap and "
            "cos_class_centered ~= -1 by construction. Confirm on a multi-class "
            "dataset (e.g. cub2011) before trusting the control."
        )

    if sh_ctr > 0.5:
        print(
            f"INVALID: shuffled control is {sh_ctr:.3f} (>0.5). The cosine is not "
            "discriminative here; do not read the other columns."
        )
    elif mo_full > 0.95 and mo_ctr > 0.90:
        print(
            f"B1 CONFIRMED: cos_full={mo_full:.4f}, cos_class_centered={mo_ctr:.4f}. "
            "The gradient is a function of class means only."
        )
    elif mo_full < 0.80:
        print(
            f"B1 REFUTED: cos_full={mo_full:.4f}. The gradient carries substantial "
            "information beyond first moments. Proposal 1 does not hold as stated."
        )
    else:
        print(
            f"INCONCLUSIVE: cos_full={mo_full:.4f}, cos_class_centered={mo_ctr:.4f}. "
            "Partial collapse -- the claim must be weakened to a quantitative one."
        )
    print("=" * 78 + "\n")


if __name__ == "__main__":

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(3407)
    np.random.seed(3407)

    cfg = MomentProbeCfg(explicit_bool=True).parse_args()

    summary = main(cfg)

    os.makedirs(cfg.out_dir, exist_ok=True)
    out_file = os.path.join(
        cfg.out_dir, f"moment_probe_{cfg.dataset}_{cfg.model}.json"
    )
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {out_file}")
