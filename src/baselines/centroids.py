import os

import kornia
import torch
from torch import Tensor, nn
from tqdm import tqdm

from config import CentroidRealsCfg
from data.dataloaders import (
    BaseRealDataset,
    get_dataset,
)
from models import get_model


@torch.no_grad()
def get_centroid_images(
    labels: Tensor,
    model: nn.Module,
    train_dataset: BaseRealDataset,
) -> Tensor:
    crop = kornia.augmentation.CenterCrop(224)
    real_neighbors = []
    for y in tqdm(labels):
        # kept on cpu: a single class can be thousands of images, and normalizing
        # or cropping the whole stack on gpu at once exhausts vram
        real_images = train_dataset.get_single_class(y.item())

        real_embeddings = []
        for chunk in torch.split(real_images, 100):
            chunk = chunk.cuda()
            chunk = train_dataset.normalize(chunk)
            chunk = crop(chunk)
            real_embeddings.append(model(chunk))
        real_embeddings = torch.cat(real_embeddings)
        mean_embedding = torch.mean(real_embeddings, dim=0, keepdim=True)

        distances = torch.norm(real_embeddings - mean_embedding, dim=1)

        # .item() because distances live on gpu while real_images stays on cpu
        nearest_idx = torch.argmin(distances).item()
        nearest_image = real_images[nearest_idx].clone()

        real_neighbors.append(nearest_image)

    real_neighbors = torch.stack(real_neighbors)

    return real_neighbors


def main(cfg: CentroidRealsCfg):
    save_directory = os.path.join(
        "logged_files", "real_centroids", cfg.dataset, cfg.model, "run"
    )
    save_file = os.path.join(save_directory, "data.pth")
    if os.path.exists(save_file) and cfg.skip_if_exists:
        print("This eval already done.")
        print("Exiting...")
        exit()

    train_dataset, test_dataset = get_dataset(
        name=cfg.dataset,
        res=cfg.real_res,
        crop_res=cfg.crop_res,
        train_crop_mode="center",
        data_root=cfg.data_root,
    )

    eval_model, num_feats = get_model(
        cfg.model, distributed=torch.cuda.device_count() > 1
    )

    labels = torch.cat(
        [
            torch.tensor([c] * 1, dtype=torch.long)
            for c in range(train_dataset.num_classes)
        ],
        dim=0,
    ).cuda()

    real_neighbors = get_centroid_images(
        labels=labels, model=eval_model, train_dataset=train_dataset
    )

    os.makedirs(save_directory, exist_ok=True)

    save_dict = {
        "images": real_neighbors.cpu(),
        "labels": labels.cpu(),
    }
    torch.save(save_dict, os.path.join(save_directory, "data.pth"))


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    cfg = CentroidRealsCfg(explicit_bool=True).parse_args()
    main(cfg)
