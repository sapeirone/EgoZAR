import os

import torch
import clip
import argparse

from torch.utils.data import DataLoader

from tqdm import tqdm
from einops import rearrange

from dataset import EK100Dataset


@torch.no_grad()
def save_feat(loader, device: torch.device = torch.device("cuda")):
    features = []

    for images, *_ in tqdm(loader):
        images = images.to(device)

        bs = images.shape[0]
        images = rearrange(images, "bs n ch h w -> (bs n) ch h w")
        images = clip_model.encode_image(images).float()
        images = rearrange(images, "(bs n) h -> bs n h", bs=bs)
        features += images.cpu().unbind(0)

    return torch.stack(features)


if __name__ == "__main__":
    args = argparse.ArgumentParser("CLIP features extraction")
    args.add_argument("--clip-model", default="ViT-B/32", choices=["RN50", "RN101", "ViT-B/32", "ViT-B/16", "ViT-L/14"], required=True)
    args.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = args.parse_args()

    # Load CLIP model
    clip_model, preprocess = clip.load(args.clip_model, device=args.device)

    # Load dataset and dataloaders
    train_dataset = EK100Dataset("source", "train", 'RGB', preprocess, n=5, num_segments=25, return_frames=True)
    val_dataset = EK100Dataset("target", "val", 'RGB', preprocess, n=5, num_segments=25, return_frames=True)

    train_loader = DataLoader(train_dataset, 128, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True, drop_last=False)
    val_loader = DataLoader(val_dataset, 128, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True, drop_last=False)

    prefix = args.clip_model.replace("/", "_")
    os.makedirs("clip_features", exist_ok=True)
    torch.save(save_feat(train_loader, args.device), f"clip_features/{prefix}_source_train.pth")
    torch.save(save_feat(val_loader, args.device), f"clip_features/{prefix}_target_val.pth")
