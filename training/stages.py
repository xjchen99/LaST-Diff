"""VAE, spatial diffusion, and v1 temporal fine-tuning on a single device."""
import copy
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from tqdm import tqdm
from data import CAMUSDataset
from model.st_model import UNetSpatioTemporalDiffusion
from model.image_feature_encoder import ImageFeatureEncoder
from utils.checkpoints import load_model, load_encoder
from utils.config import checkpoint_paths


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(cfg, stage):
    settings = cfg["vae"]["training"] if stage == "vae" else cfg["diffusion"][stage]
    dataset = CAMUSDataset(**cfg["data"], split="train", video=(stage == "vdm"),
                           random_flip=True, require_labels=True)
    batch_size = settings["batch_size"]
    if stage == "idm":
        batch_size *= cfg["data"]["frames"]
    if stage != "vae" and len(dataset) < batch_size:
        raise ValueError("Training data is smaller than one complete batch.")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=cfg["num_workers"], worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(cfg["seed"]),
        drop_last=(stage != "vae"), pin_memory=True,
        persistent_workers=(cfg["num_workers"] > 0),
    )


def freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def update_ema(ema, model, decay, first_update):
    # Match AveragedModel: copy the first update, then average parameters.
    for average, parameter in zip(ema.parameters(), model.parameters()):
        if first_update:
            average.copy_(parameter)
        else:
            average.mul_(decay).add_(parameter, alpha=1.0 - decay)
    for average, buffer in zip(ema.buffers(), model.buffers()):
        average.copy_(buffer)


def save_epoch(model, ema, directory, stage, epoch, encoder=None, scheduler=None):
    torch.save(model, directory / ("%s_%04d.pth" % (stage, epoch)))
    torch.save(ema, directory / ("ema_%s_%04d.pth" % (stage, epoch)))
    if encoder is not None:
        prefix = "image_encoder_vdm" if stage == "vdm" else "image_encoder"
        torch.save(encoder.state_dict(), directory / ("%s_%04d.pth" % (prefix, epoch)))
    if scheduler is not None:
        name = "vdm_scheduler.pth" if stage == "vdm" else "scheduler.pth"
        torch.save(scheduler, directory / name)


def train_stage(cfg, stage, paths, device):
    loader = make_loader(cfg, stage)
    settings = cfg["vae"]["training"] if stage == "vae" else cfg["diffusion"][stage]
    directory = Path(cfg["output_dir"]) / "models"
    directory.mkdir(parents=True, exist_ok=True)
    encoder = vae = scheduler = None
    if stage == "vae":
        model = AutoencoderKL(**cfg["vae"]["model"]).to(device)
        model.set_attn_processor(AttnProcessor2_0())
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    else:
        vae = freeze(load_model(paths["vae"], device))
        vae.set_attn_processor(AttnProcessor2_0())
        scheduler = DDPMScheduler(**cfg["diffusion"]["scheduler"])
        if stage == "idm":
            model = UNetSpatioTemporalDiffusion(**cfg["diffusion"]["model"]).to(device)
            encoder = ImageFeatureEncoder(**cfg["image_encoder"]["model"]).to(device)
            encoder.train()
            optimizer = torch.optim.Adam([
                {"params": model.parameters()},
                {"params": encoder.parameters(), "lr": cfg["image_encoder"]["lr"]},
            ], lr=settings["lr"])
        else:
            model = load_model(paths["idm"], device)
            model.requires_grad_(True)
            encoder = freeze(load_encoder(paths["image_encoder"], cfg, device))
            optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
        model.set_attn_processor(AttnProcessor2_0())
    model.train()
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    first_update = True
    for epoch in range(settings["num_epochs"]):
        progress = tqdm(loader, desc="%s epoch %d/%d" % (stage, epoch + 1, settings["num_epochs"]))
        for images, masks in progress:
            masks = masks.to(device, dtype=torch.float32, non_blocking=True)
            if stage == "vae":
                distribution = model.encode(masks).latent_dist
                latents = distribution.sample()
                logits = model.decode(latents).sample
                loss = nn.functional.cross_entropy(logits, masks.squeeze(1).long())
                loss = loss + settings["beta"] * distribution.kl().mean()
            else:
                images = images.to(device, dtype=torch.float32, non_blocking=True)
                if stage == "idm":
                    batch = images.shape[0] // cfg["data"]["frames"]
                    frames = cfg["data"]["frames"]
                    flat_images, flat_masks = images, masks
                else:
                    batch, frames = images.shape[:2]
                    flat_images, flat_masks = images.flatten(0, 1), masks.flatten(0, 1)
                with torch.no_grad():
                    mask_latents = vae.encode(flat_masks).latent_dist.sample()
                if stage == "vdm":
                    with torch.no_grad():
                        features = encoder(flat_images)
                else:
                    features = encoder(flat_images)
                if features.shape != mask_latents.shape:
                    raise ValueError("Image features and label latents have different shapes.")
                mask_latents = mask_latents.reshape(batch, frames, *mask_latents.shape[1:])
                features = features.reshape(batch, frames, *features.shape[1:])
                noise = torch.randn_like(mask_latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                          (batch,), device=device, dtype=torch.long)
                noisy = scheduler.add_noise(mask_latents, noise, timesteps)
                indicator = torch.full((batch, frames), stage == "idm",
                                       device=device, dtype=torch.bool)
                prediction = model(torch.cat([noisy, features], dim=2), timesteps,
                                   image_only_indicator=indicator).sample
                loss = nn.functional.mse_loss(prediction, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # The original VAE script clipped before backward; clip actual gradients.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema, model, settings["ema_decay"], first_update)
            first_update = False
            progress.set_postfix(loss=float(loss.detach()))
        if ((epoch + 1) % settings["save_interval"] == 0
                or epoch + 1 == settings["num_epochs"]):
            save_epoch(model, ema, directory, stage, epoch, encoder, scheduler)
    # Export the final EMA under the same filename used by released weights.
    final_paths = checkpoint_paths(cfg)
    torch.save(ema, final_paths[stage])
    if encoder is not None:
        torch.save(encoder.state_dict(), final_paths["image_encoder"])
