"""Run CAMUS segmentation and save one NIfTI per patient/view sequence."""
import argparse
import json
from pathlib import Path
from utils.config import DEFAULT_CONFIG, load_config, checkpoint_paths


def denoise(model, scheduler, features, mode):
    import torch
    batch, frames = features.shape[:2]
    if mode == "idm":
        features = features.flatten(0, 1).unsqueeze(1)
    sample = torch.randn_like(features)
    indicator = torch.full(sample.shape[:2], mode == "idm", device=sample.device,
                           dtype=torch.bool)
    for timestep in scheduler.timesteps:
        prediction = model(torch.cat([sample, features], dim=2),
                           timestep.to(sample.device),
                           image_only_indicator=indicator).sample
        sample = scheduler.step(prediction, timestep, sample).prev_sample
    return sample.reshape(batch, frames, *sample.shape[2:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["idm", "vdm"], default="vdm")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--model-dir", type=Path)
    for name in ("vae", "idm", "vdm", "image-encoder", "scheduler"):
        parser.add_argument("--" + name + "-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = str(args.data_root)
    steps = args.steps if args.steps is not None else cfg["inference"]["steps"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["inference"]["batch_size"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    if steps < 1 or batch_size < 1:
        parser.error("Steps and batch size must be positive.")
    paths = checkpoint_paths(cfg, args.model_dir)
    for name in ("vae", "idm", "vdm", "image_encoder"):
        value = getattr(args, name + "_checkpoint")
        if value:
            paths[name] = value
    import torch
    import SimpleITK as sitk
    from diffusers import DDPMScheduler
    from torch.utils.data import DataLoader
    from data import CAMUSDataset
    from training.stages import seed_everything
    from utils.checkpoints import load_model, load_encoder
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; specify --device cpu.")
    dataset = CAMUSDataset(**cfg["data"], split=args.split, video=True,
                           random_flip=False, require_labels=False)
    targets = [args.output_dir / (case + ".nii.gz") for case, _, _ in dataset.sequences]
    if not args.overwrite and any(p.exists() for p in targets + [args.output_dir / "inference.json"]):
        raise FileExistsError("Output exists; use a new output directory or --overwrite.")
    seed_everything(seed)
    vae = load_model(paths["vae"], device).eval()
    model = load_model(paths[args.mode], device).eval()
    encoder = load_encoder(paths["image_encoder"], cfg, device).eval()
    if args.scheduler_checkpoint:
        scheduler = torch.load(args.scheduler_checkpoint, map_location="cpu", weights_only=False)
    else:
        scheduler = DDPMScheduler(**cfg["diffusion"]["scheduler"])
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("This release expects an epsilon-prediction scheduler.")
    if steps > scheduler.config.num_train_timesteps:
        parser.error("Inference steps exceed the scheduler's training timesteps.")
    scheduler.set_timesteps(steps)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=cfg["num_workers"])
    chunk = cfg["inference"]["decode_batch_size"]
    if chunk < 1:
        raise ValueError("decode_batch_size must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for images, cases in loader:
            images = images.to(device=device, dtype=torch.float32)
            batch, frames = images.shape[:2]
            features = encoder(images.flatten(0, 1))
            features = features.reshape(batch, frames, *features.shape[1:])
            latents = denoise(model, scheduler, features, args.mode).flatten(0, 1)
            predictions = []
            for start in range(0, len(latents), chunk):
                logits = vae.decode(latents[start:start + chunk]).sample
                predictions.append(logits.argmax(dim=1).cpu())
            masks = torch.cat(predictions).reshape(batch, frames, *images.shape[-2:])
            for case, mask in zip(cases, masks):
                # Prepared arrays have no physical metadata; evaluate in pixel units.
                volume = sitk.GetImageFromArray(mask.numpy().astype("uint8"))
                sitk.WriteImage(volume, str(args.output_dir / (case + ".nii.gz")))
    record = {
        "mode": args.mode, "steps": steps, "seed": seed, "batch_size": batch_size,
        "checkpoints": {key: str(paths[key].resolve()) for key in ("vae", "image_encoder", args.mode)},
        "scheduler_checkpoint": str(args.scheduler_checkpoint) if args.scheduler_checkpoint else None,
        "scheduler_config": dict(scheduler.config), "config": cfg, "split": args.split,
        "cases": [case for case, _, _ in dataset.sequences],
        "output_spacing": "unit spacing; coordinates are preprocessed pixels",
    }
    (args.output_dir / "inference.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Saved %d sequences to %s" % (len(dataset), args.output_dir))


if __name__ == "__main__":
    main()
