"""Train the CAMUS VAE, IDM, or v1 VDM."""
import argparse
from pathlib import Path
from utils.config import DEFAULT_CONFIG, load_config, checkpoint_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["all", "vae", "idm", "vdm"], default="all")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-dir", type=Path, help="Input checkpoints; defaults to output-dir/models.")
    for name in ("vae", "idm", "image-encoder"):
        parser.add_argument("--" + name + "-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, help="Override epoch count for one stage.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = str(args.data_root)
    if args.output_dir:
        cfg["output_dir"] = str(args.output_dir)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        if args.stage == "all" or args.epochs < 1:
            parser.error("--epochs requires one stage and a positive count.")
        settings = cfg["vae"]["training"] if args.stage == "vae" else cfg["diffusion"][args.stage]
        settings["num_epochs"] = args.epochs
    paths = checkpoint_paths(cfg, args.model_dir)
    for name in ("vae", "idm", "image_encoder"):
        value = getattr(args, name + "_checkpoint")
        if value:
            paths[name] = value
    from training.stages import seed_everything, train_stage
    import torch
    import yaml
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for small checks.")
    stages = ["vae", "idm", "vdm"] if args.stage == "all" else [args.stage]
    directory = Path(cfg["output_dir"]) / "models"
    output_paths = checkpoint_paths(cfg)
    for stage in stages:
        required = [] if stage == "vae" else ["vae"]
        if stage == "vdm":
            required += ["idm", "image_encoder"]
        for key in required:
            if not paths[key].is_file():
                raise FileNotFoundError("Missing %s checkpoint: %s" % (key, paths[key]))
        if (output_paths[stage].exists() or list(directory.glob("*%s_*.pth" % stage))) and not args.overwrite:
            raise FileExistsError("Stage %s checkpoints exist; use a new output directory or --overwrite." % stage)
        directory.mkdir(parents=True, exist_ok=True)
        record = {"config": cfg, "stage": stage, "input_checkpoints": {k: str(paths[k]) for k in required}}
        with (directory / ("%s_config.yaml" % stage)).open("w") as stream:
            yaml.safe_dump(record, stream, sort_keys=False)
        seed_everything(cfg["seed"])
        train_stage(cfg, stage, paths, device)
        if stage == "vae":
            paths["vae"] = output_paths["vae"]
        elif stage == "idm":
            paths["idm"], paths["image_encoder"] = output_paths["idm"], output_paths["image_encoder"]


if __name__ == "__main__":
    main()
