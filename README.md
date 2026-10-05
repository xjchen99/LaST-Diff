# LaST-Diff

Code for **LaST-Diff: Latent Spatiotemporal Diffusion for Temporally Stable Echocardiography Video Segmentation**.

[Paper](https://papers.miccai.org/miccai-2026-sat/DGM4MICCAI_028.html) | [BibTeX](CITATION.bib)

**Pretrained weights are coming soon.**

## Installation

Use Python 3.8 and a PyTorch build compatible with your GPU. Run commands from the repository root.

```bash
conda create -n last-diff python=3.8
conda activate last-diff
pip install -r requirements.txt
```

## Data preparation

Download CAMUS and its split files; see the [CAMUS paper](https://doi.org/10.1109/TMI.2019.2900516) (Leclerc et al., 2019).

Place paired half sequences under each patient directory, using filenames such as `patient0001_2CH_half_sequence.nii.gz` and `patient0001_2CH_half_sequence_gt.nii.gz`, for both 2CH and 4CH views.

```bash
python preprocess.py --raw-root /path/to/half_sequences \
  --split-dir /path/to/camus/database_split --output-dir data/camus
```

Use an empty output directory. Preparation produces normalized 16-frame sequences at 256 x 256 pixels. Training and validation patients are combined; test patients remain held out. Add `--images-only` when preparing unlabeled inference data.

## Training

```bash
python train.py --stage all --data-root data/camus --output-dir outputs/camus
```

This trains VAE, IDM, and VDM sequentially. Use `--stage vae`, `--stage idm`, or `--stage vdm` for individual stages. Settings are in [configs/CAMUS.yaml](configs/CAMUS.yaml).

## Inference

Place released weights in `weights/`: `vae.pth`, `image_encoder.pth`, and `vdm.pth` (or `idm.pth` for image-only inference). For models trained locally, use `--model-dir outputs/camus/models`.

```bash
python infer.py --mode vdm --data-root data/camus --model-dir weights \
  --output-dir outputs/predictions --steps 20 --seed 37
```

Use `--mode idm` for image-only inference. Predictions are saved as one NIfTI file per patient/view sequence.

## Evaluation

Compute Dice and temporal errors for tDSC, TCS, and ACD:

```bash
python evaluate.py --pred-dir outputs/predictions --data-root data/camus \
  --output-dir outputs/metrics
```

Results are saved as CSV files and a JSON summary. Add `--case-list /path/to/test_poor.txt` to evaluate the poor-quality subset.

## Citation

If you find LaST-Diff helpful in your research, please cite the [paper](https://papers.miccai.org/miccai-2026-sat/DGM4MICCAI_028.html) using [CITATION.bib](CITATION.bib).

## Acknowledgments

Many thanks to [STeP](https://github.com/zhangbingliang2019/STeP)! See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
