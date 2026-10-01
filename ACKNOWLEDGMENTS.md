# Acknowledgments

The implementation derives from STeP by Bingliang Zhang, Zihui Wu, Berthy T. Feng, Yang Song, Yisong Yue, and Katherine L. Bouman. The spatiotemporal model class is retained from the research source with its original module path for checkpoint compatibility.

- Upstream repository: https://github.com/zhangbingliang2019/STeP
- Upstream work: *STeP: A General and Scalable Framework for Solving Video Inverse Problems with Spatiotemporal Diffusion Priors*, arXiv:2504.07549 (2025).

```bibtex
@misc{zhang2025stepgeneralscalableframework,
  title={STeP: A General and Scalable Framework for Solving Video Inverse Problems with Spatiotemporal Diffusion Priors},
  author={Bingliang Zhang and Zihui Wu and Berthy T. Feng and Yang Song and Yisong Yue and Katherine L. Bouman},
  year={2025},
  eprint={2504.07549},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2504.07549}
}
```

The model uses [Hugging Face Diffusers](https://github.com/huggingface/diffusers) and [PyTorch](https://github.com/pytorch/pytorch). CAMUS data and split files should be obtained from their original distribution and used under the dataset's terms.
