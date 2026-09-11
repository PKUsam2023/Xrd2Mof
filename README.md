# Xrd2Mof

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

**Title.** *Interpreting X-Ray Diffraction Patterns of Metal-Organic Frameworks via Generative Artificial Intelligence*
**Authors.** Bin Feng1#, Bingxu Wang1#, Linpeng Lv1, Mingzheng Zhang1, Zhefeng Chen1, Feng Pan1*, Shunning Li1*

---

## Table of Contents

* [Introduction](#introduction)
* [Model Architecture](#model-architecture)
* [Getting Started](#getting-started)

  * [Obtaining the Code](#obtaining-the-code)
  * [Prerequisites](#prerequisites)
  * [Data Processing](#data-processing)
  * [Feature Extraction](#feature-extraction)
  * [Coarse-Grained Structure Generation](#coarse-grained-structure-generation)
  * [Atomic MOF Assembly](#atomic-mof-assembly)
* [License](#license)
* [Acknowledgements](#acknowledgements)
* [Citation](#citation)

---

## Introduction

**Xrd2Mof** present a crystal structure generation framework based on the Stable Diffusion architecture designed to enable intelligent, high-throughput interpretation of powder X-ray diffraction (PXRD) patterns and subsequent reconstruction of atomic-level crystal structures for metal–organic frameworks (MOFs). It leverages pretrained models to extract salient features from PXRD spectra alongside essential prior information, facilitating the generation of structural building-block sites within the unit cell. Using experimental MOF structures from the Cambridge Structural Database (CSD) as the training dataset, and experimentally obtained PXRD patterns as test cases, we demonstrate that Xrd2Mof effectively captures spectral features, integrates physicochemical information from metal nodes and organic linkers, and subsequently generates the critical structural sites necessary for assembling complete atomic structures. By employing a coarse-grained(CG) strategy, Xrd2Mof effectively overcomes the limitations traditionally imposed by atom count in crystal structure prediction (CSP) tasks, thus enabling its broad applicability across diverse MOF domains. The proposed framework offers a novel technological route toward automated structural determination in high-throughput MOF experiments, significantly advancing the application of machine learning methods within analytical and structural chemistry.
> **Keywords:** metal–organic frameworks (MOFs), crystal structure prediction (CSP), powder X-ray diffraction (PXRD), interpretable ML.

---

## Model Architecture

A schematic of the overall Xrd2Mof pipeline:

![Model Overview](fig/Overview.png)
![Demo](fig/Demo.png)

Further details are provided in the accompanying [paper](https://doi.org/10.1021/jacs.5c16416).

---

## Getting Started

### Obtaining the Code

There are two ways to obtain this repository, and they produce **different directory names**. This matters because every command in this README uses paths relative to
that directory.

```bash
# Option 1 — clone (recommended)
git clone https://github.com/PKUsam2023/Xrd2Mof.git
# creates:  Xrd2Mof/

# Option 2 — download the ZIP from the GitHub web interface
# extracting it creates:  Xrd2Mof-master/
```

GitHub appends the branch name to ZIP downloads, which is where the `-master` suffix comes from. A ZIP download also has no `.git/` directory, so you cannot commit, pull updates, or contribute changes back — use `git clone` unless you have a reason not to.

> [!NOTE]
> The commands throughout this README assume the ZIP layout (`Xrd2Mof-master/`).
> If you cloned the repository instead, either rename the directory:
>
> ```bash
> mv Xrd2Mof Xrd2Mof-master
> ```
>
> or replace `Xrd2Mof-master` with `Xrd2Mof` in the commands below. The same applies > to `raw_path` in `pretrained_model/conf/config.yaml` and to the `PYTHONPATH` value
> used during generation.

### Prerequisites

Tested software versions:
```
Python>=3.8.5
torch>=1.13.1
numpy>=1.24.4
scikit-learn>=1.3.2
matplotlib>=3.7.5
pymatgen==2023.8.10
```

Install dependencies:

```bash
conda env create -f environment.yml
conda activate Xrd2Mof
```
> [!IMPORTANT]
> Two dependency groups must **not** be installed with a plain `pip install`.
> Handle them separately as described below, otherwise the environment will break.

**1. PyTorch Geometric extensions (`torch-scatter`, `torch-sparse`, `torch-geometric`)**

These are distributed as pre-built wheels from the PyG index. Installing them from PyPI triggers a local C++/CUDA compilation that is slow and frequently fails. Always pass the matching wheel index:

```bash
# check your torch build first
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# then install with the matching index (example: torch 1.13.1, CPU-only)
pip install torch-scatter torch-sparse torch-geometric \
  -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
```

Replace `1.13.1+cpu` with your own build string (e.g. `1.13.1+cu117` for CUDA 11.7). The suffix must match exactly, or pip silently falls back to source compilation.

**2. `smact` and `pyxtal`**

Current releases of both packages require pymatgen 2024 or newer, and installing them normally will upgrade the pinned `pymatgen==2023.8.10` that the DiffCSP-based generation model depends on. Install them without dependency resolution:

```bash
pip install "smact==2.5.5" --no-deps
pip install "pyxtal==0.6.1" --no-deps
```

Then confirm that pymatgen was left untouched:

```bash
pip list | grep -E "pymatgen|smact|pyxtal"
```

We recommend the Anaconda distribution (Windows/macOS/Linux). Installation has been validated using the standard instructions from each provider.

**Optional data:** Preprocessed databases `linker_valence_table.pt` and `valid_bbs_space_408626.pt` are available on Zenodo.
For full data access, please contact: **[chenwei_wu@stu.pku.edu.cn](mailto:chenwei_wu@stu.pku.edu.cn)**.

---

### Data Processing

Create a test dataset following the steps in [`create_data/README.md`](create_data/README.md).

### Feature Extraction

Extract dataset features as described in [`pretrained_model/README.md`](pretrained_model/README.md).

### Coarse-Grained Structure Generation

Generate CG crystal structures following [`generation_model/README.md`](generation_model/README.md).

### Atomic MOF Assembly

Assemble atomic-level structures from CG predictions via [`assemble/README.md`](assemble/README.md).

---

## License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This codebase builds upon the following open-source projects and tools:

* [DiffCSP](https://github.com/jiaor17/DiffCSP)
* [MOFDiff](https://github.com/microsoft/MOFDiff)
* [Pymatgen](https://github.com/materialsproject/pymatgen)
* [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
* [PyTorch](https://github.com/pytorch/pytorch)
* [Lightning](https://github.com/Lightning-AI/pytorch-lightning/)
* [Hydra](https://github.com/facebookresearch/hydra)

We thank the authors and communities of these projects.

---

## Citation

If you use this repository or the pretrained models in your research, please cite:

**Bin Feng**, **Bingxu Wang**, L Lv, Mingzheng Zhang, Feng Pan* and Shunning Li*. Interpreting X-Ray Diffraction Patterns of Metal-Organic Frameworks via Generative Artificial Intelligence. 

```bibtex
@article{feng2026xrd2mof,
  title   = {Interpreting {X}-ray Diffraction Patterns of {M}etal-{O}rganic Frameworks via Generative Artificial Intelligence},
  author  = {Feng, Bin and Wang, Bingxu and Lv, Linpeng and Zhang, Mingzheng and Chen, Zhefeng and Pan, Feng and Li, Shunning},
  journal = {Journal of the American Chemical Society},
  volume  = {148},
  number  = {1},
  pages   = {869--878},
  year    = {2026},
  doi     = {10.1021/jacs.5c16416}
}
```
