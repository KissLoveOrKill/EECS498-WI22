# EECS 498/598: Deep Learning for Computer Vision (Winter 2022)

An unofficial self-study collection for the University of Michigan's EECS 498.008 / 598.008 course. It brings together assignment notebooks, Python implementations, Chinese study notes, and a Windows setup script.

> **Study use:** This repository contains completed assignment implementations and notebooks. Use them to study and compare approaches; do not submit this work as your own. Follow the course's collaboration and academic-integrity rules.

This project is not affiliated with or endorsed by the University of Michigan or the course staff. Original course instructions and third-party materials retain their respective terms. See [Sources and rights](#sources-and-rights) before redistributing material.

## Course material

| Folder | Topics |
| --- | --- |
| `A1/` | PyTorch fundamentals and k-nearest-neighbor classification |
| `A2/` | SVM and softmax classifiers, two-layer networks, and an MNIST challenge |
| `A3/` | Fully connected networks, optimization, and convolutional neural networks |
| `A4/` | Object detection, including one-stage and two-stage detectors |
| `A5/` | RNN/LSTM image captioning, attention, Transformers, and BERT study notes |
| `A6/` | GANs, VAEs, style transfer, network visualization, and related vision exercises |
| `assignment3/` | Additional Transformer captioning, self-supervised learning, CLIP/DINO, and diffusion exercises |

Most assignment folders include a notebook, its supporting Python modules, and the course helper code. Chinese task checklists are collected in [TASKS_ZH.md](TASKS_ZH.md); topic-specific explanations are in the `*_学习总结.md` files. The additional material under `assignment3/` keeps its upstream folder name and is separate from the course's `A1/`–`A6/` sequence.

## Setup (Windows)

Python 3.10–3.12 is supported by the setup script. The pinned environment was tested with Python 3.12, PyTorch 2.9.1, and torchvision 0.24.1.

From PowerShell at the repository root, create the environment and install the dependencies:

```powershell
.\setup.ps1
```

For a CPU-only installation:

```powershell
.\setup.ps1 -Cpu
```

Activate the environment according to how it was created:

```powershell
conda activate .\.venv
```

or, for a standard Python virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then start JupyterLab:

```powershell
jupyter lab
```

Choose the `Python (EECS 498 WI22)` kernel. Open each notebook with its own assignment folder as the working directory because the notebooks use relative paths. Verify the environment with `python scripts\check_setup.py` after activation.

The default setup script tries the PyTorch CUDA 12.8 wheel index; `-Cpu` uses the regular package index. CUDA availability depends on the installed driver and hardware. CPU runs are supported, but training-heavy notebooks may need smaller sample counts or longer runtimes.

## Datasets and generated files

Datasets and model checkpoints are intentionally excluded from Git. CIFAR-10, MNIST, and Pascal VOC can be downloaded by the corresponding assignment helpers. Other exercises may require their own datasets or pretrained weights; check the notebook and helper documentation before running them.

`A4/mAP` is an upstream Git submodule used by the object-detection evaluation helper. Clone this repository with `git clone --recurse-submodules https://github.com/KissLoveOrKill/EECS498-WI22.git`, or run `git submodule update --init --recursive` after a regular clone. Local evaluator inputs and results are not published.

The rebuilt COCO captioning file used locally for A5 is not included. The original course download is unavailable, and this repository currently does not include the dataset-building script. [A5/DATASETS.md](A5/DATASETS.md) records the local dataset's source and preprocessing details. The arithmetic dataset `A5/two_digit_op.json` is included.

Notebook checkpoints, local browser logs, and generated `output/` files are also ignored. Results or figures that are part of the learning materials remain in their assignment folders.

## Sources and rights

- [Official EECS 498/598 course page](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/)
- The repository does not include a top-level license. That does not grant permission to reuse or redistribute course or third-party material; check the original source's terms.
- Dataset and pretrained-weight rights belong to their respective providers. The `A4/mAP` submodule retains its upstream license and attribution.

If you find an attribution or dataset-provenance issue, please check the relevant assignment documentation before reusing that material.
